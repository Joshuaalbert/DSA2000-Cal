import dataclasses
import os
import time as time_mod
from functools import partial

import astropy.coordinates as ac
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
from astropy import constants
from astropy.coordinates import offset_by
from jax import lax
from jax._src.typing import SupportsDType

from dsa2000_cal.antenna_model.utils import get_dish_model_beam_widths
from dsa2000_cal.assets.content_registry import fill_registries, NoMatchFound
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common import wgridder
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.fits_utils import ImageModel
from dsa2000_cal.common.fourier_utils import find_optimal_fft_size
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet
from dsa2000_cal.source_models.corr_translation import linear_to_stokes


@dataclasses.dataclass(eq=False)
class DirtyImaging:
    """
    Runs forward modelling using a sharded data structure over devices.
    """

    # Imaging parameters

    plot_folder: str

    field_of_view: au.Quantity | None = None
    oversample_factor: float = 2.5
    nthreads: int = 1
    epsilon: float = 1e-4
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64
    verbose: bool = False
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        if self.field_of_view is not None and not self.field_of_view.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected field_of_view to be in degrees, got {self.field_of_view.unit}")

    def image(self, image_name: str, ms: MeasurementSet) -> ImageModel:
        print(f"Imaging {ms}")
        # Metrics
        t0 = time_mod.time()
        gen = ms.create_block_generator(vis=True, weights=True, flags=True)
        gen_response = None
        uvw = []
        vis = []
        weights = []
        flags = []

        while True:
            try:
                time, visibility_coords, data = gen.send(gen_response)
            except StopIteration:
                break
            uvw.append(visibility_coords.uvw)
            vis.append(data.vis)
            weights.append(data.weights)
            flags.append(data.flags)

        uvw = jnp.concatenate(uvw, axis=0)  # [num_rows, 3]
        vis = jnp.concatenate(vis, axis=0)  # [num_rows, chan, 4]
        weights = jnp.concatenate(weights, axis=0)  # [num_rows, chan, 4]
        flags = jnp.concatenate(flags, axis=0)  # [num_rows, chan]
        freqs = quantity_to_jnp(ms.meta.freqs)

        wavelengths = quantity_to_np(constants.c / ms.meta.freqs)
        if self.field_of_view is not None:
            field_of_view = self.field_of_view
        else:
            # Try to get HPFW from the actual beam
            try:
                fill_registries()
                antenna_model = array_registry.get_instance(
                    array_registry.get_match(ms.meta.array_name)).get_antenna_model()
                _freqs, _beam_widths = get_dish_model_beam_widths(antenna_model)
                field_of_view = np.max(np.interp(ms.meta.freqs, _freqs, _beam_widths))
            except NoMatchFound as e:
                print(str(e))
                print(f"Using fallback")
                field_of_view = au.Quantity(
                    1.22 * np.max(wavelengths) / np.min(quantity_to_np(ms.meta.antenna_diameters)),
                    au.rad
                )

        # Get the maximum baseline length
        min_wavelength = np.min(wavelengths)
        max_baseline = np.max(np.linalg.norm(uvw, axis=-1))

        # Number of pixels
        diffraction_limit_resolution = 1.22 * min_wavelength / max_baseline
        num_pixel = find_optimal_fft_size(
            int(self.oversample_factor * field_of_view.to('rad').value / diffraction_limit_resolution)
        )
        lon_top, lat_top = offset_by(
            lon=ms.meta.pointing.ra,
            lat=ms.meta.pointing.dec,
            posang=0 * au.deg,  # North
            distance=field_of_view / 2.
        )
        source_top = ac.ICRS(lon_top, lat_top)

        lon_east, lat_east = offset_by(
            lon=ms.meta.pointing.ra,
            lat=ms.meta.pointing.dec,
            posang=90 * au.deg,  # East -- increasing RA
            distance=field_of_view / 2.
        )
        source_east = ac.ICRS(lon_east, lat_east)

        source_centre = ac.ICRS(ms.meta.pointing.ra, ms.meta.pointing.dec)

        lmn_ref_points = icrs_to_lmn(
            sources=ac.concatenate([source_centre, source_top, source_east]).transform_to(ac.ICRS),
            time=ms.meta.times[0],
            phase_tracking=ms.meta.pointing
        )
        dl = (lmn_ref_points[2, 0] - lmn_ref_points[0, 0]) / (num_pixel / 2.)
        dm = (lmn_ref_points[1, 1] - lmn_ref_points[0, 1]) / (num_pixel / 2.)

        # TODO: check that x==l axis here, as it is -y in antenna frame
        center_x = lmn_ref_points[0, 0]
        center_y = lmn_ref_points[0, 1]

        print(f"Center x: {center_x}, Center y: {center_y}")
        print(f"Image size: {num_pixel} x {num_pixel}")
        print(f"Pixel size: {dl} x {dm}")

        dirty_image = self._image_visibilties_jax(
            uvw=uvw,
            vis=vis,
            weights=weights,
            flags=flags,
            freqs=freqs,
            num_pixel=num_pixel,
            dl=quantity_to_jnp(dl),
            dm=quantity_to_jnp(dm),
            center_l=quantity_to_jnp(center_x),
            center_m=quantity_to_jnp(center_y)
        )  # [num_pixel, num_pixel]
        dirty_image = dirty_image[:, :, None, None]  # [num_pixel, num_pixel, 1, 1]
        dirty_image.block_until_ready()
        t1 = time_mod.time()
        print(f"Completed imaging in {t1 - t0:.2f} seconds.")
        image_model = ImageModel(
            phase_tracking=ms.meta.pointing,
            obs_time=ms.ref_time,
            dl=dl,
            dm=dm,
            freqs=np.mean(ms.meta.freqs, keepdims=True),
            coherencies=['I'],
            beam_major=diffraction_limit_resolution * au.rad,
            beam_minor=diffraction_limit_resolution * au.rad,
            beam_pa=0 * au.deg,
            unit='JY/PIXEL',  # TODO: check this is correct.
            object_name='forward_model',
            image=au.Quantity(np.asarray(dirty_image), 'Jy')
        )
        with open(f"{image_name}.json", 'w') as fp:
            fp.write(image_model.json(indent=2))
        image_model.save_image_to_fits(f"{image_name}.fits", overwrite=False)
        print(f"Saved FITS image to {image_name}.fits")
        return image_model

    @partial(jax.jit, static_argnames=['self', 'num_pixel'])
    def _image_visibilties_jax(self, uvw: jax.Array, vis: jax.Array, weights: jax.Array,
                               flags: jax.Array, freqs: jax.Array, num_pixel: int,
                               dl: jax.Array, dm: jax.Array, center_l: jax.Array, center_m: jax.Array):
        """
        Multi-channel synthesis image stokes I using a simple w-gridding algorithm.

        Args:
            uvw: [num_rows, 3]
            vis: [num_rows, num_chan, 4] in linear, i.e. [XX, XY, YX, YY]
            weights: [num_rows, num_chan, 4]
            flags: [num_rows, num_chan]
            freqs: [num_chan]
            num_pixel: int
            dl: dl pixel size
            dm: dm pixel size
            center_l: centre l
            center_m: centre m

        Returns:
            dirty_image: [num_pixel, num_pixel]
        """

        # Convert linear visibilities to stokes, and create stokes I image
        def _transform_to_stokes_I(vis):
            num_rows, num_chan, _ = np.shape(vis)
            vis = lax.reshape(vis, (num_rows * num_chan, 4))

            def _single(_vis):
                return linear_to_stokes(_vis, flat_output=True)[0]

            vis_I = jax.vmap(_single)(vis)  # [num_rows * num_chan]
            return lax.reshape(vis_I, (num_rows, num_chan))

        vis_I = _transform_to_stokes_I(vis)  # [num_rows, num_chan]
        weights_I = _transform_to_stokes_I(jnp.reciprocal(weights))  # [num_rows, num_chan]

        if self.convention == 'casa':
            uvw = jnp.negative(uvw)  # CASA convention

        # Mask those needed for stokes I
        mask = jnp.bitwise_not(jnp.bitwise_or(flags[..., 0], flags[..., -1]))  # [num_rows, num_chan]

        dirty_image = wgridder.vis2dirty(
            uvw=uvw,
            freqs=freqs,
            vis=vis_I,
            npix_m=num_pixel,
            npix_l=num_pixel,
            pixsize_m=dm,
            pixsize_l=dl,
            center_m=center_m,
            center_l=center_l,
            epsilon=self.epsilon,
            do_wgridding=True,
            flip_v=False,
            mask=mask,
            wgt=weights_I,
            divide_by_n=False,  # Don't divide by n, the result is already I(l,m)/n.
            verbosity=0,
            nthreads=self.nthreads
        )  # [num_l, num_m]
        # TODO: Should actually multiply by n? Since this returns I(l,m)/n
        return dirty_image
