import dataclasses
import os
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
    field_of_view: au.Quantity

    plot_folder: str
    cache_folder: str

    oversample_factor: float = 2.5
    epsilon: float = 1e-4
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64
    verbose: bool = False

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)
        if not self.field_of_view.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected field_of_view to be in degrees, got {self.field_of_view.unit}")

    def image(self, image_name: str, ms: MeasurementSet) -> ImageModel:
        gen = ms.create_block_generator(vis=True, weights=False, flags=True)
        gen_response = None
        uvw = []
        vis = []
        flags = []

        while True:
            try:
                time, visibility_coords, data = gen.send(gen_response)
            except StopIteration:
                break
            uvw.append(visibility_coords.uvw)
            vis.append(data.vis)
            flags.append(data.flags)

        uvw = jnp.concatenate(uvw, axis=0)  # [num_rows, 3]
        vis = jnp.concatenate(vis, axis=0)  # [num_rows, chan, 4]
        flags = jnp.concatenate(flags, axis=0)  # [num_rows, chan]
        freqs = quantity_to_jnp(ms.meta.freqs)

        # Get the maximum baseline length
        max_baseline = np.max(np.linalg.norm(uvw, axis=-1))
        # minimum wavelength
        min_wavelength = quantity_to_np(np.min(constants.c / ms.meta.freqs))
        # Number of pixels
        diffraction_limit_resolution = 1.22 * min_wavelength / max_baseline
        num_pixel = find_optimal_fft_size(
            int(self.oversample_factor * self.field_of_view.to('rad').value / diffraction_limit_resolution)
        )
        lon_top, lat_top = offset_by(
            lon=ms.meta.phase_tracking.ra,
            lat=ms.meta.phase_tracking.dec,
            posang=0 * au.deg,  # North
            distance=self.field_of_view / 2.
        )
        source_top = ac.ICRS(lon_top, lat_top)

        lon_east, lat_east = offset_by(
            lon=ms.meta.phase_tracking.ra,
            lat=ms.meta.phase_tracking.dec,
            posang=90 * au.deg,  # East -- increasing RA
            distance=self.field_of_view / 2.
        )
        source_east = ac.ICRS(lon_east, lat_east)

        source_centre = ac.ICRS(ms.meta.phase_tracking.ra, ms.meta.phase_tracking.dec)

        lmn_ref_points = icrs_to_lmn(
            sources=ac.concatenate([source_centre, source_top, source_east]).transform_to(ac.ICRS),
            time=ms.meta.times[0],
            phase_tracking=ms.meta.phase_tracking
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
            flags=flags,
            freqs=freqs,
            num_pixel=num_pixel,
            dl=quantity_to_jnp(dl),
            dm=quantity_to_jnp(dm),
            center_l=quantity_to_jnp(center_x),
            center_m=quantity_to_jnp(center_y)
        )  # [num_pixel, num_pixel]
        dirty_image = dirty_image[:, :, None, None]  # [num_pixel, num_pixel, 1, 1]
        image_model = ImageModel(
            phase_tracking=ms.meta.phase_tracking,
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
        return image_model

    @partial(jax.jit, static_argnames=['self', 'num_pixel'])
    def _image_visibilties_jax(self, uvw: jax.Array, vis: jax.Array, flags: jax.Array, freqs: jax.Array, num_pixel: int,
                               dl: jax.Array, dm: jax.Array, center_l: jax.Array, center_m: jax.Array):
        """
        Multi-channel synthesis image stokes I using a simple w-gridding algorithm.

        Args:
            uvw: [num_rows, 3]
            vis: [num_rows, num_chan, 4] in linear, i.e. [XX, XY, YX, YY]
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

        if self.convention == 'casa':
            uvw = jnp.negative(uvw)  # CASA convention

        # Mask those needed for stokes I
        mask = jnp.bitwise_or(flags[..., 0], flags[..., -1])  # [num_rows, num_chan]

        dirty_image = wgridder.vis2dirty(
            uvw=uvw,
            freqs=freqs,
            vis=vis_I,
            npix_x=num_pixel,
            npix_y=num_pixel,
            pixsize_x=dl,
            pixsize_y=dm,
            center_x=center_l,
            center_y=center_m,
            epsilon=self.epsilon,
            do_wgridding=True,
            mask=mask,
            divide_by_n=True,
            verbosity=0
        )  # [num_pixel, num_pixel]

        return dirty_image
