import dataclasses
import os
import time as time_mod
from functools import partial

import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
from astropy import constants

from dsa2000_cal.antenna_model.utils import get_dish_model_beam_widths
from dsa2000_cal.assets.content_registry import fill_registries, NoMatchFound
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common import wgridder
from dsa2000_cal.common.fits_utils import ImageModel
from dsa2000_cal.common.fourier_utils import find_optimal_fft_size
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet


@dataclasses.dataclass(eq=False)
class DirtyImaging:
    """
    Performs imaging of visibilties using W-gridder.
    """

    # Imaging parameters

    plot_folder: str

    field_of_view: au.Quantity | None = None
    oversample_factor: float = 5.
    nthreads: int = 1
    epsilon: float = 1e-4
    convention: str = 'casa'
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
        vis = jnp.concatenate(vis, axis=0)  # [num_rows, chan, 4/1]
        weights = jnp.concatenate(weights, axis=0)  # [num_rows, chan, 4/1]
        flags = jnp.concatenate(flags, axis=0)  # [num_rows, chan, 4/1]
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
        pixel_size = (diffraction_limit_resolution / self.oversample_factor) * au.rad
        num_pixel = find_optimal_fft_size(
            int(field_of_view / pixel_size)
        )

        dl = pixel_size.to('rad').value * au.dimensionless_unscaled
        dm = pixel_size.to('rad').value * au.dimensionless_unscaled

        center_l = 0. * au.dimensionless_unscaled
        center_m = 0. * au.dimensionless_unscaled

        print(f"Center x: {center_l}, Center y: {center_m}")
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
            center_l=quantity_to_jnp(center_l),
            center_m=quantity_to_jnp(center_m)
        )  # [4/1, Nl, Nm]
        dirty_image.block_until_ready()
        dirty_image = dirty_image[:, :, None, :]  # [nl, nm, 1, coh]
        t1 = time_mod.time()
        print(f"Completed imaging in {t1 - t0:.2f} seconds.")
        image_model = ImageModel(
            phase_tracking=ms.meta.phase_tracking,
            obs_time=ms.ref_time,
            dl=dl,
            dm=dm,
            freqs=np.mean(ms.meta.freqs, keepdims=True),
            bandwidth=ms.meta.channel_width * len(ms.meta.freqs),
            coherencies=ms.meta.coherencies,
            beam_major=diffraction_limit_resolution * au.rad,
            beam_minor=diffraction_limit_resolution * au.rad,
            beam_pa=0 * au.deg,
            unit='JY/PIXEL',
            object_name='forward_model',
            image=au.Quantity(np.asarray(dirty_image), 'Jy')
        )
        # with open(f"{image_name}.json", 'w') as fp:
        #     fp.write(image_model.json(indent=2))
        image_model.save_image_to_fits(f"{image_name}.fits", overwrite=False)
        print(f"Saved FITS image to {image_name}.fits")

        # TODO: compute PSF for flux scale and fit beam

        return image_model

    @partial(jax.jit, static_argnames=['self', 'num_pixel'])
    def _image_visibilties_jax(self, uvw: jax.Array, vis: jax.Array, weights: jax.Array,
                               flags: jax.Array, freqs: jax.Array, num_pixel: int,
                               dl: jax.Array, dm: jax.Array, center_l: jax.Array, center_m: jax.Array):
        """
        Multi-channel synthesis image using a simple w-gridding algorithm.

        Args:
            uvw: [num_rows, 3]
            vis: [num_rows, num_chan, 4/1] in linear, i.e. [XX, XY, YX, YY] or [I]
            weights: [num_rows, num_chan, 4]
            flags: [num_rows, num_chan]
            freqs: [num_chan]
            num_pixel: int
            dl: dl pixel size
            dm: dm pixel size
            center_l: centre l
            center_m: centre m

        Returns:
            dirty_image: [num_pixel, num_pixel, 4/1]
        """

        if self.convention == 'casa':
            uvw = jnp.negative(uvw)  # CASA convention

        @partial(multi_vmap,
                 in_mapping="[r,c,p],[r,c,p],[r,c,p]",
                 out_mapping="[...,p]",
                 verbose=True)
        def _image(vis, weights, mask):
            dirty_image = wgridder.vis2dirty(
                uvw=uvw,
                freqs=freqs,
                vis=vis,
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
                wgt=weights,
                divide_by_n=False,  # Don't divide by n, the result is already I(l,m)/n.
                verbosity=0,
                nthreads=self.nthreads
            )  # [num_l, num_m]
            # TODO: Should actually multiply by n? Since this returns I(l,m)/n
            return dirty_image

        return _image(vis, weights, jnp.logical_not(flags))
