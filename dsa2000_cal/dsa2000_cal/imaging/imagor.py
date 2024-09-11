import dataclasses
import os
import time as time_mod
from functools import partial

import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
from astropy import constants
from jax import lax

from dsa2000_cal.antenna_model.antenna_model_utils import get_dish_model_beam_widths
from dsa2000_cal.assets.content_registry import fill_registries, NoMatchFound
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.corr_translation import unflatten_coherencies, flatten_coherencies
from dsa2000_cal.common.fits_utils import ImageModel
from dsa2000_cal.common.fourier_utils import find_optimal_fft_size
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np
from dsa2000_cal.common.types import mp_policy
from dsa2000_cal.common.vec_utils import kron_inv
from dsa2000_cal.common.wgridder import vis_to_image
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.geodesic_model import GeodesicModel
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet


@dataclasses.dataclass(eq=False)
class Imagor:
    """
    Performs imaging (without deconvolution) of visibilties using W-gridder.
    """

    # Imaging parameters

    plot_folder: str

    field_of_view: au.Quantity | None = None
    oversample_factor: float = 5.
    nthreads: int | None = None
    epsilon: float = 1e-4
    convention: str = 'physical'
    verbose: bool = False
    weighting: str = 'natural'
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        if self.field_of_view is not None and not self.field_of_view.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected field_of_view to be in degrees, got {self.field_of_view.unit}")

    def image(self, image_name: str, ms: MeasurementSet, psf: bool = False, overwrite: bool = False) -> ImageModel:
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
        diameter = np.min(quantity_to_np(ms.meta.antenna_diameters))
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
                print(f"Failed to get beam width from antenna model: {e}")
                field_of_view = au.Quantity(
                    1.22 * np.max(wavelengths) / diameter,
                    au.rad
                )
                print(f"Using diffraction limit: {field_of_view}")
        # D/ 4F = 1.22 wavelength / D ==> F = D^2 / (4 * 1.22 * wavelength)
        effective_focal_length = diameter ** 2 / (4 * 1.22 * np.max(wavelengths))

        print(f"Effective focal length: {effective_focal_length}")

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

        if psf:
            vis = jnp.ones_like(vis)

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
        for coh in range(dirty_image.shape[-1]):
            # plot to png
            plt.imshow(
                np.log10(np.abs(dirty_image[..., 0, coh].T)),
                origin='lower',
                extent=(-num_pixel / 2 * dl, num_pixel / 2 * dl, -num_pixel / 2 * dm, num_pixel / 2 * dm)
            )
            plt.xlabel('l [rad]')
            plt.ylabel('m [rad]')
            plt.colorbar()
            plt.title(f"{image_name} {ms.meta.coherencies[coh]}")
            plt.savefig(f"{self.plot_folder}/{image_name}_{ms.meta.coherencies[coh]}.png")
            plt.show()
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
        image_model.save_image_to_fits(f"{image_name}.fits", overwrite=overwrite)
        print(f"Saved FITS image to {image_name}.fits")

        return image_model

    def image_visibilities(self, uvw: jax.Array, vis: jax.Array, weights: jax.Array,
                           flags: jax.Array, freqs: jax.Array, num_pixel: int,
                           dl: jax.Array, dm: jax.Array, center_l: jax.Array, center_m: jax.Array) -> jax.Array:
        ...

    @partial(jax.jit, static_argnames=['self', 'num_pixel'])
    def _image_visibilties_jax(self, uvw: jax.Array, vis: jax.Array, weights: jax.Array,
                               flags: jax.Array, freqs: jax.Array, num_pixel: int,
                               dl: jax.Array, dm: jax.Array, center_l: jax.Array, center_m: jax.Array):
        """
        Multi-channel synthesis image using a simple w-gridding algorithm.

        Args:
            uvw: [num_rows, 3]
            vis: [num_rows, num_chan, 4/1] in linear, i.e. [XX, XY, YX, YY] or [I]
            weights: [num_rows, num_chan, 4/1]
            flags: [num_rows, num_chan, 4/1]
            freqs: [num_chan]
            num_pixel: int
            dl: dl pixel size
            dm: dm pixel size
            center_l: centre l
            center_m: centre m

        Returns:
            dirty_image: [num_pixel, num_pixel, 4/1]
        """

        # Remove auto-correlations
        baseline_length = jnp.linalg.norm(uvw, axis=-1)  # [num_rows]
        flags = jnp.logical_or(flags, baseline_length[:, None, None] < 1.0)  # [num_rows, num_chan, coh]

        if self.convention == 'engineering':
            uvw = jnp.negative(uvw)

        if self.weighting == 'uniform':
            @partial(multi_vmap,
                     in_mapping="[r,c,p]",
                     out_mapping="[...,c,p]",
                     verbose=True)
            def update_weights(weights):
                u_bins = jnp.linspace(jnp.min(uvw[:, 0]) - 1, jnp.max(uvw[:, 0]) + 1, 256)
                v_bins = jnp.linspace(jnp.min(uvw[:, 1]) - 1, jnp.max(uvw[:, 1]) + 1, 256)
                hist, _, _ = jnp.histogram2d(uvw[:, 0], uvw[:, 1], weights=weights,
                                             bins=[u_bins, v_bins])
                # Convolve with 3x3 avg kernel to smooth the weights
                kernel = jnp.ones((3, 3)) / 9.
                hist = jax.scipy.signal.convolve2d(hist, kernel, mode='same')
                # determine which bins each point in uvw falls into
                u_bin = jnp.digitize(uvw[:, 0], u_bins) - 1
                v_bin = jnp.digitize(uvw[:, 1], v_bins) - 1
                weights = jnp.reciprocal(hist[u_bin, v_bin])
                return weights

            weights = update_weights(weights)  # [num_rows, num_chan, 4/1]
        elif self.weighting == 'natural':
            pass

        else:
            raise ValueError(f"Unknown weighting scheme {self.weighting}")

        @partial(multi_vmap,
                 in_mapping="[r,c,p],[r,c,p],[r,c,p]",
                 out_mapping="[...,p]",
                 verbose=True)
        def image_single_coh(vis, weights, mask):
            dirty_image = vis_to_image(
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
                mask=mask,
                wgt=weights,
                verbosity=0,
                nthreads=self.nthreads,
                scale_by_n=True,
                normalise=True
            )  # [num_l, num_m]
            return dirty_image

        return image_single_coh(vis, weights, jnp.logical_not(flags))


def evaluate_beam(freqs: jax.Array, times: jax.Array,
                  beam_gain_model: GainModel, geodesic_model: GeodesicModel,
                  num_l: int, num_m: int,
                  dl: jax.Array, dm: jax.Array, center_l: jax.Array, center_m: jax.Array) -> jax.Array:
    """
    Evaluate the beam at a grid of lmn points.

    Args:
        freqs: [num_freqs] the frequency values
        times: [num_time] the time values
        beam_gain_model: the beam gain model
        geodesic_model: the geodesic model
        num_l: the number of l points
        num_m: the number of m points
        dl: the l spacing
        dm: the m spacing
        center_l: the center l
        center_m: the center m

    Returns:
        beam: [num_l, num_m, num_time, num_freqs[, 2, 2]]
    """
    lvec = (-0.5 * num_l + jnp.arange(num_l)) * dl + center_l  # [num_l]
    mvec = (-0.5 * num_m + jnp.arange(num_m)) * dm + center_m  # [num_m]
    l, m = jnp.meshgrid(lvec, mvec, indexing='ij')  # [num_l, num_m]
    n = jnp.sqrt(1. - (jnp.square(l) + jnp.square(m)))  # [num_l, num_m]
    lmn_sources = mp_policy.cast_to_angle(jnp.reshape(jnp.stack([l, m, n], axis=-1), (-1, 3)))  # [num_sources, 3]
    if not beam_gain_model.tile_antennas:
        raise ValueError("Beam gain model must be identical for all antennas.")
    geodesics = geodesic_model.compute_far_field_geodesic(
        times=times, lmn_sources=lmn_sources, antenna_indices=jnp.asarray([0], mp_policy.index_dtype)
    )  # [num_sources, num_time, 1, 3]
    # jax.debug.print("geodesics={geodesics}", geodesics=geodesics)
    beam = beam_gain_model.compute_gain(freqs=freqs, times=times,
                                        geodesics=geodesics)  # [num_sources, num_time, 1, num_freqs[, 2, 2]]
    beam = beam[:, :, 0, ...]  # [num_sources, num_time, num_freqs[, 2, 2]]
    res_shape = (num_l, num_m) + beam.shape[1:]
    return lax.reshape(beam, res_shape)  # [num_l, num_m, num_time, num_freqs[, 2, 2]]


def divide_out_beam(image: jax.Array, beam: jax.Array
                    ) -> jax.Array:
    """
    Divide out the beam from the image.

    Args:
        image: [num_pixel, num_pixel, 4/1]
        beam: [num_pixel, num_pixel[,2,2]]

    Returns:
        image: [num_pixel, num_pixel, 4/1]
    """

    def _remove(image, beam):
        if (np.shape(image) == ()) or (np.shape(image) == (1,)):
            if np.shape(beam) != ():
                raise ValueError(f"Expected beam to be scalar, got {np.shape(beam)}")
            return image / beam
        elif np.shape(image) == (4,):
            if np.shape(beam) != (2, 2):
                raise ValueError(f"Expected beam to be full-stokes.")
            return flatten_coherencies(kron_inv(beam, unflatten_coherencies(image), beam.T.conj()))
        elif np.shape(image) == (2, 2):
            if np.shape(beam) != (2, 2):
                raise ValueError(f"Expected beam to be full-stokes.")
            return kron_inv(beam, image, beam.T.conj())
        else:
            raise ValueError(f"Unknown image shape {np.shape(image)} and beam shape {np.shape(beam)}.")

    pb_cor_image = jax.vmap(jax.vmap(_remove))(image, beam)
    return jnp.where(jnp.isnan(pb_cor_image), 0., pb_cor_image)
