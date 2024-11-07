import dataclasses
import itertools
from functools import partial

import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
from astropy import constants
from jax import lax

from dsa2000_cal.antenna_model.antenna_model_utils import get_dish_model_beam_widths
from dsa2000_cal.assets.content_registry import fill_registries, NoMatchFound
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardt
from dsa2000_cal.common.corr_translation import unflatten_coherencies, flatten_coherencies
from dsa2000_cal.common.ellipse_utils import Gaussian
from dsa2000_cal.common.fourier_utils import find_optimal_fft_size
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np
from dsa2000_cal.common.array_types import FloatArray
from dsa2000_cal.common.vec_utils import kron_inv
from dsa2000_cal.common.wgridder import vis_to_image
from dsa2000_cal.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_cal.measurement_sets.measurement_set import  MeasurementSet


@dataclasses.dataclass(eq=False)
class BaseImagor:
    """
    Performs imaging (without deconvolution) of visibilties using W-gridder.

    Args:
        baseline_min: the minimum baseline length in meters, shorter baselines are flagged
        nthreads: the number of threads to use, None for all
        epsilon: the epsilon value of wgridder
        convention: the convention to use
        verbose: whether to print verbose output
        weighting: the weighting scheme to use
    """

    # Imaging parameters

    baseline_min: FloatArray = 0.  # meters
    nthreads: int | None = None
    epsilon: float = 1e-4
    convention: str = 'physical'
    verbose: bool = False
    weighting: str = 'natural'

    @staticmethod
    def get_image_parameters(ms: MeasurementSet, field_of_view: au.Quantity | None = None,
                             oversample_factor: float = 5.):
        wavelengths = quantity_to_np(constants.c / ms.meta.freqs)
        diameter = np.min(quantity_to_np(ms.meta.antenna_diameters))
        if field_of_view is not None:
            field_of_view = field_of_view
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
        antennas_itrs = ms.meta.antennas.get_itrs().cartesian.xyz.to(au.m).value.T
        antenna1, antenna2 = np.asarray(list(itertools.combinations_with_replacement(range(len(antennas_itrs)), 2)),
                                        dtype=mp_policy.index_dtype).T
        uvw = np.linalg.norm(antennas_itrs[antenna2] - antennas_itrs[antenna1], axis=-1)  # [num_baselines]
        max_baseline = np.max(uvw)

        # Number of pixels
        diffraction_limit_resolution = 1.22 * min_wavelength / max_baseline
        pixel_size = (diffraction_limit_resolution / oversample_factor) * au.rad
        num_pixel = find_optimal_fft_size(
            int(field_of_view / pixel_size)
        )

        dl = pixel_size.to('rad')
        dm = pixel_size.to('rad')

        center_l = 0. * au.rad
        center_m = 0. * au.rad

        print(f"Center x: {center_l}, Center y: {center_m}")
        print(f"Image size: {num_pixel} x {num_pixel}")
        print(f"Pixel size: {dl} x {dm}")
        return num_pixel, quantity_to_jnp(dl), quantity_to_jnp(dm), quantity_to_jnp(center_l), quantity_to_jnp(center_m)

    def image_psf(
            self,
            uvw: jax.Array, weights: jax.Array,
            flags: jax.Array, freqs: jax.Array,
            num_pixel: int, dl: jax.Array, dm: jax.Array, center_l: jax.Array, center_m: jax.Array
    ):
        """
        Image the point spread function.

        Args:
            uvw: [num_rows, 3] the uvw coordinates
            weights: [num_rows, num_chan, 4/1] the weights
            flags: [num_rows, num_chan, 4/1] the flags
            freqs: [num_chan] the frequencies
            num_pixel: int the number of pixels
            dl: jax.Array the pixel size in l
            dm: jax.Array the pixel size in m
            center_l: jax.Array the center l
            center_m: jax.Array the center m

        Returns:
            dirty_image: [num_pixel, num_pixel, 4/1] the dirty image
        """
        if np.shape(weights) != np.shape(flags):
            raise ValueError(
                f"Expected weights and flags to have the same shape, got {np.shape(weights)} and {np.shape(flags)}")
        vis = jnp.ones(np.shape(weights), dtype=mp_policy.vis_dtype)
        return self.image_visibilties(uvw, vis, weights, flags, freqs, num_pixel, dl, dm, center_l, center_m)

    def image_visibilties(self, uvw: jax.Array, vis: jax.Array, weights: jax.Array,
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

        if np.shape(weights) != np.shape(flags):
            raise ValueError(
                f"Expected weights and flags to have the same shape, got {np.shape(weights)} and {np.shape(flags)}")
        if np.shape(vis) != np.shape(weights):
            raise ValueError(
                f"Expected vis and weights to have the same shape, got {np.shape(vis)} and {np.shape(weights)}")

        # Remove auto-correlations and small baselines as desired, in m, not lambda

        baseline_flags = jnp.linalg.norm(uvw, axis=-1) <= self.baseline_min  # [num_rows]

        flags = jnp.logical_or(flags, baseline_flags[:, None, None])  # [num_rows, num_chan, coh]

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
                 in_mapping="[r,c,coh],[r,c,coh],[r,c,coh]",
                 out_mapping="[...,coh]",
                 verbose=True)
        def image_per_coh(vis, weights, mask):
            dirty_image = vis_to_image(uvw=uvw, freqs=freqs, vis=vis, pixsize_m=dm, pixsize_l=dl, center_m=center_m,
                                       center_l=center_l, npix_m=num_pixel, npix_l=num_pixel, wgt=weights, mask=mask,
                                       epsilon=self.epsilon, nthreads=self.nthreads, verbosity=0, scale_by_n=True,
                                       normalise=True)  # [num_l, num_m]
            return dirty_image

        return image_per_coh(vis, weights, jnp.logical_not(flags))


def evaluate_beam(freqs: jax.Array, times: jax.Array,
                  beam_gain_model: BaseSphericalInterpolatorGainModel, geodesic_model: BaseGeodesicModel,
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
                                        lmn_geodesic=geodesics)  # [num_sources, num_time, 1, num_freqs[, 2, 2]]
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


def fit_beam(psf, dl, dm):
    num_l, num_m = psf.shape[:2]
    lvec = (-0.5 * num_l + jnp.arange(num_l)) * dl  # [num_l]
    mvec = (-0.5 * num_m + jnp.arange(num_m)) * dm  # [num_m]
    L, M = jnp.meshgrid(lvec, mvec, indexing='ij')  # [num_l, num_m]
    lm = jnp.stack([L, M], axis=-1).reshape((-1, 2))  # [num_l, num_m, 2]
    psf = jnp.reshape(psf, (-1,))

    def residual_fn(params):
        major, minor, pos_angle = params
        g = Gaussian(x0=jnp.zeros(2), minor_fwhm=minor, major_fwhm=major, pos_angle=pos_angle,
                     total_flux=Gaussian.total_flux_from_peak(1, major_fwhm=major, minor_fwhm=minor))
        return jax.vmap(g.compute_flux_density)(lm) - psf

    solver = MultiStepLevenbergMarquardt(
        residual_fn=residual_fn,
        num_iterations=100,
        verbose=False
    )
    state, diagnostics = solver.solve(solver.create_initial_state(jnp.array([dl * 5, dm * 5, 0.])))
    major, minor, posang = state.x
    swap = minor > major
    major, minor, posang = (
        jnp.where(swap, minor, major),
        jnp.where(swap, major, minor),
        jnp.where(swap, posang + jnp.pi / 2., posang)
    )
    return major, minor, posang
