from functools import partial

import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_common.common.sum_utils import kahan_sum
from dsa2000_fm.array_layout.fast_psf_evaluation import compute_psf

tfpd = tfp.distributions


def sparse_annulus(key, inner_radius, outer_radius, dtype, num_samples):
    def _single_sample(key):
        D = 2
        direction_key, radii_key = jax.random.split(key, 2)
        direction = jax.random.normal(direction_key, shape=(D,), dtype=dtype)
        direction = direction / jnp.linalg.norm(direction)
        t = jax.random.uniform(radii_key, dtype=dtype, minval=(inner_radius / outer_radius) ** D,
                               maxval=1.) ** jnp.asarray(1. / D, dtype)
        u_circ = direction * t
        u = outer_radius * u_circ
        return u.astype(dtype)

    return jax.vmap(_single_sample)(jax.random.split(key, num_samples))


def dense_annulus(inner_radius, outer_radius, dl, frac, dtype):
    lvec = mvec = np.arange(-outer_radius, outer_radius, dl)
    L, M = np.meshgrid(lvec, mvec, indexing='ij')
    L = L.flatten()
    M = M.flatten()
    LM = L ** 2 + M ** 2
    _lm = np.stack([L, M], axis=-1)
    keep = np.logical_and(np.sqrt(LM) >= inner_radius, np.sqrt(LM) < outer_radius)
    _lm = _lm[keep]
    if frac < 1:
        select_idx = np.random.choice(_lm.shape[0], int(frac * _lm.shape[0]), replace=False)
        _lm = _lm[select_idx]
    return _lm.astype(dtype)


def create_target(key, target_array_name: str, lmn: FloatArray, freqs: au.Quantity, transit_decs: au.Quantity,
                  num_samples: int, accumulate_dtype, num_antennas: int | None = None):
    """
    Creates a target PSF distribution for the given array and parameters.

    Args:
        key: the random key
        target_array_name: the name of the target array with circular PSF
        lmn: [..., 3] the lmn coordinates to evaluate at
        freqs: [C] the frequencies
        transit_decs: [M] the declinations
        num_samples: the number of samples to take
        accumulate_dtype: the dtype to use for accumulation
        num_antennas: the number of antennas to sample from the array. If None, all antennas are used.

    Returns:
        [M, ...] the target PSF mean, [M, ...] the target PSF standard deviation
    """
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(target_array_name))
    antennas = array.get_antennas()
    array_location = array.get_array_location()
    obstime = at.Time('2022-01-01T00:00:00', scale='utc')
    antennas_enu = antennas.get_itrs(obstime=obstime, location=array_location).transform_to(
        ENU(0, 0, 1, obstime=obstime, location=array_location)
    )
    antennas_enu_xyz = quantity_to_jnp(antennas_enu.cartesian.xyz.T, 'm')
    latitude = quantity_to_jnp(array_location.geodetic.lat, 'rad')
    freqs_jax = quantity_to_jnp(freqs, 'Hz')
    transit_decs_jax = quantity_to_jnp(transit_decs, 'rad')
    return jax.block_until_ready(
        compute_target_psf_distribution(
            key=key,
            lmn=lmn,
            freqs=freqs_jax,
            latitude=latitude,
            transit_decs=transit_decs_jax,
            ideal_antennas_circular=antennas_enu_xyz,
            num_samples=num_samples,
            num_antennas=num_antennas,
            accumulate_dtype=accumulate_dtype
        )
    )


@partial(jax.jit, static_argnames=['num_samples', 'num_antennas', 'accumulate_dtype'])
def compute_target_psf_distribution(key, lmn: FloatArray, freqs: FloatArray, latitude: FloatArray,
                                    transit_decs: FloatArray, ideal_antennas_circular: FloatArray, num_samples: int,
                                    num_antennas: int | None = None, accumulate_dtype=jnp.float32):
    """
    Compute the target PSF distribution for the given parameters.

    Args:
        key: the random key
        lmn: [..., 3] the lmn coordinates to evaluate at
        freqs: [C] the frequencies in Hz
        latitude: the latitude of the array in radians
        transit_decs: the declinations in radians
        ideal_antennas_circular: [N, 3] the ideal antennas in ENU in meters
        num_samples: the number of samples to take
        num_antennas: the number of antennas to sample from the array. If None, all antennas are used.
        accumulate_dtype: the dtype to use for accumulation

    Returns:
        [M, ...] the target PSF mean, [M, ...] the target PSF standard deviation
    """
    # Elongate the array north-south to give circular PSF at DEC=0
    base_projected_array = ideal_antennas_circular.at[..., 1].divide(jnp.cos(latitude))

    def compute_target_stat_deltas(key):
        if num_antennas is not None:
            key, sub_key = jax.random.split(key)
            replace_idxs = jax.random.choice(sub_key, num_antennas, (num_antennas,), replace=False)
            ants = base_projected_array[replace_idxs]
        else:
            ants = base_projected_array
        # sample antennas, compute psf, compute deltas for mean and square
        antenna_projected_dist = tfpd.Normal(loc=0, scale=200.)
        delta = antenna_projected_dist.sample(ants.shape, key).at[:, 2].set(0.)
        antennas_enu = ants + delta
        psf = jax.vmap(
            lambda transit_dec: compute_psf(
                antennas=antennas_enu,
                lmn=lmn,
                freqs=freqs,
                latitude=latitude,
                transit_dec=transit_dec,
                with_autocorr=True,
                accumulate_dtype=accumulate_dtype
            )
        )(transit_decs)  # [M, ...]
        psf_dB = (10. * jnp.log10(psf)).astype(accumulate_dtype)
        return psf_dB, jnp.square(psf_dB)

    psf_dB_mean_init = psf_dB2_init = jnp.zeros(jnp.shape(transit_decs) + lmn.shape[:-1], accumulate_dtype)  # [M, ...]
    (psf_dB_mean, psf_dB2), _ = kahan_sum(
        compute_target_stat_deltas,
        (psf_dB_mean_init, psf_dB2_init),
        jax.random.split(key, num_samples)
    )
    psf_dB_mean /= num_samples
    psf_dB2 /= num_samples
    psf_dB_stddev = jnp.sqrt(jnp.abs(psf_dB2 - psf_dB_mean ** 2))

    return psf_dB_mean, psf_dB_stddev


@partial(jax.jit, static_argnames=['accumulate_dtype'])
def evaluate_psf(antennas_enu, lmn, latitude, freqs, transit_decs, target_psf_dB_mean, target_psf_dB_stddev,
                 accumulate_dtype):
    """
    Evaluate the PSF of the array and compare to the target PSF.

    Args:
        antennas_enu: [N, 3] the antennas in ENU
        lmn: [..., 3] the lmn coordinates to evaluate the PSF
        latitude: the latitude of the array
        freqs: [C] the frequencies in Hz
        transit_decs: [M] the declinations in radians
        target_psf_dB_mean: [M, ...] the target log PSF mean
        target_psf_dB_stddev: [M, ...] the target log PSF standard deviation

    Returns:
        quality: the negative chi squared value
    """

    @jax.vmap
    def compute_quality_at_dec(dec, target_psf_dB_mean, target_psf_dB_stddev):
        psf_dB = 10. * jnp.log10(
            compute_psf(
                antennas=antennas_enu,
                lmn=lmn,
                freqs=freqs,
                latitude=latitude,
                transit_dec=dec,
                with_autocorr=True,
                accumulate_dtype=accumulate_dtype
            )
        )
        z_scores = (psf_dB - target_psf_dB_mean) / (target_psf_dB_stddev + 1e-2)
        quality = -jnp.mean(z_scores ** 2)
        return quality

    quality = compute_quality_at_dec(transit_decs, target_psf_dB_mean, target_psf_dB_stddev)
    return jnp.mean(quality)
