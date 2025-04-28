from functools import partial

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.astropy_utils import get_time_of_local_meridean
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_common.common.sum_utils import scan_sum
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


def elongate_antennas(antennas: ac.EarthLocation, array_location: ac.EarthLocation,
                      ref_time: at.Time) -> ac.EarthLocation:
    """
    Elongate the target array in the north-south direction to give circular PSF at DEC=0.
    """
    # Get the array
    latitude = array_location.lat
    antennas_enu = antennas.get_itrs(obstime=ref_time, location=array_location).transform_to(
        ENU(obstime=ref_time, location=array_location)).cartesian.xyz.T
    antennas_enu[:, 1] /= np.cos(latitude)
    antennas = ENU(antennas_enu[:, 0], antennas_enu[:, 1], antennas_enu[:, 2], obstime=ref_time,
                   location=array_location).transform_to(
        ac.ITRS(obstime=ref_time, location=array_location)).earth_location
    return antennas


def create_target(
        antennas: ac.EarthLocation, array_location: ac.EarthLocation,
        lmn: FloatArray,
        freqs: au.Quantity, ref_time: at.Time, ra0: au.Quantity,
        dec0s: au.Quantity,
        num_samples: int, accumulate_dtype, num_antennas: int | None = None
):
    """
    Creates a target PSF distribution for the given array and parameters.

    Args:
        ra0:
        key: the random key
        target_array_name: the name of the target array with circular PSF
        lmn: [..., 3] the lmn coordinates to evaluate at
        freqs: [C] the frequencies
        dec0s: [M] the declinations
        num_samples: the number of samples to take
        accumulate_dtype: the dtype to use for accumulation
        num_antennas: the number of antennas to sample from the array. If None, all antennas are used.

    Returns:
        [M, ...] the target PSF mean, [M, ...] the target PSF standard deviation
    """
    freqs = quantity_to_jnp(freqs, 'Hz')
    ra0 = quantity_to_jnp(ra0, 'rad')
    dec0s = quantity_to_jnp(dec0s, 'rad')
    # move antennas arround ineast-north by conv_size
    antennas_enu0 = antennas.get_itrs(obstime=ref_time, location=array_location).transform_to(
        ENU(obstime=ref_time, location=array_location)).cartesian.xyz.T
    psf_dB_mean, psf_dB2_mean = 0, 0
    N = len(antennas)
    for _ in range(num_samples):
        delta = 200 * np.random.normal(size=(N, 3))
        delta[..., 2] = 0.
        antennas_enu = antennas_enu0 + delta * au.m
        antennas_enu = ENU(antennas_enu[:, 0], antennas_enu[:, 1], antennas_enu[:, 2], obstime=ref_time,
                           location=array_location)
        antennas_pert = antennas_enu.transform_to(ac.ITRS(obstime=ref_time, location=array_location)).earth_location
        antennas_gcrs = quantity_to_jnp(antennas_pert.get_gcrs(obstime=ref_time).cartesian.xyz.T, 'm')
        psf_dB = compute_only_psf(antennas_gcrs, lmn, freqs, ra0, dec0s, accumulate_dtype)
        psf_dB_mean += psf_dB
        psf_dB2_mean += np.square(psf_dB)

    psf_dB_mean /= num_samples
    psf_dB2_mean /= num_samples
    psf_dB_stddev = np.sqrt(np.abs(psf_dB2_mean - psf_dB_mean ** 2))
    return psf_dB_mean, psf_dB_stddev


def test_create_target():
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa1650_P279'))
    array_location = array.get_array_location()
    antennas = array.get_antennas()
    ref_time = at.Time("2025-06-10T00:00:00", format='isot', scale='utc')
    phase_tracking = ENU(0, 0, 1, obstime=ref_time, location=array_location).transform_to(ac.ICRS())
    ra0 = phase_tracking.ra
    ref_time = get_time_of_local_meridean(ac.ICRS(ra=ra0, dec=0 * au.deg), location=array_location, ref_time=ref_time)
    antennas = elongate_antennas(antennas=antennas, array_location=array_location, ref_time=ref_time)
    lm = dense_annulus(
        0.,
        5 * np.pi / 180 / 3600,
        0.25 * np.pi / 180 / 3600,
        1.,
        np.float64
    )
    n = np.sqrt(1 - lm[..., 0] ** 2 - lm[..., 1] ** 2)
    lmn = np.stack([lm[..., 0], lm[..., 1], n], axis=-1)
    target_psf_dB_mean, target_psf_dB_stddev = create_target(
        antennas=antennas,
        array_location=array_location,
        lmn=lmn,
        freqs=[1350] * au.MHz,
        ref_time=ref_time,
        ra0=ra0,
        dec0s=[0] * au.deg,
        num_samples=10,
        accumulate_dtype=jnp.float32,
        num_antennas=None
    )
    import pylab as plt
    rad2arcsec = 180 * 3600 / np.pi
    plt.scatter(lmn[..., 0].flatten() * rad2arcsec, lmn[..., 1].flatten() * rad2arcsec,
                c=target_psf_dB_mean[0, ...].flatten(), s=20)
    plt.grid()
    plt.show()


@partial(jax.jit, static_argnames=['num_samples', 'num_antennas', 'accumulate_dtype'])
def compute_target_psf_distribution(key, lmn: FloatArray, freqs: FloatArray, ra0: FloatArray, dec0s: FloatArray,
                                    ideal_antennas_uvw: FloatArray, num_samples: int,
                                    antenna_conv_size_m: FloatArray = 200.,
                                    num_antennas: int | None = None,
                                    accumulate_dtype=jnp.float32):
    """
    Compute the target PSF distribution for the given parameters.

    Args:
        ra0:
        key: the random key
        lmn: [..., 3] the lmn coordinates to evaluate at
        freqs: [C] the frequencies in Hz
        dec0s: the declinations in radians
        ideal_antennas_uvw: [N, 3] the ideal antennas in UVW in meters
        num_samples: the number of samples to take
        num_antennas: the number of antennas to sample from the array. If None, all antennas are used.
        accumulate_dtype: the dtype to use for accumulation

    Returns:
        [M, ...] the target PSF mean, [M, ...] the target PSF standard deviation
    """

    # Elongate the array north-south to give circular PSF at DEC=0
    # base_projected_array = ideal_antennas_uvw.at[..., 1].divide(jnp.cos(latitude))

    def compute_target_stat_deltas(key):
        if num_antennas is not None:
            key, sub_key = jax.random.split(key)
            replace_idxs = jax.random.choice(sub_key, num_antennas, (num_antennas,), replace=False)
            antennas_uvw = ideal_antennas_uvw[replace_idxs]
        else:
            antennas_uvw = ideal_antennas_uvw
        # sample antennas, compute psf, compute deltas for mean and square
        delta_uvw_dist = tfpd.Normal(loc=0, scale=antenna_conv_size_m)
        delta_uvw = delta_uvw_dist.sample(antennas_uvw.shape, key).at[:, 2].set(0.)
        antennas_uvw = antennas_uvw + delta_uvw
        # antennas_gcrs = gcrs_from_geometric_uvw(antennas_uvw, ra0, 0)
        psf = jax.vmap(
            lambda dec0: compute_psf(antennas=antennas_uvw, lmn=lmn, freqs=freqs, time=0, ra0=ra0,
                                     dec0=dec0, with_autocorr=True, accumulate_dtype=accumulate_dtype, already_uvw=True)
        )(dec0s)  # [M, ...]
        psf_dB = (10. * jnp.log10(psf)).astype(accumulate_dtype)
        return psf_dB, jnp.square(psf_dB)

    psf_dB_mean_init = psf_dB2_init = jnp.zeros(jnp.shape(dec0s) + lmn.shape[:-1], accumulate_dtype)  # [M, ...]
    (psf_dB_mean, psf_dB2) = scan_sum(
        compute_target_stat_deltas,
        (psf_dB_mean_init, psf_dB2_init),
        jax.random.split(key, num_samples)
    )
    psf_dB_mean /= num_samples
    psf_dB2 /= num_samples
    psf_dB_stddev = jnp.sqrt(jnp.abs(psf_dB2 - psf_dB_mean ** 2))

    return psf_dB_mean, psf_dB_stddev


@partial(jax.jit, static_argnames=['accumulate_dtype'])
def evaluate_psf(antennas_gcrs, lmn, freqs, ra0, dec0s, target_psf_dB_mean, target_psf_dB_stddev, accumulate_dtype):
    """
    Evaluate the PSF of the array and compare to the target PSF.

    Args:
        ra0:
        antennas_gcrs: [N, 3] the antennas in GCRS
        lmn: [..., 3] the lmn coordinates to evaluate the PSF
        freqs: [C] the frequencies in Hz
        dec0s: [M] the declinations in radians
        target_psf_dB_mean: [M, ...] the target log PSF mean
        target_psf_dB_stddev: [M, ...] the target log PSF standard deviation

    Returns:
        quality: the negative chi squared value
    """

    @jax.vmap
    def compute_loss_at_dec(dec, target_psf_dB_mean, target_psf_dB_stddev):
        psf_dB = 10. * jnp.log10(
            compute_psf(antennas=antennas_gcrs, lmn=lmn, freqs=freqs, time=0, ra0=ra0, dec0=dec,
                        with_autocorr=True, accumulate_dtype=accumulate_dtype)
        )
        z_scores = (psf_dB - target_psf_dB_mean) / (target_psf_dB_stddev + 1e-2)
        loss = jnp.mean(z_scores ** 2)
        return loss, psf_dB, z_scores

    loss, psf_dB, z_scores = compute_loss_at_dec(dec0s, target_psf_dB_mean, target_psf_dB_stddev)
    return jnp.mean(loss), psf_dB, z_scores


@partial(jax.jit, static_argnames=['accumulate_dtype'])
def compute_only_psf(antennas_gcrs, lmn, freqs, ra0, dec0s, accumulate_dtype):
    """
    Evaluate the PSF of the array and compare to the target PSF.

    Args:
        ra0:
        antennas_gcrs: [N, 3] the antennas in GCRS
        lmn: [..., 3] the lmn coordinates to evaluate the PSF
        freqs: [C] the frequencies in Hz
        dec0s: [M] the declinations in radians

    Returns:
        quality: the negative chi squared value
    """

    @jax.vmap
    def compute_loss_at_dec(dec):
        psf_dB = 10. * jnp.log10(
            compute_psf(antennas=antennas_gcrs, lmn=lmn, freqs=freqs, time=0, ra0=ra0, dec0=dec,
                        with_autocorr=True, accumulate_dtype=accumulate_dtype)
        )
        return psf_dB

    psf_dB = compute_loss_at_dec(dec0s)
    return psf_dB
