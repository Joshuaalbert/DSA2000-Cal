import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.mixed_precision_utils import mp_policy
from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_common.common.sum_utils import scan_sum
from dsa2000_common.delay_models.uvw_utils import celestial_to_cartesian, perley_icrs_from_lmn
from dsa2000_fm.array_layout.fast_psf_evaluation import compute_psf
from dsa2000_fm.systematics.ionosphere import evolve_gcrs


def compute_psf_from_gcrs(antennas_gcrs: FloatArray, ra, dec, lmn: FloatArray,
                          times: FloatArray, freqs: FloatArray,
                          with_autocorr: bool = True,
                          accumulate_dtype=jnp.float32) -> FloatArray:
    """
    Compute the point spread function of the array. Uses short cut,

    B(l,m) = (sum_i e^(-i2pi (u_i l + v_i m)))^2/N^2

    To remove auto-correlations, there are N values of 1 to subtract from N^2 values, then divide by (N-1)N
    PSF(l,m) = (N^2 B(l,m) - N)/(N-1)/N = (N B(l,m) - 1)/(N-1) where B(l,m) in [0, 1].
    Thus the amount of negative is (-1/(N-1))

    Args:
        antennas_gcrs: [N, 3] the antennas in GCRS
        lmn: [..., 3] the lmn coordinates to evaluate the PSF
        times: [T] the times of the observations, in seconds since start of the observation
        freqs: [C] the frequency in Hz of each channel
        dec: the transit DEC of the array in radians
        with_autocorr: whether to include the autocorrelation in the PSF
        accumulate_dtype: the dtype to use for accumulation

    Returns:
        psf: [...] the point spread function
    """

    def accum_over_time(t):
        antennas_gcrs_t = jax.vmap(lambda x: evolve_gcrs(x, t))(antennas_gcrs)
        antennas_gcrs_t -= antennas_gcrs_t[0]

        K_gcrs = celestial_to_cartesian(ra, dec)  # [3]
        tau = antennas_gcrs_t @ K_gcrs  # [N]

        ra_, dec_ = perley_icrs_from_lmn(lmn[..., 0], lmn[..., 1], lmn[..., 2], ra, dec)  # [...]
        k_gcrs = celestial_to_cartesian(ra_, dec_)  # [..., 3]

        def compute_psf_delta(freq):
            wavelength = mp_policy.cast_to_length(299792458. / freq)
            r = antennas_gcrs_t / wavelength
            delay = (2 * jnp.pi) * (jnp.sum(r * k_gcrs[..., None, :], axis=-1) - tau / wavelength)  # [..., N]
            delay = delay.astype(accumulate_dtype)
            voltage_beam_real = jnp.cos(delay).mean(axis=-1)  # [...]
            voltage_beam_imag = jnp.sin(delay).mean(axis=-1)  # [...]
            # voltage_beam = jax.lax.complex(jnp.cos(delay), jnp.sin(delay))  # [..., N]
            # voltage_beam = jnp.mean(voltage_beam, axis=-1)  # [...]
            power_beam = jnp.square(voltage_beam_real) + jnp.square(voltage_beam_imag)  # [...]
            power_beam *= lmn[..., 2].astype(accumulate_dtype)
            if with_autocorr:
                delta = power_beam
            else:
                N = tau.shape[-1]
                N1 = jnp.asarray(N / (N - 1), accumulate_dtype)
                N2 = jnp.asarray(-1 / (N - 1), accumulate_dtype)
                # (N * power_beam - 1)/(N-1)
                delta = N1 * power_beam + N2
            return delta

        psf_accum = jnp.zeros(np.shape(lmn)[:-1], dtype=accumulate_dtype)
        psf = scan_sum(compute_psf_delta, psf_accum, freqs)
        return psf / np.shape(freqs)[0]  # average over frequencies

    psf_accum = jnp.zeros(np.shape(lmn)[:-1], dtype=accumulate_dtype)
    psf = scan_sum(accum_over_time, psf_accum, times, unroll=1)
    return psf / np.shape(times)[0]  # average over times


def test_compute_psf_from_gcrs():
    import tensorflow_probability.substrates.jax as tfp
    tfpd = tfp.distributions
    import astropy.time as at
    import astropy.coordinates as ac
    import pylab as plt
    D = 1000
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    obstime = at.Time('2025-06-10T16:00:00', scale='utc')
    array_location = array.get_array_location()
    antennas = array.get_antennas()
    antennas_enu = quantity_to_jnp(antennas.get_itrs(obstime=obstime, location=array_location).transform_to(
        ENU(obstime=obstime, location=array_location)).cartesian.xyz.T)
    antennas_gcrs = quantity_to_jnp(antennas.get_gcrs(obstime=obstime).cartesian.xyz.T)
    lvec = mvec = np.linspace(-0.0005, 0.0005, 100)
    L, M = np.meshgrid(lvec, lvec, indexing='ij')
    N = np.sqrt(1 - L ** 2 - M ** 2)
    lmn = np.stack([L.flatten(), M.flatten(), N.flatten()], axis=-1)
    freqs = jnp.linspace(700e6, 2000e6, 1)
    latitude = array_location.lat.rad
    transit_dec = np.pi / 2

    psf = compute_psf(
        antennas=antennas_enu,
        lmn=lmn,
        freqs=freqs,
        latitude=latitude,
        transit_dec=transit_dec,
        with_autocorr=True,
        accumulate_dtype=jnp.float64
    ).reshape(L.shape)
    plt.imshow(
        10 * jnp.log10(psf),
        extent=[lvec.min(), lvec.max(), mvec.min(), mvec.max()],
        aspect='auto',
        interpolation='nearest',
        cmap='jet',
        vmin=-60,
        vmax=10 * np.log10(0.5)
    )

    plt.show()

    zenith = ENU(0, 0, 1, obstime=obstime, location=array_location).transform_to(ac.ICRS())
    psf = compute_psf_from_gcrs(
        antennas_gcrs=antennas_gcrs,
        ra=zenith.ra.rad,
        dec=transit_dec,
        lmn=lmn,
        times=jnp.arange(4) * 600,
        freqs=freqs,
        with_autocorr=True,
        accumulate_dtype=jnp.float64
    ).reshape(L.shape)
    plt.imshow(
        10 * jnp.log10(psf),
        extent=[lvec.min(), lvec.max(), mvec.min(), mvec.max()],
        aspect='auto',
        interpolation='nearest',
        cmap='jet',
        vmin=-60,
        vmax=10 * np.log10(0.5)
    )

    plt.show()
