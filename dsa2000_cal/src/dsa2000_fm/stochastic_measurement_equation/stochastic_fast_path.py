import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.astropy_utils import create_spherical_spiral_grid
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.quantity_utils import quantity_to_np
from dsa2000_common.common.sum_utils import scan_sum
from dsa2000_common.delay_models.uvw_utils import geometric_uvw_from_gcrs, perley_lmn_from_icrs


@partial(jax.jit, static_argnames=['num_samples'])
def stochastic_dft_from_generalised_gains(key, generalised_gains, antenna1, antenna2, num_samples: int):
    """
    Compute the DFT of the visibilities using a stochastic approach.

    Args:
        key: JAX random key.
        generalised_gains: [N, D] the generalised gains.
        antenna1: [B] the first antenna indices.
        antenna2: [B] the second antenna indices.
        num_samples: int, number of samples to use for the stochastic DFT.

    Returns:
        [B]
    """
    N, D = np.shape(generalised_gains)[:2]

    def accum_samples(key):
        rademacher = jax.random.rademacher(key, shape=(D,))
        g = jnp.sum(generalised_gains * rademacher, axis=1)  # [N]
        g1 = g[antenna1]  # [B]
        g2 = g[antenna2]  # [B]
        return (g1 * g2.conj()).astype(jnp.complex64)  # [B]

    zero_accum = jnp.zeros(np.shape(antenna1), jnp.complex64)
    accum = scan_sum(accum_samples, zero_accum, jax.random.split(key, num_samples))
    accum /= num_samples
    return accum


@partial(jax.jit, static_argnames=[])
def generalised_gains_point_sources(antennas_gcrs, lmn, A, frequency, ra0, dec0):
    """
    Compute the generalised gains for point sources.

    Args:
        antennas_gcrs: [N, 3] the antennas in GCRS coordinates
        lmn: [D, 3] the lmn coordinates
        A: [D] the flux
        frequency: the frequency in Hz of each channel
        ra0: the right ascension of the phase center in radians
        dec0: the declination of the phase center in radians

    Returns:
        [N, D] the generalised gains
    """
    x_gcrs = geometric_uvw_from_gcrs(antennas_gcrs, ra0, dec0)  # [N, 3]
    wavelength = 299792458. / frequency
    x_gcrs /= wavelength
    phase = -2 * np.pi * (jnp.sum(x_gcrs[:, None, :] * lmn[None, :, :], axis=-1) - x_gcrs[:, 2:3])  # [N, D]
    g = jnp.sqrt(A) * jax.lax.complex(jnp.cos(phase), jnp.sin(phase))  # [N, D]
    return g


def exact_dft(antennas_gcrs, antenna1, antenna2, lmn, A, frequency, ra0, dec0):
    x_gcrs = geometric_uvw_from_gcrs(antennas_gcrs, ra0, dec0)  # [N, 3]
    uvw = x_gcrs[antenna1] - x_gcrs[antenna2]  # [B, 3]
    wavelength = 299792458. / frequency
    uvw /= wavelength

    def accum_dir(x):
        lmn, A = x  # [3], [2, 2]
        phase = -2 * np.pi * (jnp.sum(uvw * lmn, axis=-1) - uvw[:, 2])  # [B]
        dV = A * jax.lax.complex(jnp.cos(phase), jnp.sin(phase))  # [B]
        return dV.astype(jnp.complex64)

    zero_accum = jnp.zeros(np.shape(antenna1), jnp.complex64)
    accum = scan_sum(accum_dir, zero_accum, (lmn, A))
    return accum


def _test_stochastic_dft_from_generalised_gains():
    import astropy.time as at
    import astropy.units as au
    import astropy.coordinates as ac
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa1650_9P'))
    antennas = array.get_antennas()
    array_location = array.get_array_location()
    obstime = at.Time('2025-06-10T00:00:00', scale='utc')
    freq = quantity_to_np(au.Quantity(700, 'MHz'))
    antennas_gcrs = quantity_to_np(antennas.get_gcrs(obstime=obstime).cartesian.xyz.T)
    phase_center = ENU(0, 0, 1, location=array_location, obstime=obstime).transform_to(ac.ICRS())
    antenna1, antenna2 = np.asarray(list(itertools.combinations(range(np.shape(antennas)[0]), 2))).T

    D = 1000
    directions = create_spherical_spiral_grid(pointing=phase_center, num_points=D, angular_radius=1. * au.deg,
                                              inner_radius=0. * au.deg)
    lmn = np.stack(
        perley_lmn_from_icrs(directions.ra.rad, directions.dec.rad, phase_center.ra.rad, phase_center.dec.rad), axis=-1)
    A = np.ones((D,))

    plt.scatter(directions.ra, directions.dec)
    plt.xlabel("RA (deg)")
    plt.ylabel("DEC (deg)")
    plt.title(f"Source distibution: D={D}")
    plt.show()

    g = generalised_gains_point_sources(
        antennas_gcrs=antennas_gcrs,
        lmn=lmn,
        A=A,
        frequency=freq,
        ra0=phase_center.ra.rad,
        dec0=phase_center.dec.rad
    )

    V_perfect = exact_dft(antennas_gcrs, antenna1, antenna2, lmn, A, freq, phase_center.ra.rad, phase_center.dec.rad)
    m_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100]
    error_mean_array = []
    error_std_array = []
    var_mean_array = []
    var_std_array = []
    for M in m_array:
        _error_mean_array = []
        _var_mean_array = []
        for i in range(10):
            V = stochastic_dft_from_generalised_gains(jax.random.PRNGKey(i), g, antenna1, antenna2, num_samples=M)
            error = (V - V_perfect)
            error_mean = np.abs(np.mean(error))
            _error_mean_array.append(error_mean)
            var = np.var(error)
            _var_mean_array.append(var)
        error_mean_array.append(np.mean(_error_mean_array))
        error_std_array.append(np.std(_error_mean_array))
        var_mean_array.append(np.sqrt(np.mean(_var_mean_array)))
        var_std_array.append(np.std(np.sqrt(_var_mean_array)))
        print(f"Stochastic DFT rel bias for {M} samples: {error_mean_array[-1]} +- {error_std_array[-1]}")
        print(f"Stochastic DFT rel stddev for {M} samples: {var_mean_array[-1]} +- {var_std_array[-1]}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    # Plot error stats v M using error bars
    ax[0].scatter(m_array, error_mean_array, c='black')
    ax[0].errorbar(m_array,error_mean_array, error_std_array, fmt='o')
    ax[0].set_xlabel('M')
    ax[0].set_ylabel(r'Bias $\langle\mathbb{E}[\delta V]\rangle$')
    ax[0].set_title(f'Bias: D={D}, N={len(antennas)}')

    ax[1].scatter(m_array, var_mean_array, c='black')
    ax[1].set_yscale('log')
    ax[1].errorbar(m_array, var_mean_array, var_std_array, fmt='o')
    ax[1].set_xlabel('M')
    ax[1].set_ylabel(r'Std.Dev. $\sqrt{\langle\mathrm{Var}[\delta V]\rangle}$')
    ax[1].set_title(f'Std.Dev.: D={D}, N={len(antennas)}')
    plt.show()


if __name__ == '__main__':
    _test_stochastic_dft_from_generalised_gains()
