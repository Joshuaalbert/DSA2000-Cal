import jax
import numpy as np
import pylab as plt
from jax import numpy as jnp
from matplotlib import pyplot as plt

from dsa2000_fm.array_layout.psf_evaluation import compute_fwhm, compute_ideal_psf, find_sigma, \
    sample_projected_antennas, compute_psf, pareto_front


def test_compute_fwhm():
    wavelength = 0.21
    R_array = np.linspace(8e3, 10e3, 10)
    sigma_array = np.linspace(1e2, 5e3, 10)
    s_array = np.zeros((len(R_array), len(sigma_array)))
    for i, R in enumerate(R_array):
        for j, sigma in enumerate(sigma_array):
            s = compute_fwhm(R / wavelength, sigma / wavelength)
            s_arcsec = s * 180 / np.pi * 3600
            # if 3.25 < s_arcsec < 3.5:
            print(f"R={R}, sigma={sigma}, s={s * 180 / np.pi * 3600:.2f} arcsec")
            s_array[i, j] = s_arcsec
    plt.imshow(s_array, origin='lower', extent=(sigma_array[0], sigma_array[-1], R_array[0], R_array[-1]),
               interpolation='nearest')
    plt.xlabel('sigma')
    plt.ylabel('R')
    plt.colorbar()
    plt.show()


def test_compute_ideal_psf():
    wavelength = 0.21
    s_array = np.linspace(0, 20, 100) * np.pi / 180 / 3600
    F = np.zeros_like(s_array)
    for i, s in enumerate(s_array):
        F[i] = compute_ideal_psf(s, 10e3 / wavelength, 11.6e3 / wavelength)
    plt.plot(s_array * 180 / np.pi * 3600, F)
    plt.xlabel('s (arcsec)')
    plt.ylabel('F(s)')
    plt.show()


def test_find_sigma():
    wavelength = 0.21
    R = 10e3
    Rmin = 8
    s_fwhm = 3.33 * np.pi / 180 / 3600
    sigma = find_sigma(R / wavelength, s_fwhm, Rmin=Rmin / wavelength)
    print(sigma)

    s_array = np.linspace(0, 20, 100) * np.pi / 180 / 3600
    F = np.zeros_like(s_array)
    for i, s in enumerate(s_array):
        F[i] = compute_ideal_psf(s, R / wavelength, sigma / wavelength, Rmin=Rmin / wavelength)
    plt.plot(s_array * 180 / np.pi * 3600, F)
    plt.xlabel('s (arcsec)')
    plt.ylabel('F(s)')
    plt.show()


def test_sample_projected_antennas():
    key = jax.random.PRNGKey(0)
    R = 10e3
    Rmin = 8
    sigma = 36193
    num_antennas = 2048
    projected_positions = sample_projected_antennas(key, R, sigma, num_antennas)
    plt.scatter(projected_positions[:, 0], projected_positions[:, 1])
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.show()


def test_compute_psf():
    key = jax.random.PRNGKey(1)
    R = 10e3
    Rmin = 8
    sigma = 36193
    num_antennas = 2048
    projected_positions = sample_projected_antennas(key, R, sigma, num_antennas)

    # out to 1 arcmin
    mvec = lvec = np.linspace(-1 / 60, 1 / 60, 256) * np.pi / 180
    L, M = np.meshgrid(lvec, mvec, indexing='ij')
    N = jnp.sqrt(1 - L ** 2 - M ** 2)
    lmn = jnp.stack([L, M, N], axis=-1)
    psf = compute_psf(
        antennas=projected_positions,
        lmn=lmn,
        freq=1350e6,
        latitude=0,
        transit_dec=0,
        with_autocorr=True
    )
    psf = 10 * jnp.log10(psf)
    plt.imshow(psf.T, origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]), interpolation='nearest',
               vmin=-60, vmax=10 * np.log10(0.5))
    plt.xlabel('l')
    plt.ylabel('m')
    plt.colorbar()
    plt.show()

    # plot profile

    lvec = np.linspace(0, 1 / 60, 1024) * np.pi / 180
    m = jnp.zeros_like(lvec)
    n = jnp.sqrt(1 - lvec ** 2)
    lmn = jnp.stack([lvec, m, n], axis=-1)
    psf = compute_psf(
        antennas=projected_positions,
        lmn=lmn,
        freq=1350e6,
        latitude=0,
        transit_dec=0,
        with_autocorr=True
    )
    psf = 10 * jnp.log10(psf)
    plt.plot(lvec * 180 / np.pi * 60, psf)
    plt.xlabel('l (arcmin)')
    plt.ylabel('PSF (dB)')
    plt.show()


def test_pareto_front():
    np.random.seed(42)
    points = np.random.normal(size=(1000, 2)) - 5  # Generate random (x, y) points in [-5, 5]

    pareto_points_idxs = pareto_front(points)
    # plot
    plt.scatter(points[:, 0], points[:, 1], label='Points')
    plt.scatter(points[pareto_points_idxs, 0], points[pareto_points_idxs, 1], color='red', label='Pareto points')
    plt.ylabel('log(L) of PSF')
    plt.xlabel('distance of minimal spanning tree')
    plt.legend()
    plt.show()
