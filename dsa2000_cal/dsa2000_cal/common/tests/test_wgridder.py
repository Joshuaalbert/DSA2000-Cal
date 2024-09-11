import itertools

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.types import mp_policy
from dsa2000_cal.common.wgridder import vis_to_image, image_to_vis


@pytest.mark.parametrize("center_offset", [0.0, 0.1, 0.2])
def test_against_explicit(center_offset: float):
    np.random.seed(42)
    N = 512
    num_ants = 10
    num_freqs = 1

    pixsize = 0.5 * np.pi / 180 / 3600.  # 1 arcsec ~ 4 pixels / beam, so we'll avoid aliasing
    l0 = center_offset
    m0 = center_offset
    dl = pixsize
    dm = pixsize
    dirty = np.zeros((N, N), dtype=mp_policy.image_dtype)

    dirty[N // 2, N // 2] = 1.
    dirty[N // 3, N // 3] = 1.

    def pixel_to_lmn(xi, yi):
        l = l0 + (-N / 2 + xi) * dl
        m = m0 + (-N / 2 + yi) * dm
        n = np.sqrt(1. - l ** 2 - m ** 2)
        return np.asarray([l, m, n])

    x, y = np.where(dirty)
    lmn = [pixel_to_lmn(xi, yi) for xi, yi in zip(x, y)]
    pixel_fluxes = [dirty[xi, yi] for xi, yi in zip(x, y)]

    antenna_1, antenna_2 = np.asarray(list(itertools.combinations(range(num_ants), 2))).T
    antennas = 10e3 * np.random.normal(size=(num_ants, 3))
    antennas[:, 2] *= 0.001
    uvw = jnp.asarray(antennas[antenna_2] - antennas[antenna_1])

    freqs = jnp.linspace(700e6, 2000e6, num_freqs)

    vis = image_to_vis(
        uvw=uvw,
        freqs=freqs,
        dirty=jnp.asarray(dirty),
        pixsize_l=dl,
        pixsize_m=dm,
        center_l=l0,
        center_m=m0,
        epsilon=1e-6
    )

    vis_explicit = explicit_degridder(uvw, freqs, lmn, pixel_fluxes)

    num_test = 100
    # Now test the gridding
    dirty_rec = vis_to_image(
        uvw=uvw,
        freqs=freqs,
        vis=vis,
        npix_m=N,
        npix_l=N,
        pixsize_m=dm,
        pixsize_l=dl,
        center_m=m0,
        center_l=l0,
        scale_by_n=False,
        normalise=False
    )

    dirty_explicit = explicit_gridder(uvw, freqs, vis, N, N, dl, dm, l0, m0)

    np.testing.assert_allclose(dirty_rec, dirty_explicit, atol=5e-5)
    np.testing.assert_allclose(vis.real, vis_explicit.real, atol=1e-6)
    np.testing.assert_allclose(vis.imag, vis_explicit.imag, atol=1e-6)


@pytest.mark.parametrize("center_offset", [0.0, 0.1, 0.2])
def test_spectral_predict(center_offset: float):
    np.random.seed(42)
    N = 512
    num_ants = 10
    num_freqs = 2

    pixsize = 0.5 * np.pi / 180 / 3600.  # 1 arcsec ~ 4 pixels / beam, so we'll avoid aliasing
    l0 = m0 = jnp.repeat(center_offset, num_freqs)
    dl = dm = jnp.repeat(pixsize, num_freqs)
    dirty = np.zeros((num_freqs, N, N), dtype=mp_policy.image_dtype)

    dirty[:, N // 2, N // 2] = 1.
    dirty[:, N // 3, N // 3] = 1.

    antenna_1, antenna_2 = np.asarray(list(itertools.combinations(range(num_ants), 2))).T
    antennas = 10e3 * np.random.normal(size=(num_ants, 3))
    antennas[:, 2] *= 0.001
    uvw = jnp.asarray(antennas[antenna_2] - antennas[antenna_1])

    freqs = jnp.asarray([700e6, 700e6])

    vis = jax.vmap(
        lambda dirty, dl, dm, l0, m0, freqs:
        image_to_vis(
            uvw=uvw,
            freqs=freqs[None],
            dirty=dirty,
            pixsize_l=dl,
            pixsize_m=dm,
            center_l=l0,
            center_m=m0,
            epsilon=1e-6
        )
    )(dirty, dl, dm, l0, m0, freqs)
    assert np.shape(vis) == (num_freqs, len(uvw), 1)
    vis = vis[:, :, 0].T  # [num_rows, chan]
    assert np.all(vis[:, 0] == vis[:, 1])

    dirty_rec = vis_to_image(
        uvw=uvw,
        freqs=freqs,
        vis=vis,
        npix_m=N,
        npix_l=N,
        pixsize_m=dm,
        pixsize_l=dl,
        center_m=m0,
        center_l=l0,
        scale_by_n=False,
        normalise=True,
        spectral_cube=True
    )  # [N, N, chan]
    assert np.shape(dirty_rec) == (N, N, num_freqs)
    assert np.all(dirty_rec[:, :, 0] == dirty_rec[:, :, 1])


def explicit_degridder(uvw, freqs, lmn, pixel_fluxes):
    vis = np.zeros((len(uvw), len(freqs)), dtype=mp_policy.vis_dtype)
    c = 299792458.  # m/s

    for row, (u, v, w) in enumerate(uvw):
        for col, freq in enumerate(freqs):
            for flux, (l, m, n) in zip(pixel_fluxes, lmn):
                wavelength = c / freq
                phase = -2j * np.pi * (u * l + v * m + w * (n - 1)) / wavelength
                vis[row, col] += flux * np.exp(phase) / n
    return vis


def explicit_gridder(uvw, freqs, vis, num_l, num_m, dl, dm, center_l, center_m):
    c = 299792458.  # m/s
    lvec = (-0.5 * num_l + np.arange(num_l)) * dl + center_l
    mvec = (-0.5 * num_m + np.arange(num_m)) * dm + center_m
    L, M = np.meshgrid(lvec, mvec, indexing='ij')
    N = np.sqrt(1. - L ** 2 - M ** 2)
    dirty = np.zeros((num_l, num_m), mp_policy.image_dtype)
    for row, (u, v, w) in enumerate(uvw):
        for col, freq in enumerate(freqs):
            wavelength = c / freq
            phase = 2j * np.pi * (u * L + v * M + w * (N - 1)) / wavelength
            dirty += (vis[row, col] * np.exp(phase)).real
    return dirty


def test_normalisation():
    # Ensure that L-M axes of wgridder are correct, i.e. X=M, Y=-L
    np.random.seed(42)
    import pylab as plt
    # Validate the units of image
    # To do this we simulate an image with a single point source in the centre, and compute the visibilties from that.
    N = 512
    num_ants = 100
    num_freqs = 1

    pixsize = 0.5 * np.pi / 180 / 3600.  # 5 arcsec
    x0 = 0.
    y0 = 0.
    l0 = y0
    m0 = x0
    dl = pixsize
    dm = pixsize

    antenna_1, antenna_2 = np.asarray(list(itertools.combinations(range(num_ants), 2))).T

    num_rows = len(antenna_1)
    antennas = 10e3 * np.random.normal(size=(num_ants, 3))
    antennas[:, 2] *= 0.001
    uvw = jnp.asarray(antennas[antenna_2] - antennas[antenna_1])

    freqs = np.linspace(700e6, 2000e6, num_freqs)

    vis = jnp.ones(
        (num_rows, num_freqs),
        dtype=mp_policy.vis_dtype
    )

    dirty = vis_to_image(
        uvw=uvw,
        freqs=freqs,
        vis=vis,
        npix_m=N,
        npix_l=N,
        pixsize_m=dm,
        pixsize_l=dl,
        center_m=m0,
        center_l=l0,
        epsilon=1e-6,
        normalise=True
    )
    plt.imshow(dirty)
    plt.colorbar()
    plt.show()

    np.testing.assert_allclose(np.max(dirty), 1., atol=2e-2)
