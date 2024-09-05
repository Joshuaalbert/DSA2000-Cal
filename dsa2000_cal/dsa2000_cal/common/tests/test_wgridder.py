import itertools

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.types import complex_type
from dsa2000_cal.common.wgridder import dirty2vis, vis2dirty


def test_dirty2vis():
    uvw = 20e3 * jax.random.normal(jax.random.PRNGKey(0), (100, 3))
    uvw = uvw.at[:, 2].set(0.001 * uvw[:, 2])

    freqs = 1400e6 * jnp.ones((4,))
    dirty = jnp.ones((100, 100))
    wgt = jnp.ones((100, 4))
    pixsize_x = 0.001
    pixsize_y = 0.001
    center_x = 0.0
    center_y = 0.0
    epsilon = 1e-4
    do_wgridding = True
    flip_v = False
    divide_by_n = True
    sigma_min = 1.1
    sigma_max = 2.6
    nthreads = 1
    verbosity = 0
    visibilities = dirty2vis(
        uvw=uvw, freqs=freqs, dirty=dirty, wgt=wgt,
        pixsize_m=pixsize_x, pixsize_l=pixsize_y,
        center_m=center_x, center_l=center_y,
        epsilon=epsilon, do_wgridding=do_wgridding,
        flip_v=flip_v, divide_by_n=divide_by_n,
        sigma_min=sigma_min, sigma_max=sigma_max,
        nthreads=nthreads, verbosity=verbosity
    )
    assert np.all(np.isfinite(visibilities))

    # wgt works as expected
    visibilities2 = dirty2vis(
        uvw=uvw, freqs=freqs, dirty=dirty, wgt=2 * wgt,
        pixsize_m=pixsize_x, pixsize_l=pixsize_y,
        center_m=center_x, center_l=center_y,
        epsilon=epsilon, do_wgridding=do_wgridding,
        flip_v=flip_v, divide_by_n=divide_by_n,
        sigma_min=sigma_min, sigma_max=sigma_max,
        nthreads=nthreads, verbosity=verbosity
    )

    np.testing.assert_allclose(visibilities2, visibilities * 2)

    # JIT works
    fn = jax.jit(lambda uvw:
                 dirty2vis(
                     uvw=uvw, freqs=freqs, dirty=dirty, wgt=wgt,
                     pixsize_m=pixsize_x, pixsize_l=pixsize_y,
                     center_m=center_x, center_l=center_y,
                     epsilon=epsilon, do_wgridding=do_wgridding,
                     flip_v=flip_v, divide_by_n=divide_by_n,
                     sigma_min=sigma_min, sigma_max=sigma_max,
                     nthreads=nthreads, verbosity=verbosity
                 ))
    assert np.all(np.isfinite(fn(uvw)))


def test_vis2dirty():
    uvw = 20e3 * jax.random.normal(jax.random.PRNGKey(0), (100, 3))
    uvw = uvw.at[:, 2].set(0.001 * uvw[:, 2])

    freqs = 1400e6 * jnp.ones((4,))
    dirty = jnp.ones((100, 100))
    wgt = jnp.ones((100, 4))
    pixsize_x = 0.001
    pixsize_y = 0.001
    center_x = 0.0
    center_y = 0.0
    npix_x = 100
    npix_y = 100

    vis = jnp.ones((100, 4), dtype=complex_type)
    epsilon = 1e-4
    do_wgridding = True
    flip_v = False
    divide_by_n = True
    sigma_min = 1.1
    sigma_max = 2.6
    nthreads = 1
    verbosity = 0
    dirty = vis2dirty(
        uvw=uvw, freqs=freqs, vis=vis, npix_m=npix_x, npix_l=npix_y,
        pixsize_m=pixsize_x, pixsize_l=pixsize_y, center_m=center_x, center_l=center_y,
        epsilon=epsilon, do_wgridding=do_wgridding, flip_v=flip_v, divide_by_n=divide_by_n,
        sigma_min=sigma_min, sigma_max=sigma_max, nthreads=nthreads, verbosity=verbosity
    )
    assert np.all(np.isfinite(dirty))

    # wgt works as expected
    dirty2 = vis2dirty(
        uvw=uvw, freqs=freqs, vis=vis, wgt=2 * vis, npix_m=npix_x, npix_l=npix_y,
        pixsize_m=pixsize_x, pixsize_l=pixsize_y, center_m=center_x, center_l=center_y,
        epsilon=epsilon, do_wgridding=do_wgridding, flip_v=flip_v, divide_by_n=divide_by_n,
        sigma_min=sigma_min, sigma_max=sigma_max, nthreads=nthreads, verbosity=verbosity
    )

    np.testing.assert_allclose(dirty2, dirty)

    # JIT works
    fn = jax.jit(lambda uvw:
                 vis2dirty(
                     uvw=uvw, freqs=freqs, vis=vis, npix_m=npix_x, npix_l=npix_y,
                     pixsize_m=pixsize_x, pixsize_l=pixsize_y, center_m=center_x,
                     center_l=center_y, epsilon=epsilon, do_wgridding=do_wgridding,
                     flip_v=flip_v, divide_by_n=divide_by_n, sigma_min=sigma_min,
                     sigma_max=sigma_max, nthreads=nthreads, verbosity=verbosity))

    assert np.all(np.isfinite(fn(uvw)))


@pytest.mark.parametrize("num_ants", [100, 200])
@pytest.mark.parametrize("num_freqs", [2, 4, 5])
def test_gh53(num_ants: int, num_freqs: int):
    import pylab as plt
    # Validate the units of image
    # To do this we simulate an image with a single point source in the centre, and compute the visibilties from that.
    n = 1024

    pixsize = 0.5 * np.pi / 180 / 3600.
    dirty = np.zeros((n, n))
    # [0, 1, 3]
    dirty[n // 2, n // 2] = 1.
    dirty[n // 3, n // 3] = 1.
    antenna_1, antenna_2 = np.asarray(list(itertools.combinations(range(num_ants), 2))).T

    num_rows = len(antenna_1)
    antennas = 10e3 * np.random.normal(size=(num_ants, 3))
    antennas[:, 2] *= 0.001
    uvw = jnp.asarray(antennas[antenna_2] - antennas[antenna_1])

    plt.scatter(uvw[:, 0], uvw[:, 1])
    plt.show()
    freqs = np.linspace(700e6, 2000e6, num_freqs)

    # wgt = np.ones((num_rows, num_freqs))
    wgt = np.random.uniform(size=(num_rows, num_freqs))

    vis = dirty2vis(
        uvw=uvw,
        freqs=freqs,
        dirty=dirty,
        wgt=None,
        pixsize_m=pixsize,
        pixsize_l=pixsize,
        center_m=0.,
        center_l=0.,
        epsilon=1e-4,
        do_wgridding=True,
        flip_v=False,
        divide_by_n=True,
        nthreads=1,
        verbosity=0
    )

    dirty_rec = vis2dirty(
        uvw=uvw,
        freqs=freqs,
        vis=vis,
        npix_m=n,
        npix_l=n,
        pixsize_m=pixsize,
        pixsize_l=pixsize,
        wgt=wgt,
        center_m=0.,
        center_l=0.,
        epsilon=1e-4,
        do_wgridding=True,
        flip_v=False,
        divide_by_n=True,
        nthreads=1,
        verbosity=0
    )

    plt.imshow(dirty_rec, origin='lower',
               interpolation='nearest', cmap='inferno')
    plt.colorbar()
    plt.show()
    plt.imshow(dirty, origin='lower',
               interpolation='nearest', cmap='inferno')
    plt.colorbar()
    plt.show()

    np.testing.assert_allclose(dirty_rec.max(), dirty.max(), atol=2e-1)


@pytest.mark.parametrize("center_offset", [0.0, 0.1, 0.2])
@pytest.mark.parametrize("negate_w", [False, True])
def test_wrong_w(center_offset: float, negate_w: bool):
    np.random.seed(42)
    N = 512
    num_ants = 100
    num_freqs = 1

    pixsize = 0.5 * np.pi / 180 / 3600.  # 1 arcsec ~ 4 pixels / beam, so we'll avoid aliasing
    l0 = center_offset
    m0 = center_offset
    dl = pixsize
    dm = pixsize
    dirty = np.zeros((N, N))

    dirty[N // 2, N // 2] = 1.
    dirty[N // 3, N // 3] = 1.

    def pixel_to_lmn(xi, yi):
        l = l0 + (-N / 2 + xi) * dl
        m = m0 + (-N / 2 + yi) * dm
        n = np.sqrt(1. - l ** 2 - m ** 2)
        return np.asarray([l, m, n])

    lmn1 = pixel_to_lmn(N // 2, N // 2)
    lmn2 = pixel_to_lmn(N // 3, N // 3)

    antenna_1, antenna_2 = np.asarray(list(itertools.combinations(range(num_ants), 2))).T
    antennas = 10e3 * np.random.normal(size=(num_ants, 3))
    antennas[:, 2] *= 0.001
    uvw = jnp.asarray(antennas[antenna_2] - antennas[antenna_1])

    freqs = jnp.linspace(700e6, 2000e6, num_freqs)

    vis = dirty2vis(
        uvw=uvw,
        freqs=freqs,
        dirty=jnp.asarray(dirty),
        wgt=None,
        pixsize_l=dl,
        pixsize_m=dm,
        center_l=l0,
        center_m=m0,
        epsilon=1e-4,
        do_wgridding=True,
        flip_v=False,
        divide_by_n=True,
        nthreads=1,
        verbosity=0,
    )

    lmn = [lmn1, lmn2]
    pixel_fluxes = [1., 1.]
    vis_explicit = explicit_degridder(uvw, freqs, lmn, pixel_fluxes, negate_w)

    if negate_w:
        try:
            np.testing.assert_allclose(vis.real, vis_explicit.real, atol=1e-4)
            np.testing.assert_allclose(vis.imag, vis_explicit.imag, atol=1e-4)
        except AssertionError as e:
            print(f"Error for center_offset={center_offset}")
            print(str(e))
    else:
        # tolerances need 64bit to be correct
        np.testing.assert_allclose(vis.real, vis_explicit.real, atol=1e-4)
        np.testing.assert_allclose(vis.imag, vis_explicit.imag, atol=1e-4)


@pytest.fixture(scope='function')
def mock_vis():
    np.random.seed(42)
    N = 512
    num_ants = 100

    antenna_1, antenna_2 = np.asarray(list(itertools.combinations(range(num_ants), 2))).T
    antennas = 10e3 * np.random.normal(size=(num_ants, 3))
    antennas[:, 2] *= 0.001
    uvw = jnp.asarray(antennas[antenna_2] - antennas[antenna_1])
    num_freqs = 1
    freqs = jnp.linspace(700e6, 2000e6, num_freqs)

    c = 299792458.  # m/s
    wavelengths = c / freqs

    max_baseline = jnp.max(jnp.linalg.norm(uvw / jnp.min(wavelengths), axis=-1))

    center_offset = 0.2

    # pixsize = 0.5 * np.pi / 180 / 3600.  # 0.5 arcsec ~ 4 pixels / beam, so we'll avoid aliasing
    pixsize = (1. / max_baseline) / 4.

    l0 = center_offset
    m0 = center_offset
    dl = pixsize
    dm = pixsize
    dirty = np.zeros((N, N))

    dirty[N // 2, N // 2] = 1.
    dirty[N // 3, N // 3] = 1.

    def pixel_to_lmn(xi, yi):
        l = l0 + (-N / 2 + xi) * dl
        m = m0 + (-N / 2 + yi) * dm
        n = np.sqrt(1. - l ** 2 - m ** 2)
        return np.asarray([l, m, n])

    lmn1 = pixel_to_lmn(N // 2, N // 2)
    lmn2 = pixel_to_lmn(N // 3, N // 3)

    lmn = [lmn1, lmn2]
    pixel_fluxes = [1., 1.]
    vis_explicit = explicit_degridder(uvw, freqs, lmn, pixel_fluxes, negate_w=False)
    return vis_explicit, dirty, uvw, freqs, l0, m0, dl, dm


def explicit_degridder(uvw, freqs, lmn, pixel_fluxes, negate_w):
    vis = np.zeros((len(uvw), len(freqs)), dtype=np.complex128)
    c = 299792458.  # m/s

    for row, (u, v, w) in enumerate(uvw):
        if negate_w:
            w = -w
        for col, freq in enumerate(freqs):
            for flux, (l, m, n) in zip(pixel_fluxes, lmn):
                wavelength = c / freq
                phase = -2j * np.pi * (u * l + v * m + w * (n - 1)) / wavelength
                vis[row, col] += flux * np.exp(phase) / n
    return vis



def test_adjoint_factor():
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
        dtype=complex_type
    )

    dirty = vis2dirty(
        uvw=uvw,
        freqs=freqs,
        vis=vis,
        npix_m=N,
        npix_l=N,
        pixsize_m=dm,
        pixsize_l=dl,
        wgt=None,
        center_m=m0,
        center_l=l0,
        epsilon=1e-6,
        do_wgridding=True,
        flip_v=False,
        divide_by_n=False,
        nthreads=1,
        verbosity=0
    )
    plt.imshow(dirty)
    plt.colorbar()
    plt.show()

    np.testing.assert_allclose(np.max(dirty), 1., atol=2e-2)
