import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_cal.common.wgridder import dirty2vis, vis2dirty


def test_dirty2vis():
    uvw = jnp.ones((100, 3))
    freqs = jnp.ones((4,))
    dirty = jnp.ones((100, 100))
    wgt = jnp.ones((100, 4))
    pixsize_x = 0.1
    pixsize_y = 0.1
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
        pixsize_x=pixsize_x, pixsize_y=pixsize_y,
        center_x=center_x, center_y=center_y,
        epsilon=epsilon, do_wgridding=do_wgridding,
        flip_v=flip_v, divide_by_n=divide_by_n,
        sigma_min=sigma_min, sigma_max=sigma_max,
        nthreads=nthreads, verbosity=verbosity
    )
    assert np.all(np.isfinite(visibilities))

    # wgt works as expected
    visibilities2 = dirty2vis(
        uvw=uvw, freqs=freqs, dirty=dirty, wgt=2 * wgt,
        pixsize_x=pixsize_x, pixsize_y=pixsize_y,
        center_x=center_x, center_y=center_y,
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
                     pixsize_x=pixsize_x, pixsize_y=pixsize_y,
                     center_x=center_x, center_y=center_y,
                     epsilon=epsilon, do_wgridding=do_wgridding,
                     flip_v=flip_v, divide_by_n=divide_by_n,
                     sigma_min=sigma_min, sigma_max=sigma_max,
                     nthreads=nthreads, verbosity=verbosity
                 ))
    assert np.all(np.isfinite(fn(uvw)))


def test_vis2dirty():
    uvw = jnp.ones((100, 3))
    freqs = jnp.ones((4,))
    vis = jnp.ones((100, 4), dtype=jnp.complex64)
    npix_x = 100
    npix_y = 100
    pixsize_x = 0.1
    pixsize_y = 0.1
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
    dirty = vis2dirty(
        uvw=uvw, freqs=freqs, vis=vis, npix_x=npix_x, npix_y=npix_y,
        pixsize_x=pixsize_x, pixsize_y=pixsize_y, center_x=center_x, center_y=center_y,
        epsilon=epsilon, do_wgridding=do_wgridding, flip_v=flip_v, divide_by_n=divide_by_n,
        sigma_min=sigma_min, sigma_max=sigma_max, nthreads=nthreads, verbosity=verbosity
    )
    assert np.all(np.isfinite(dirty))

    # wgt works as expected
    dirty2 = vis2dirty(
        uvw=uvw, freqs=freqs, vis=vis, wgt=2 * vis, npix_x=npix_x, npix_y=npix_y,
        pixsize_x=pixsize_x, pixsize_y=pixsize_y, center_x=center_x, center_y=center_y,
        epsilon=epsilon, do_wgridding=do_wgridding, flip_v=flip_v, divide_by_n=divide_by_n,
        sigma_min=sigma_min, sigma_max=sigma_max, nthreads=nthreads, verbosity=verbosity
    )

    np.testing.assert_allclose(dirty2, dirty * 2)

    # JIT works
    fn = jax.jit(lambda uvw:
                 vis2dirty(
                     uvw=uvw, freqs=freqs, vis=vis, npix_x=npix_x, npix_y=npix_y,
                     pixsize_x=pixsize_x, pixsize_y=pixsize_y, center_x=center_x,
                     center_y=center_y, epsilon=epsilon, do_wgridding=do_wgridding,
                     flip_v=flip_v, divide_by_n=divide_by_n, sigma_min=sigma_min,
                     sigma_max=sigma_max, nthreads=nthreads, verbosity=verbosity))

    assert np.all(np.isfinite(fn(uvw)))
