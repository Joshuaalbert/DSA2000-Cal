import itertools

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.wgridder import dirty2vis, vis2dirty
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.point_predict import PointPredict, PointModelData
from dsa2000_cal.source_models.corr_translation import linear_to_stokes


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


@pytest.mark.parametrize("num_ants", [100, 200])
@pytest.mark.parametrize("num_freqs", [2, 4, 5])
def test_gh53(num_ants: int, num_freqs: int):
    import pylab as plt
    # Validate the units of image
    # To do this we simulate an image with a single point source in the centre, and compute the visibilties from that.
    n = 1024

    pixsize = 5 * np.pi / 180 / 3600.
    dirty = np.zeros((n, n))
    # [0, 1, 3]
    dirty[n // 2, n // 2] = 1.
    dirty[n // 3, n // 3] = 1.
    antenna_1, antenna_2 = np.asarray(list(itertools.combinations(range(num_ants), 2))).T

    num_rows = len(antenna_1)
    antennas = 20e3 * np.random.normal(size=(num_ants, 3))
    antennas[:, 2] *= 0.001
    uvw = antennas[antenna_2] - antennas[antenna_1]

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
        pixsize_x=pixsize,
        pixsize_y=pixsize,
        center_x=0.,
        center_y=0.,
        epsilon=1e-4,
        do_wgridding=True,
        flip_v=False,
        divide_by_n=True,
        nthreads=1,
        verbosity=0
    )
    print(vis)

    dirty_rec = vis2dirty(
        uvw=uvw,
        freqs=freqs,
        vis=vis,
        npix_x=n,
        npix_y=n,
        pixsize_x=pixsize,
        pixsize_y=pixsize,
        wgt=wgt,
        center_x=0.,
        center_y=0.,
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

    predict = PointPredict()
    image = np.zeros((1, num_freqs, 2, 2))  # [source, chan, 2, 2]
    image[:, :, 0, 0] = 0.5
    image[:, :, 1, 1] = 0.5
    gains = np.zeros((1, num_ants, num_freqs, 2, 2))  # [[source,] time, ant, chan, 2, 2]
    gains[..., 0, 0] = 1.
    gains[..., 1, 1] = 1.
    lmn = np.zeros((1, 3))  # [source, 3]
    lmn[0, 2] = 1.

    vis_point_predict_linear = predict.predict(
        freqs=freqs,
        dft_model_data=PointModelData(
            image=image,
            gains=gains,
            lmn=lmn
        ),
        visibility_coords=VisibilityCoords(
            uvw=uvw,
            time_obs=np.zeros(num_rows),
            antenna_1=antenna_1,
            antenna_2=antenna_2,
            time_idx=np.zeros(num_rows, jnp.int64)
        )
    )  # [row, chan, 2, 2]
    vis_point_predict_stokes = jax.vmap(jax.vmap(linear_to_stokes))(vis_point_predict_linear)[:, :, 0, 0]
    # print(vis_point_predict_stokes)
    # np.testing.assert_allclose(vis_point_predict_stokes, vis, atol=1e-3)
    np.testing.assert_allclose(dirty_rec, dirty, atol=0.1)
