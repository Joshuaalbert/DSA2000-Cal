import itertools

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.corr_translation import linear_to_stokes
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.wgridder import image_to_vis, vis_to_image
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.visibility_model.source_models.celestial.point_source_model import PointPredict, PointModelData


def build_mock_point_model_data(di_gains: bool, chan: int, source: int, time: int, ant: int) -> PointModelData:
    lm = 1e-3 * jax.random.normal(jax.random.PRNGKey(42), (source, 2))
    n = jnp.sqrt(1. - jnp.sum(lm ** 2, axis=-1))
    lmn = jnp.concatenate([lm, n[:, None]], axis=-1)
    if di_gains:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (source, time, ant, chan, 2, 2)
    freqs = jnp.ones((chan,))
    model_data = PointModelData(
        freqs=mp_policy.cast_to_freq(freqs),
        image=mp_policy.cast_to_image(jnp.ones((source, chan, 2, 2))),
        gains=mp_policy.cast_to_gain(jnp.ones(gain_shape, jnp.complex64)),
        lmn=mp_policy.cast_to_angle(lmn)
    )
    print(model_data)
    return model_data


def build_mock_visibility_coord(rows: int, ant: int, time: int) -> VisibilityCoords:
    uvw = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (rows, 3))
    uvw = uvw.at[:, 2].mul(1e-3)
    time_obs = jnp.zeros((rows,))
    antenna_1 = jax.random.randint(jax.random.PRNGKey(42), (rows,), 0, ant)
    antenna_2 = jax.random.randint(jax.random.PRNGKey(43), (rows,), 0, ant)
    time_idx = jax.random.randint(jax.random.PRNGKey(44), (rows,), 0, time)

    visibility_coords = VisibilityCoords(
        uvw=mp_policy.cast_to_length(uvw),
        time_obs=mp_policy.cast_to_time(time_obs),
        antenna_1=mp_policy.cast_to_index(antenna_1),
        antenna_2=mp_policy.cast_to_index(antenna_2),
        time_idx=mp_policy.cast_to_index(time_idx)
    )
    return visibility_coords


@pytest.mark.parametrize("di_gains", [True, False])
def test_dft_predict(di_gains: bool):
    dft_predict = PointPredict()
    row = 100
    chan = 4
    source = 1
    time = 15
    ant = 24
    model_data = build_mock_point_model_data(di_gains, chan, source, time, ant)
    visibility_coords = build_mock_visibility_coord(row, ant, time)

    visibilities = dft_predict.predict(model_data=model_data, visibility_coords=visibility_coords)
    assert np.all(np.isfinite(visibilities))
    assert np.shape(visibilities) == (row, chan, 2, 2)

    # Note: correctness is tested against wgridder


@pytest.mark.parametrize("di_gains", [True, False])
def test_ensure_gradients_work(di_gains: bool):
    point_predict = PointPredict()
    row = 100
    chan = 4
    source = 2
    time = 2
    ant = 3

    model_data = build_mock_point_model_data(di_gains, chan, source, time, ant)
    _visibility_coords = build_mock_visibility_coord(row, ant, time)

    def func(model_data: PointModelData, uvw: jax.Array):
        visibility_coords = VisibilityCoords(
            uvw=uvw,
            time_obs=_visibility_coords.time_obs,
            antenna_1=_visibility_coords.antenna_1,
            antenna_2=_visibility_coords.antenna_2,
            time_idx=_visibility_coords.time_idx
        )
        vis = point_predict.predict(model_data=model_data, visibility_coords=visibility_coords)

        return vis

    grad = jax.grad(lambda *args: jnp.sum(jnp.abs(func(*args)) ** 2), argnums=(0, 1))(model_data,
                                                                                      _visibility_coords.uvw)
    # print(func(freqs, model_data, uvw))
    # print(grad)
    (model_data_grad, uvw_grad) = grad
    if di_gains:
        # gain_shape = (time, ant, chan, 2, 2)
        for t in range(time):
            for a in range(ant):
                print(f"Time: {t}, Ant: {a}")
                print("\tXX", model_data_grad.gains[t, a, :, 0, 0])
                print("\tXY", model_data_grad.gains[t, a, :, 0, 1])
                print("\tYX", model_data_grad.gains[t, a, :, 1, 0])
                print("\tYY", model_data_grad.gains[t, a, :, 1, 1])
                # Ensure gradient is not zero
                assert np.all(np.abs(model_data_grad.gains[t, a, :, :, :]) > 1e-10)

    else:
        # gain_shape = (source, time, ant, chan, 2, 2)
        for s in range(source):
            for t in range(time):
                for a in range(ant):
                    print(f"Source: {s}, Time: {t}, Ant: {a}")
                    print("\tXX", model_data_grad.gains[s, t, a, :, 0, 0])
                    print("\tXY", model_data_grad.gains[s, t, a, :, 0, 1])
                    print("\tYX", model_data_grad.gains[s, t, a, :, 1, 0])
                    print("\tYY", model_data_grad.gains[s, t, a, :, 1, 1])
                    # Ensure gradient is not zero
                    assert np.all(np.abs(model_data_grad.gains[s, t, a, :, :, :]) > 1e-10)


def test_gh55_point():
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
    l0 = x0
    m0 = y0
    dl = pixsize
    dm = pixsize
    dirty = np.zeros((N, N))  # [Nl, Nm]
    # place a central pixel: 0, 1, 2, 3 ==> 4/2 - 0.5 = 1.5 == l0
    # ==> l(n) = l0 + (n - (N - 1)/2) * dl
    # l(0) = l0 - (N-1)/2 * dl
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

    num_rows = len(antenna_1)
    antennas = 10e3 * np.random.normal(size=(num_ants, 3))
    antennas[:, 2] *= 0.001
    uvw = jnp.asarray(antennas[antenna_2] - antennas[antenna_1])

    plt.scatter(uvw[:, 0], uvw[:, 1])
    plt.show()
    freqs = np.linspace(700e6, 2000e6, num_freqs)

    wgt = None  # np.ones((num_rows, num_freqs))
    # wgt = np.random.uniform(size=(num_rows, num_freqs))

    vis = image_to_vis(
        uvw=uvw,
        freqs=freqs,
        dirty=dirty,
        pixsize_m=dm,
        pixsize_l=dl,
        center_m=m0,
        center_l=l0,
        epsilon=1e-4
    )
    print(vis)

    dirty_rec = vis_to_image(
        uvw=uvw,
        freqs=freqs,
        vis=vis,
        npix_m=N,
        npix_l=N,
        pixsize_m=dm,
        pixsize_l=dl,
        wgt=wgt,
        center_m=m0,
        center_l=l0,
        epsilon=1e-4
    )

    plt.imshow(dirty_rec, origin='lower',
               interpolation='nearest', cmap='inferno')
    plt.colorbar()
    plt.show()
    plt.imshow(dirty, origin='lower',
               interpolation='nearest', cmap='inferno')
    plt.colorbar()
    plt.show()

    diff_dirty = dirty_rec - dirty
    plt.imshow(diff_dirty, origin='lower',
               interpolation='nearest', cmap='inferno')
    plt.colorbar()
    plt.show()

    predict = PointPredict(convention='physical')
    image = np.zeros((2, num_freqs, 2, 2))  # [source, chan, 2, 2]
    image[:, :, 0, 0] = 0.5
    image[:, :, 1, 1] = 0.5
    gains = np.zeros((2, num_ants, num_freqs, 2, 2), jnp.complex64)  # [[source,] time, ant, chan, 2, 2]
    gains[..., 0, 0] = 1.
    gains[..., 1, 1] = 1.
    lmn = np.stack([lmn1, lmn2], axis=0)  # [source, 3]

    vis_point_predict_linear = predict.predict(
        model_data=PointModelData(
            freqs=mp_policy.cast_to_freq(freqs),
            image=mp_policy.cast_to_image(image),
            gains=mp_policy.cast_to_gain(gains),
            lmn=mp_policy.cast_to_angle(lmn)
        ), visibility_coords=VisibilityCoords(
            uvw=mp_policy.cast_to_length(uvw),
            time_obs=mp_policy.cast_to_time(np.zeros(num_rows)),
            antenna_1=mp_policy.cast_to_index(antenna_1),
            antenna_2=mp_policy.cast_to_index(antenna_2),
            time_idx=mp_policy.cast_to_index(np.zeros(num_rows), quiet=True)
        ))  # [row, chan, 2, 2]
    vis_point_predict_stokes = jax.vmap(jax.vmap(linear_to_stokes))(vis_point_predict_linear)[:, :, 0, 0]
    print(vis_point_predict_stokes)
    np.testing.assert_allclose(vis_point_predict_stokes.real, vis.real, atol=1e-3)
    np.testing.assert_allclose(vis_point_predict_stokes.imag, vis.imag, atol=1e-3)
    np.testing.assert_allclose(dirty_rec.max(), dirty.max(), atol=1e-2)
