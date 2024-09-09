import itertools

import jax
import numpy as np
import pytest
from astropy import units as au
from jax import numpy as jnp

from dsa2000_cal.common.corr_translation import linear_to_stokes
from dsa2000_cal.common.types import mp_policy
from dsa2000_cal.common.wgridder import dirty2vis, vis2dirty, image_to_vis, vis_to_image
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.visibility_model.source_models.celestial.gaussian_source_model import \
    GaussianPredict, GaussianModelData, GaussianSourceModel


def build_mock_point_model_data(di_gains: bool, chan: int, source: int, time: int, ant: int) -> GaussianModelData:
    lm = 1e-3 * jax.random.normal(jax.random.PRNGKey(42), (source, 2))
    n = jnp.sqrt(1. - jnp.sum(lm ** 2, axis=-1))
    lmn = jnp.concatenate([lm, n[:, None]], axis=-1)
    if di_gains:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (source, time, ant, chan, 2, 2)
    freqs = jnp.ones((chan,))
    model_data = GaussianModelData(
        freqs=mp_policy.cast_to_freq(freqs),
        image=mp_policy.cast_to_image(jnp.ones((source, chan, 2, 2))),
        gains=mp_policy.cast_to_gain(jnp.ones(gain_shape)),
        lmn=mp_policy.cast_to_angle(lmn),
        ellipse_params=mp_policy.cast_to_angle(jnp.ones((source, 3)))
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
@pytest.mark.parametrize("order_approx", [0, 1])
def test_gaussian_predict(di_gains: bool, order_approx: int):
    gaussian_predict = GaussianPredict(
        order_approx=order_approx
    )
    row = 100
    chan = 4
    source = 1
    time = 15
    ant = 24
    model_data = build_mock_point_model_data(di_gains, chan, source, time, ant)
    visibility_coords = build_mock_visibility_coord(row, ant, time)
    visibilities = gaussian_predict.predict(model_data=model_data, visibility_coords=visibility_coords)
    print(order_approx, visibilities)
    assert np.all(np.isfinite(visibilities))
    assert np.shape(visibilities) == (row, chan, 2, 2)

    # Note: correctness is tested against wgridder


@pytest.mark.parametrize("di_gains", [True, False])
@pytest.mark.parametrize("order_approx", [0, 1])
def test_ensure_gradients_work(di_gains: bool, order_approx: int):
    gaussian_predict = GaussianPredict(order_approx=order_approx)
    row = 100
    chan = 4
    source = 2
    time = 2
    ant = 3
    model_data = build_mock_point_model_data(di_gains, chan, source, time, ant)
    _visibility_coords = build_mock_visibility_coord(row, ant, time)

    def objective(model_data: GaussianModelData, uvw: jax.Array):
        visibility_coords = VisibilityCoords(
            uvw=uvw,
            time_obs=_visibility_coords.time_obs,
            antenna_1=_visibility_coords.antenna_1,
            antenna_2=_visibility_coords.antenna_2,
            time_idx=_visibility_coords.time_idx
        )
        vis = gaussian_predict.predict(model_data=model_data, visibility_coords=visibility_coords)

        return jnp.sum(jnp.abs(vis) ** 2)

    grad = jax.grad(objective, argnums=(0, 1))(model_data, _visibility_coords.uvw)
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


def test_gh55_gaussian():
    # Ensure that L-M axes of wgridder are correct, i.e. X=M, Y=-L
    np.random.seed(42)
    import pylab as plt
    # Validate the units of image
    # To do this we simulate an image with a single point source in the centre, and compute the visibilties from that.
    N = 1024
    num_ants = 200
    num_freqs = 1
    freqs = np.linspace(700e6, 2000e6, num_freqs)

    major_pix = 20
    minor_pix = 10

    pixsize = 0.5 * np.pi / 180 / 3600.
    x0 = 0.
    y0 = 0.
    l0 = y0
    m0 = x0
    dl = pixsize
    dm = pixsize
    L, M = np.meshgrid(-N / 2 + np.arange(N), -N / 2 + np.arange(N), indexing='ij')
    L *= pixsize
    M *= pixsize

    g = GaussianSourceModel(
        l0=l0 * au.dimensionless_unscaled,
        m0=m0 * au.dimensionless_unscaled,
        A=1. * au.Jy,
        major=pixsize * major_pix * au.dimensionless_unscaled,
        minor=pixsize * minor_pix * au.dimensionless_unscaled,
        theta=0. * au.rad,
        freqs=freqs * au.Hz
    )
    dirty = g.get_flux_model(lvec=(-N / 2 + np.arange(N)) * pixsize,
                             mvec=(-N / 2 + np.arange(N)) * pixsize)[2].T.value  # [Nl, Nm]
    plt.imshow(dirty.T, origin='lower')
    plt.colorbar()
    plt.show()
    np.testing.assert_allclose(np.sum(dirty), 1.)

    antenna_1, antenna_2 = np.asarray(list(itertools.combinations(range(num_ants), 2))).T

    num_rows = len(antenna_1)
    antennas = 10e3 * np.random.normal(size=(num_ants, 3))
    antennas[:, 2] *= 0.001
    uvw = jnp.asarray(antennas[antenna_2] - antennas[antenna_1])

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

    sc = plt.scatter(uvw[:, 0], uvw[:, 1], c=np.abs(vis)[:, 0], s=1, alpha=0.5)
    plt.colorbar(sc)
    plt.show()
    predict = GaussianPredict(convention='physical')
    image = np.zeros((1, num_freqs, 2, 2))  # [source, chan, 2, 2]
    image[:, :, 0, 0] = 0.5
    image[:, :, 1, 1] = 0.5
    gains = np.zeros((1, num_ants, num_freqs, 2, 2))  # [[source,] time, ant, chan, 2, 2]
    gains[..., 0, 0] = 1.
    gains[..., 1, 1] = 1.
    lmn = np.asarray([[0., 0., 1.]])  # [source, 3]

    vis_predict = predict.predict(
        model_data=GaussianModelData(
            freqs=mp_policy.cast_to_freq(freqs),
            image=mp_policy.cast_to_image(image),
            gains=mp_policy.cast_to_gain(gains),
            lmn=mp_policy.cast_to_angle(lmn),
            ellipse_params=mp_policy.cast_to_angle(np.asarray([[major_pix * pixsize, minor_pix * pixsize, 0.]]))
        ), visibility_coords=VisibilityCoords(
            uvw=mp_policy.cast_to_length(uvw),
            time_obs=mp_policy.cast_to_time(np.zeros(num_rows)),
            antenna_1=mp_policy.cast_to_index(antenna_1),
            antenna_2=mp_policy.cast_to_index(antenna_2),
            time_idx=mp_policy.cast_to_index(np.zeros(num_rows))
        )
    )  # [row, chan, 2, 2]
    vis_predict_stokes = jax.vmap(jax.vmap(linear_to_stokes))(vis_predict)[:, :, 0, 0]
    print(vis_predict_stokes)
    sc = plt.scatter(uvw[:, 0], uvw[:, 1], c=np.abs(vis_predict_stokes)[:, 0], s=1, alpha=0.5)
    plt.colorbar(sc)
    plt.show()

    sc = plt.scatter(uvw[:, 0], uvw[:, 1], c=np.abs(vis_predict_stokes - vis)[:, 0], s=1, alpha=0.5)
    plt.colorbar(sc)
    plt.show()

    dirty_rec = vis_to_image(
        uvw=uvw,
        freqs=freqs,
        vis=vis_predict_stokes,
        npix_m=N,
        npix_l=N,
        pixsize_m=dm,
        pixsize_l=dl,
        wgt=wgt,
        center_m=m0,
        center_l=l0,
        epsilon=1e-4
    )

    plt.imshow(dirty_rec.T, origin='lower',
               interpolation='nearest', cmap='inferno')
    plt.colorbar()
    plt.show()
    plt.imshow(dirty.T, origin='lower',
               interpolation='nearest', cmap='inferno')
    plt.colorbar()
    plt.show()

    diff_dirty = dirty_rec - dirty
    plt.imshow(diff_dirty.T, origin='lower',
               interpolation='nearest', cmap='inferno')
    plt.colorbar()
    plt.show()

    np.testing.assert_allclose(dirty_rec, dirty, atol=0.11)

    np.testing.assert_allclose(vis_predict_stokes.real, vis.real, atol=1e-3)
    np.testing.assert_allclose(vis_predict_stokes.imag, vis.imag, atol=1e-3)
