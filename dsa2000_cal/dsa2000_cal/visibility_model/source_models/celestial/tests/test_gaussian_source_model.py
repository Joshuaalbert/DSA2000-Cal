import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.types import mp_policy
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.visibility_model.source_models.celestial.gaussian_source_model import \
    GaussianPredict, GaussianModelData


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
