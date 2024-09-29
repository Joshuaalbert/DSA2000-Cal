import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.mixed_precision_utils import complex_type, mp_policy
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.visibility_model.source_models.celestial.fits_source_model import FITSPredict, FITSModelData


def build_mock_point_model_data(full_stokes: bool, is_gains: bool, image_has_chan: bool, Nx: int, Ny: int, chan: int,
                                time: int, ant: int) -> FITSModelData:
    if full_stokes:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (time, ant, chan)
    if is_gains:
        gains = jnp.ones(gain_shape, dtype=complex_type)
    else:
        gains = None
    if image_has_chan:
        if full_stokes:
            image_shape = (chan, Nx, Ny, 2, 2)
        else:
            image_shape = (chan, Nx, Ny)
        l0 = jnp.zeros((chan,))
        m0 = jnp.zeros((chan,))
        dl = 0.01 * jnp.ones((chan,))
        dm = 0.01 * jnp.ones((chan,))
    else:
        if full_stokes:
            image_shape = (Nx, Ny, 2, 2)
        else:
            image_shape = (Nx, Ny)
        l0 = jnp.zeros(())
        m0 = jnp.zeros(())
        dl = 0.01 * jnp.ones(())
        dm = 0.01 * jnp.ones(())
    image = jnp.ones(image_shape, dtype=jnp.float32)
    freqs = 700e6 * jnp.ones((chan,))
    model_data = FITSModelData(
        image=mp_policy.cast_to_image(image),
        gains=mp_policy.cast_to_gain(gains),
        l0=mp_policy.cast_to_angle(l0), m0=mp_policy.cast_to_angle(m0),
        dl=mp_policy.cast_to_angle(dl), dm=mp_policy.cast_to_angle(dm),
        freqs=mp_policy.cast_to_freq(freqs)
    )
    # print(model_data)
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


@pytest.mark.parametrize("is_gains", [True, False])
@pytest.mark.parametrize("full_stokes", [True, False])
@pytest.mark.parametrize("image_has_chan", [True, False])
@pytest.mark.parametrize("chan", [1, 4])
def test_shapes_correctness(is_gains: bool, image_has_chan: bool, full_stokes: bool, chan: int):
    faint_predict = FITSPredict()
    Nx = 100
    Ny = 100
    time = 15
    ant = 24
    row = 1000
    model_data = build_mock_point_model_data(full_stokes, is_gains, image_has_chan, Nx, Ny, chan, time, ant)
    visibility_coords = build_mock_visibility_coord(row, ant, time)
    visibilities = faint_predict.predict(model_data=model_data, visibility_coords=visibility_coords)
    assert np.all(np.isfinite(visibilities))
    if full_stokes:
        assert np.shape(visibilities) == (row, chan, 2, 2)
    else:
        assert np.shape(visibilities) == (row, chan)

    # Note: correctness is based on wgridder correctness


@pytest.mark.parametrize("full_stokes", [True, False])
@pytest.mark.parametrize("image_has_chan", [True, False])
def test_grads_good(image_has_chan: bool, full_stokes: bool):
    faint_predict = FITSPredict()
    Nx = 100
    Ny = 100
    chan = 4
    time = 2
    ant = 24
    row = 1000
    _model_data = build_mock_point_model_data(full_stokes, is_gains=True, image_has_chan=image_has_chan, Nx=Nx, Ny=Ny,
                                              chan=chan, time=time, ant=ant)
    _visibility_coords = build_mock_visibility_coord(row, ant, time)

    def objective(gains):
        model_data = FITSModelData(
            image=_model_data.image,
            gains=mp_policy.cast_to_gain(gains),
            l0=_model_data.l0, m0=_model_data.m0,
            dl=_model_data.dl, dm=_model_data.dm,
            freqs=_model_data.freqs
        )

        vis = faint_predict.predict(model_data=model_data, visibility_coords=_visibility_coords)

        return jnp.sum(jnp.abs(vis) ** 2)

    gains_grad = jax.grad(objective, argnums=0)(_model_data.gains)
    assert np.all(np.isfinite(gains_grad))
    # print(func(freqs, model_data, uvw))
    # print(grad)
    # gain_shape = (time, ant, chan, 2, 2)
    for t in range(time):
        for a in range(ant):
            print(f"Time: {t}, Ant: {a}")
            if full_stokes:
                print("\tXX", gains_grad[t, a, ..., 0, 0])
                print("\tXY", gains_grad[t, a, ..., 0, 1])
                print("\tYX", gains_grad[t, a, ..., 1, 0])
                print("\tYY", gains_grad[t, a, ..., 1, 1])
            else:
                print("\tXX", gains_grad[t, a, ...])
            assert np.all(np.isfinite(gains_grad[t, a]))
