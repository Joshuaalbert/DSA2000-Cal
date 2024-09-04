import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.visibility_model.source_models.celestial.fits_source_model import FITSPredict, FITSModelData


@pytest.mark.parametrize("is_gains", [True, False])
@pytest.mark.parametrize("full_stokes", [True, False])
@pytest.mark.parametrize("image_has_chan", [True, False])
def test_shapes_correctness(is_gains: bool, image_has_chan: bool, full_stokes: bool):
    faint_predict = FITSPredict()
    Nx = 100
    Ny = 100
    chan = 4
    time = 15
    ant = 24
    row = 1000
    if full_stokes:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (time, ant, chan)
    if is_gains:
        gains = jnp.ones(gain_shape, dtype=jnp.complex64)
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
    freqs = jnp.ones((chan,))
    model_data = FITSModelData(
        image=image,
        gains=gains,
        l0=l0, m0=m0,
        dl=dl, dm=dm,
        freqs=freqs
    )
    visibility_coords = VisibilityCoords(
        uvw=jnp.ones((row, 3)),
        time_obs=jnp.ones((row,)),
        antenna_1=jnp.ones((row,), jnp.int64),
        antenna_2=jnp.ones((row,), jnp.int64),
        time_idx=jnp.ones((row,), jnp.int64)
    )
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
    if full_stokes:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (time, ant, chan)

    gains = jax.random.normal(jax.random.PRNGKey(0), gain_shape, dtype=jnp.complex64)

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
    freqs = jnp.ones((chan,))
    antennas = 10e3 * jax.random.normal(jax.random.PRNGKey(0), (ant, 3))
    antennas = antennas.at[:, 2].set(antennas[:, 2] * 0.001)

    antenna_1 = jax.random.randint(jax.random.PRNGKey(0), (row,), 0, ant)
    antenna_2 = jax.random.randint(jax.random.PRNGKey(0), (row,), 0, ant)
    uvw = antennas[antenna_1] - antennas[antenna_2]
    visibility_coords = VisibilityCoords(
        uvw=uvw,
        time_obs=jnp.zeros((row,)),
        antenna_1=antenna_1,
        antenna_2=antenna_2,
        time_idx=jnp.zeros((row,), jnp.int64)
    )

    def objective(gains):
        model_data = FITSModelData(
            image=image,
            gains=gains,
            l0=l0, m0=m0,
            dl=dl, dm=dm,
            freqs=freqs
        )

        vis = faint_predict.predict(model_data=model_data, visibility_coords=visibility_coords)

        return jnp.sum(jnp.abs(vis) ** 2)

    gains_grad = jax.grad(objective, argnums=0)(gains)
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
            if t == 0:
                # Ensure gradient is not zero
                assert np.all(np.abs(gains_grad[t, a]) > 1e-10)
            else:
                # Ensure gradient is zero
                assert np.all(np.abs(gains_grad[t, a]) < 1e-10)
