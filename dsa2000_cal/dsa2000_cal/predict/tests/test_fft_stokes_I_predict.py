import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.fft_stokes_I_predict import FFTStokesIPredict, FFTStokesIModelData


@pytest.mark.parametrize("gain_has_chan", [True, False])
@pytest.mark.parametrize("image_has_chan", [True, False])
def test_faint_predict(gain_has_chan: bool, image_has_chan: bool):
    faint_predict = FFTStokesIPredict()
    Nx = 100
    Ny = 100
    chan = 4
    time = 15
    ant = 24
    row = 1000
    if gain_has_chan:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (time, ant, 2, 2)
    if image_has_chan:
        image_shape = (chan, Nx, Ny)
        l0 = jnp.zeros((chan,))
        m0 = jnp.zeros((chan,))
        dl = -0.01 * jnp.ones((chan,))
        dm = 0.01 * jnp.ones((chan,))
    else:
        image_shape = (Nx, Ny)
        l0 = jnp.zeros(())
        m0 = jnp.zeros(())
        dl = -0.01 * jnp.ones(())
        dm = 0.01 * jnp.ones(())
    image = jnp.ones(image_shape, dtype=jnp.float32)
    model_data = FFTStokesIModelData(
        image=image,
        gains=jnp.ones(gain_shape,
                       dtype=jnp.complex64),
        l0=l0, m0=m0, dl=dl, dm=dm
    )
    visibility_coords = VisibilityCoords(
        uvw=jnp.ones((row, 3)),
        time_obs=jnp.ones((row,)),
        antenna_1=jnp.ones((row,), jnp.int64),
        antenna_2=jnp.ones((row,), jnp.int64),
        time_idx=jnp.ones((row,), jnp.int64)
    )
    freqs = jnp.ones((chan,))
    visibilities = faint_predict.predict(
        freqs=freqs,
        faint_model_data=model_data,
        visibility_coords=visibility_coords
    )
    assert np.all(np.isfinite(visibilities))
    assert np.shape(visibilities) == (row, chan, 2, 2)


def test_with_sharding():
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec
    from jax.sharding import NamedSharding

    P = PartitionSpec

    devices = mesh_utils.create_device_mesh((len(jax.devices()),))
    mesh = Mesh(devices, axis_names=('chan',))

    def tree_device_put(tree, sharding):
        return jax.tree_map(lambda x: jax.device_put(x, sharding), tree)

    faint_predict = FFTStokesIPredict()
    Nx = 100
    Ny = 100
    chan = 4
    time = 15
    ant = 24
    row = 1000
    image = jnp.ones((chan, Nx, Ny), dtype=jnp.float32)
    gains = jnp.ones((time, ant, chan, 2, 2), dtype=jnp.complex64)
    l0 = jnp.zeros((chan,))
    m0 = jnp.zeros((chan,))
    dl = -0.1 * jnp.ones((chan,))
    dm = 0.1 * jnp.ones((chan,))
    model_data = FFTStokesIModelData(
        image=tree_device_put(image, NamedSharding(mesh, P('chan'))),
        gains=tree_device_put(gains, NamedSharding(mesh, P(None, None, 'chan'))),
        l0=tree_device_put(l0, NamedSharding(mesh, P('chan'))),
        m0=tree_device_put(m0, NamedSharding(mesh, P('chan'))),
        dl=tree_device_put(dl, NamedSharding(mesh, P('chan'))),
        dm=tree_device_put(dm, NamedSharding(mesh, P('chan')))
    )

    uvw = jnp.ones((row, 3))
    time = jnp.ones((row,))
    antenna_1 = jnp.ones((row,), jnp.int64)
    antenna_2 = jnp.ones((row,), jnp.int64)
    time_idx = jnp.ones((row,), jnp.int64)

    visibility_coords = VisibilityCoords(
        uvw=tree_device_put(uvw, NamedSharding(mesh, P())),
        time_obs=tree_device_put(time, NamedSharding(mesh, P())),
        antenna_1=tree_device_put(antenna_1, NamedSharding(mesh, P())),
        antenna_2=tree_device_put(antenna_2, NamedSharding(mesh, P())),
        time_idx=tree_device_put(time_idx, NamedSharding(mesh, P()))
    )
    freqs = jnp.ones((chan,))
    freqs = tree_device_put(freqs, NamedSharding(mesh, P('chan')))

    visibilities = faint_predict.predict(
        freqs=freqs,
        faint_model_data=model_data,
        visibility_coords=visibility_coords
    )
    assert np.all(np.isfinite(visibilities))


@pytest.mark.parametrize("gain_has_chan", [True, False])
@pytest.mark.parametrize("image_has_chan", [True, False])
def test_grads_work(gain_has_chan: bool, image_has_chan: bool):
    predict = FFTStokesIPredict()
    row = 100
    chan = 4
    Nx = Ny = 256
    time = 2
    ant = 3
    if gain_has_chan:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (time, ant, 2, 2)

    freqs = jnp.linspace(700e6, 2000e6, chan)

    antennas = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (ant, 3))
    antenna_1 = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, ant)
    antenna_2 = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, ant)

    uvw = antennas[antenna_2] - antennas[antenna_1]
    uvw = uvw.at[:, 2].mul(1e-3)

    times = jnp.linspace(0, 1, time)
    time_idx = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, time)
    time_obs = times[time_idx]

    if image_has_chan:
        image = jax.random.normal(jax.random.PRNGKey(0), (chan, Nx, Ny), dtype=jnp.float32)
        l0 = jnp.zeros((chan,))
        m0 = jnp.zeros((chan,))
        dl = -0.01 / Nx * jnp.ones((chan,))
        dm = 0.01 / Nx * jnp.ones((chan,))
    else:
        image = jax.random.normal(jax.random.PRNGKey(0), (Nx, Ny), dtype=jnp.float32)
        l0 = jnp.zeros(())
        m0 = jnp.zeros(())
        dl = -0.01 / Nx * jnp.ones(())
        dm = 0.01 / Nx * jnp.ones(())

    gains = jax.random.normal(jax.random.PRNGKey(42), gain_shape) + 1j * jax.random.normal(jax.random.PRNGKey(43),
                                                                                           gain_shape)

    visibility_coords = VisibilityCoords(
        uvw=uvw,
        time_obs=time_obs,
        antenna_1=antenna_1,
        antenna_2=antenna_2,
        time_idx=time_idx
    )

    def func(gains):
        model_data = FFTStokesIModelData(
            image=image,
            gains=gains,
            l0=l0, m0=m0, dl=dl, dm=dm
        )

        vis = predict.predict(
            freqs=freqs,
            faint_model_data=model_data,
            visibility_coords=visibility_coords
        )

        return vis

    gains_grad = jax.grad(lambda gains: jnp.sum(jnp.abs(func(gains)) ** 2), argnums=0)(gains)
    # print(func(freqs, model_data, uvw))
    # print(grad)
    # gain_shape = (time, ant, chan, 2, 2)
    for t in range(time):
        for a in range(ant):
            print(f"Time: {t}, Ant: {a}")

            print("\tXX", gains_grad[t, a, ..., 0, 0])
            print("\tXY", gains_grad[t, a, ..., 0, 1])
            print("\tYX", gains_grad[t, a, ..., 1, 0])
            print("\tYY", gains_grad[t, a, ..., 1, 1])
            # Ensure gradient is not zero
            assert np.all(np.abs(gains_grad[t, a]) > 1e-10)
