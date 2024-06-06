import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.gaussian_predict import GaussianModelData, GaussianPredict


@pytest.mark.parametrize("di_gains", [True, False])
@pytest.mark.parametrize("order_approx", [0, 1])
def test_gaussian_predict(di_gains: bool, order_approx: int):
    gaussian_predict = GaussianPredict(order_approx=order_approx)
    row = 100
    chan = 4
    source = 1
    time = 15
    ant = 24
    lm = 1e-3 * jnp.ones((source, 2))
    n = jnp.sqrt(1. - jnp.sum(lm ** 2, axis=-1))
    lmn = jnp.concatenate([lm, n[:, None]], axis=-1)
    if di_gains:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (source, time, ant, chan, 2, 2)
    model_data = GaussianModelData(
        image=jnp.ones((source, chan, 2, 2), dtype=jnp.complex64),
        gains=jnp.ones(gain_shape,
                       dtype=jnp.complex64),
        lmn=lmn,
        ellipse_params=jnp.ones((source, 3))
    )
    visibility_coords = VisibilityCoords(
        uvw=jnp.ones((row, 3)),
        time_obs=jnp.ones((row,)),
        antenna_1=jnp.ones((row,), jnp.int64),
        antenna_2=jnp.ones((row,), jnp.int64),
        time_idx=jnp.ones((row,), jnp.int64)
    )
    freqs = jnp.ones((chan,))
    visibilities = gaussian_predict.predict(
        freqs=freqs,
        gaussian_model_data=model_data,
        visibility_coords=visibility_coords
    )
    print(order_approx, visibilities)
    assert np.all(np.isfinite(visibilities))
    assert np.shape(visibilities) == (row, chan, 2, 2)

    # Note: correctness is tested against wgridder


@pytest.mark.parametrize("di_gains", [True, False])
@pytest.mark.parametrize("order_approx", [0])
def test_ensure_gradients_work(di_gains: bool, order_approx: int):
    gaussian_predict = GaussianPredict(order_approx=order_approx)
    row = 100
    chan = 4
    source = 2
    time = 2
    ant = 3
    lm = 0. * jax.random.normal(jax.random.PRNGKey(42), (source, 2))
    n = jnp.sqrt(1. - jnp.sum(lm ** 2, axis=-1))
    lmn = jnp.concatenate([lm, n[:, None]], axis=-1)
    if di_gains:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (source, time, ant, chan, 2, 2)

    freqs = jnp.linspace(700e6, 2000e6, chan)

    antennas = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (ant, 3))
    antenna_1 = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, ant)
    antenna_2 = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, ant)

    uvw = antennas[antenna_2] - antennas[antenna_1]
    uvw = uvw.at[:, 2].mul(1e-3)

    times = jnp.linspace(0, 1, time)
    time_idx = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, time)
    time_obs = times[time_idx]

    image = jnp.zeros((source, chan, 2, 2), dtype=jnp.complex64)
    image = image.at[:, :, 0, 0].set(1.)
    image = image.at[:, :, 1, 1].set(1.)

    gains = jax.random.normal(jax.random.PRNGKey(42), gain_shape) + 1j * jax.random.normal(jax.random.PRNGKey(43),
                                                                                           gain_shape)
    # 50arcsec FWHM
    ellipse_params = jnp.zeros((source, 3))
    ellipse_params = ellipse_params.at[:, 0].set(50 / 3600. * np.pi / 180.)
    ellipse_params = ellipse_params.at[:, 1].set(50 / 3600. * np.pi / 180.)
    ellipse_params = ellipse_params.at[:, 2].set(0.)

    model_data = GaussianModelData(
        image=image,
        gains=gains,
        lmn=lmn,
        ellipse_params=ellipse_params
    )

    def func(freqs: jnp.ndarray, model_data: GaussianModelData, uvw: jax.Array):
        visibility_coords = VisibilityCoords(
            uvw=uvw,
            time_obs=time_obs,
            antenna_1=antenna_1,
            antenna_2=antenna_2,
            time_idx=time_idx
        )
        vis = gaussian_predict.predict(
            freqs=freqs,
            gaussian_model_data=model_data,
            visibility_coords=visibility_coords
        )

        return vis

    grad = jax.grad(lambda *args: jnp.sum(jnp.abs(func(*args)) ** 2), argnums=(0, 1, 2))(freqs, model_data, uvw)
    # print(func(freqs, model_data, uvw))
    # print(grad)
    (freq_grad, model_data_grad, uvw_grad) = grad
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

    gaussian_predict = GaussianPredict()
    row = 100
    chan = 4
    source = 1
    time = 15
    ant = 24
    lm = 1e-3 * jnp.ones((source, 2))
    n = jnp.sqrt(1. - jnp.sum(lm ** 2, axis=-1))
    lmn = jnp.concatenate([lm, n[:, None]], axis=-1)
    ellipse_params = jnp.ones((source, 3))

    image = jnp.ones((source, chan, 2, 2), dtype=jnp.float64)
    gains = jnp.ones((source, time, ant, chan, 2, 2), dtype=jnp.complex64)

    model_data = GaussianModelData(
        image=tree_device_put(image, NamedSharding(mesh, P(None, 'chan'))),
        gains=tree_device_put(gains, NamedSharding(mesh, P(None, None, None, 'chan'))),
        lmn=tree_device_put(lmn, NamedSharding(mesh, P())),
        ellipse_params=tree_device_put(ellipse_params, NamedSharding(mesh, P()))
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

    visibilities = gaussian_predict.predict(
        freqs=freqs,
        gaussian_model_data=model_data,
        visibility_coords=visibility_coords
    )
    assert np.all(np.isfinite(visibilities))
