import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.dft_predict import DFTPredict, DFTModelData


def test_dft_predict():
    dft_predict = DFTPredict()
    row = 100
    chan = 4
    source = 1
    time = 15
    ant = 24
    lm = 1e-3 * jnp.ones((source, 2))
    n = jnp.sqrt(1. - jnp.sum(lm ** 2, axis=-1))
    lmn = jnp.concatenate([lm, n[:, None]], axis=-1)
    model_data = DFTModelData(
        image=jnp.ones((source, chan, 2, 2), dtype=jnp.complex64),
        gains=jnp.ones((source, time, ant, chan, 2, 2), dtype=jnp.complex64),
        lmn=lmn
    )
    visibility_coords = VisibilityCoords(
        uvw=jnp.ones((row, 3)),
        time_obs=jnp.ones((row,)),
        antenna_1=jnp.ones((row,), jnp.int64),
        antenna_2=jnp.ones((row,), jnp.int64),
        time_idx=jnp.ones((row,), jnp.int64)
    )
    freqs = jnp.ones((chan,))
    visibilities = dft_predict.predict(
        freqs=freqs,
        dft_model_data=model_data,
        visibility_coords=visibility_coords
    )
    assert np.all(np.isfinite(visibilities))


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

    dft_predict = DFTPredict()
    row = 100
    chan = 4
    source = 1
    time = 15
    ant = 24
    lm = 1e-3 * jnp.ones((source, 2))
    n = jnp.sqrt(1. - jnp.sum(lm ** 2, axis=-1))
    lmn = jnp.concatenate([lm, n[:, None]], axis=-1)

    image = jnp.ones((source, chan, 2, 2), dtype=jnp.float64)
    gains = jnp.ones((source, time, ant, chan, 2, 2), dtype=jnp.complex64)

    model_data = DFTModelData(
        image=tree_device_put(image, NamedSharding(mesh, P(None, 'chan'))),
        gains=tree_device_put(gains, NamedSharding(mesh, P(None, None, None, 'chan'))),
        lmn=tree_device_put(lmn, NamedSharding(mesh, P()))
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

    visibilities = dft_predict.predict(
        freqs=freqs,
        dft_model_data=model_data,
        visibility_coords=visibility_coords
    )
    assert np.all(np.isfinite(visibilities))
