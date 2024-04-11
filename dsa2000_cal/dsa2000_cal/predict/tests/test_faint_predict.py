import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.faint_predict import FaintPredict, FaintModelData


@pytest.mark.parametrize("gain_has_chan", [True, False])
@pytest.mark.parametrize("image_has_chan", [True, False])
def test_gaint_predict(gain_has_chan: bool, image_has_chan: bool):
    faint_predict = FaintPredict()
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
    model_data = FaintModelData(
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

    faint_predict = FaintPredict()
    Nx = 100
    Ny = 100
    chan = 4
    time = 15
    ant = 24
    row = 1000
    image = jnp.ones((chan, Nx, Ny), dtype=jnp.float32)
    gains = jnp.ones((time, ant, chan, 2, 2), dtype=jnp.complex64)
    l0 = jnp.ones((chan,))
    m0 = jnp.ones((chan,))
    dl = jnp.ones((chan,))
    dm = jnp.ones((chan,))
    model_data = FaintModelData(
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
