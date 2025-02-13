import os

import jax
import jax.numpy as jnp

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=8"


def main(num_shards: int):
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec
    from jax.sharding import NamedSharding

    P = PartitionSpec

    if len(jax.devices()) < num_shards:
        raise ValueError(
            f"Number of devices {len(jax.devices())} is less than the number of shards {num_shards}"
        )

    devices = mesh_utils.create_device_mesh((num_shards,),
                                            devices=jax.devices()[:num_shards])
    mesh = Mesh(devices, axis_names=('i',))

    def tree_device_put(tree, sharding):
        return jax.tree.map(lambda x: jax.device_put(x, sharding), tree)

    @jax.jit
    def f(x):
        jax.debug.inspect_array_sharding(x, callback=print)
        y = jnp.sum(x, keepdims=True)
        jax.debug.inspect_array_sharding(y, callback=print)
        return y

    x = jnp.arange(num_shards)
    x = tree_device_put(x, NamedSharding(mesh, P('i')))
    print('x input sharding:')
    jax.debug.visualize_array_sharding(x)
    y = f(x)
    print('y output sharding:')
    jax.debug.visualize_array_sharding(y)
    assert y == num_shards * (num_shards - 1) / 2


if __name__ == '__main__':
    main(8)
