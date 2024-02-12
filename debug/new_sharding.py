import jax
from jax._src.sharding_impls import PositionalSharding
from jax import numpy as jnp

import os

# Force 2 jax  hosts
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


if __name__ == '__main__':
    x = jnp.ones((18, 18))
    sharding1 = PositionalSharding(jax.devices()[:6])
    sharding2 = PositionalSharding(jax.devices()[2:])

    y = jax.device_put(x, sharding1.reshape(3, 2))
    # jax.debug.visualize_array_sharding(y)
    z = jax.device_put(x, sharding2.reshape(3, 2))
    print(y+z)