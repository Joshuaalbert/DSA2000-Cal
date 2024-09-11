from functools import partial

import jax
import jax.numpy as jnp


def add(x, y):
    print(x.shape, y.shape)
    return x + y

@jax.jit
@partial(jax.vmap, in_axes=(0, None))
def cb(x, y):
    return jax.pure_callback(add, jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype), x, y, vectorized=False)


if __name__ == '__main__':
    x = jnp.arange(4, dtype=jnp.float32)
    y = jnp.asarray(1.)
    print(cb(x, y))
