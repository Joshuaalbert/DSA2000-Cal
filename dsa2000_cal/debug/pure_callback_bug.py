import jax
import jax.numpy as jnp
import numpy as np


def fn(x: jax.Array) -> jax.Array:
    # Define the expected shape & dtype of output.
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=np.shape(x),
        dtype=x.dtype
    )

    return jax.pure_callback(_host_fn, result_shape_dtype, x, vectorized=False)


def _host_fn(x: np.ndarray):
    print('x', type(x))
    assert isinstance(x, np.ndarray)
    return x


if __name__ == '__main__':
    jax.jit(fn)(jnp.asarray([1.]))
