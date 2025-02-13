import time

import jax
import jax.numpy as jnp
import numpy as np


def dynamic_close(x):
    result_dtype = jax.tree.map(lambda _x: jax.ShapeDtypeStruct(_x.shape, _x.dtype), x)
    return jax.pure_callback(lambda _: x, result_dtype, None)


if __name__ == '__main__':
    for n in [10000]:
        big_shape = (n, n)  # Example large shape


        def f(x):
            @jax.remat
            def compute_y():
                return jnp.asarray(np.ones(big_shape))

            y = compute_y()
            # y = dynamic_close(jnp.asarray(np.ones(big_shape)))
            return x + y


        x = jnp.ones(())
        print(jax.make_jaxpr(f)(x))

        t0 = time.time()
        f_jit = jax.jit(f).lower(x).compile()
        print(n, "JIT compile time:", time.time() - t0)
        t0 = time.time()
        jax.block_until_ready(f(x))
        print(n, "Eager time:", time.time() - t0)
