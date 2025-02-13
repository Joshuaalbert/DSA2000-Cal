import time

import jax
import numpy as np
import jax.numpy as jnp
from jax import lax
from jax import random, jit


def main(batch_size1, batch_size2, length):
    # Define a simple function f
    def f(x):
        return jnp.sin(x ** 2 + 1)

    # Initialize the random number generator
    key = random.PRNGKey(0)

    # Define batch dimensions

    # Generate random input data
    xs = random.normal(key, (batch_size1, batch_size2, length))

    # Define the functions
    def scan_vmap_f(xs):
        # xs: [batch_size1, batch_size2]
        def body_fn(carry, x):
            return (), jax.vmap(f)(x)  # [batch_size2]

        _, result = lax.scan(body_fn, (), xs)  # [batch_size1, batch_size2]
        return result

    def vmap_scan_f(xs):
        # xs: [batch_size1, batch_size2]

        def scan(x):
            # x: [batch_size2]
            def body_fn(carry, x):
                return (), f(x)

            _, result = lax.scan(body_fn, (), x)  # [batch_size2]
            return result

        return jax.vmap(scan)(xs)  # [batch_size1, batch_size2]

    # JIT compile the functions
    scan_vmap_f_jit = jit(scan_vmap_f).lower(xs).compile()
    vmap_scan_f_jit = jit(vmap_scan_f).lower(xs).compile()

    # Measure execution time for scan(vmap(f))
    start_time = time.time()
    ys_scan_vmap = scan_vmap_f_jit(xs)
    ys_scan_vmap.block_until_ready()
    end_time = time.time()
    dt_scan_vmap = end_time - start_time

    # Measure execution time for vmap(scan(f))
    start_time = time.time()
    ys_vmap_scan = vmap_scan_f_jit(xs)
    ys_vmap_scan.block_until_ready()
    end_time = time.time()
    dt_vmap_scan = end_time - start_time

    # Check if the results are the same
    np.testing.assert_allclose(ys_scan_vmap, ys_vmap_scan)

    print(f"batch_size1: {batch_size1}, batch_size2: {batch_size2}, length: {length}\n"
          f"\t> scan(vmap(f)): {dt_scan_vmap:.2e} s\n"
          f"\t> vmap(scan(f)): {dt_vmap_scan:.2e} s\n"
          f"\t\t> Winner: {'scan(vmap(f))' if dt_scan_vmap < dt_vmap_scan else 'vmap(scan(f))'}\n"
          f"\t\t> Speedup: {dt_vmap_scan / dt_scan_vmap if dt_scan_vmap < dt_vmap_scan else dt_scan_vmap / dt_vmap_scan:.2f}x")


if __name__ == '__main__':
    for batch_size1 in [100, 1000, 5000]:
        for batch_size2 in [100, 1000, 5000]:
            for length in [1, 10, 100]:
                main(batch_size1=batch_size1, batch_size2=batch_size2, length=length)
