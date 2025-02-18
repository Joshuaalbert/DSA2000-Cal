import time

import jax
import jax.numpy as jnp


def main():
    # Set up array dimensions
    M = 1024  # Size of the first FFT dimension
    N = 1024  # Size of the second FFT dimension
    extra_dim = 1024  # Size of the extra dimension

    key = jax.random.PRNGKey(0)

    # Scenario 1: FFT over the last two dimensions
    # Array shape: (extra_dim, M, N)
    array1 = jax.random.normal(key, (extra_dim, M, N))

    @jax.jit
    def fft_scenario1(x):
        return jnp.fft.fftn(x, axes=(-2, -1))

    # Scenario 2: FFT over the first two dimensions
    # Array shape: (M, N, extra_dim)
    array2 = jax.random.normal(key, (M, N, extra_dim))

    @jax.jit
    def fft_scenario2(x):
        return jnp.fft.fftn(x, axes=(0, 1))

    # Warm-up runs (to exclude JIT compilation time)
    fft_scenario1(array1).block_until_ready()
    fft_scenario2(array2).block_until_ready()

    # Measure execution time for Scenario 1
    start_time1 = time.time()
    result1 = fft_scenario1(array1)
    result1.block_until_ready()  # Ensure computation is complete
    end_time1 = time.time()
    time1 = end_time1 - start_time1

    # Measure execution time for Scenario 2
    start_time2 = time.time()
    result2 = fft_scenario2(array2)
    result2.block_until_ready()  # Ensure computation is complete
    end_time2 = time.time()
    time2 = end_time2 - start_time2

    # Print the execution times
    print(f"Execution time for FFT over last dimensions (Scenario 1): {time1:.6f} seconds")
    print(f"Execution time for FFT over first dimensions (Scenario 2): {time2:.6f} seconds")

    # Compare the times
    if time1 < time2:
        print("FFT over the last dimensions is faster.")
    else:
        print("FFT over the first dimensions is faster.")

    # Execution time for FFT over last dimensions (Scenario 1): 4.060475 seconds
    # Execution time for FFT over first dimensions (Scenario 2): 13.220937 seconds
    # FFT over the last dimensions is faster.


if __name__ == '__main__':
    main()
