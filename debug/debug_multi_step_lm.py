import os
import time

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

from dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardt
from dsa2000_cal.common.jax_utils import create_mesh, tree_device_put

import jax
import jax.numpy as jnp

from dsa2000_cal.common.types import mp_policy


def main():
    ant = 2048
    row = int(ant * (ant - 1) / 2)
    chan = 1
    source = 3

    mesh = create_mesh((source,), ('source',))

    gains = mp_policy.cast_to_gain(
        jnp.ones((source, ant, chan)) + 1j * 0.1 * jax.random.normal(jax.random.PRNGKey(1),
                                                                     (source, ant, chan))
    )
    vis_per_source = jnp.ones((source, row, chan), mp_policy.vis_dtype)
    antenna1 = jax.random.randint(jax.random.PRNGKey(0), (row,), 0, ant, dtype=mp_policy.index_dtype)
    antenna2 = jax.random.randint(jax.random.PRNGKey(1), (row,), 0, ant, dtype=mp_policy.index_dtype)

    def forward(gains, antenna1, antenna2, vis_per_source):
        g1 = gains[:, antenna1, :]  # (row, source, chan)
        g2 = gains[:, antenna2, :]
        vis = jnp.sum(g1 * vis_per_source * g2.conj(), axis=0)
        return vis

    data = forward(gains, antenna1, antenna2, vis_per_source) + 1e-3 * jax.random.normal(jax.random.PRNGKey(3),
                                                                                         (row, chan),
                                                                                         dtype=mp_policy.vis_dtype) + 1e-3 * 1j * jax.random.normal(
        jax.random.PRNGKey(4), (row, chan), dtype=mp_policy.vis_dtype)

    (antenna1, antenna2) = tree_device_put((antenna1, antenna2), mesh, ())
    vis_per_source = tree_device_put(vis_per_source, mesh, ('source',))
    data = tree_device_put(data, mesh, ())

    def run(antenna1, antenna2, vis_per_source, data):
        def residuals(gains):
            residuals = data - forward(gains, antenna1, antenna2, vis_per_source)
            return residuals

        x = jnp.ones((source, ant, chan), mp_policy.gain_dtype)

        lm = MultiStepLevenbergMarquardt(
            residual_fn=residuals,
            num_approx_steps=5,
            verbose=True
        )
        state = lm.create_initial_state(x)
        return lm.solve(state)

    t0 = time.time()
    run = jax.jit(run).lower(antenna1, antenna2, vis_per_source, data).compile()
    build_time = time.time() - t0
    print(f"Build time: {build_time}")
    t0 = time.time()
    state = jax.block_until_ready(run(antenna1, antenna2, vis_per_source, data))
    run_time = time.time() - t0
    print(f"Run time: {run_time}")
    # print(state)


if __name__ == '__main__':
    main()