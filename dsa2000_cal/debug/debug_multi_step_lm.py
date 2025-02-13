import itertools
import os
import time
from typing import Tuple

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

from dsa2000_cal.calibration.solvers.multi_step_lm import MultiStepLevenbergMarquardt, MultiStepLevenbergMarquardtState, \
    MultiStepLevenbergMarquardtDiagnostic
from dsa2000_cal.common.jax_utils import create_mesh, tree_device_put, block_until_ready

import jax
import jax.numpy as jnp

from dsa2000_cal.common.mixed_precision_utils import mp_policy


def main():
    ant = 2048
    chan = 1
    num_time = 2
    source = os.cpu_count()

    mesh = create_mesh((source,), ('source',))

    gains = mp_policy.cast_to_gain(
        jnp.ones((source, ant, chan)) + 1j * 0.0 * jax.random.normal(jax.random.PRNGKey(1),
                                                                     (source, ant, chan))
    )

    antenna1, antenna2 = jnp.asarray(list(itertools.combinations_with_replacement(range(ant), 2)),
                                     dtype=mp_policy.index_dtype).T
    row = len(antenna1)

    vis_per_source = jnp.ones((source, row, chan), mp_policy.vis_dtype)
    vis_per_source_time = [vis_per_source for _ in range(num_time)]

    def forward(gains, antenna1, antenna2, vis_per_source_time):
        g1 = gains[:, antenna1, :]  # (row, source, chan)
        g2 = gains[:, antenna2, :]
        vis = [jnp.sum(g1 * vis_per_source * g2.conj(), axis=0) for vis_per_source in vis_per_source_time]
        return jnp.stack(vis, axis=0)

    data = mp_policy.cast_to_vis(
        forward(gains, antenna1, antenna2, vis_per_source_time) + 1e-1 * jax.lax.complex(
            jax.random.normal(jax.random.PRNGKey(3), (row, chan)),
            jax.random.normal(jax.random.PRNGKey(4), (row, chan))
        )
    )

    x = jnp.ones((source, ant, chan), mp_policy.gain_dtype)

    (antenna1, antenna2) = tree_device_put((antenna1, antenna2), mesh, ())
    vis_per_source_time = tree_device_put(vis_per_source_time, mesh, ('source',))
    data = tree_device_put(data, mesh, ())
    x = tree_device_put(x, mesh, ('source',))

    def run(x, antenna1, antenna2, vis_per_source_time, data) -> Tuple[
        MultiStepLevenbergMarquardtState, MultiStepLevenbergMarquardtDiagnostic]:
        def residuals(gains):
            residuals = data - forward(gains, antenna1, antenna2, vis_per_source_time)
            return residuals

        lm = MultiStepLevenbergMarquardt(
            residual_fn=residuals,
            num_iterations=2,
            num_approx_steps=3,
            verbose=True
        )
        state = lm.create_initial_state(x)
        state, diagnostics = lm.solve(state)
        return state, diagnostics

    t0 = time.time()
    run = jax.jit(run).lower(x, antenna1, antenna2, vis_per_source_time, data).compile()
    build_time = time.time() - t0
    print(f"Build time: {build_time}")
    t0 = time.time()
    state, diagnostics = block_until_ready(run(x, antenna1, antenna2, vis_per_source_time, data))
    run_time = time.time() - t0
    print(f"Run time: {run_time}")
    # print(state)
    print(diagnostics)

    # With Sharding over source
    # Build time: 4.152379035949707
    # Run time: 7.416905164718628

    # With no sharding
    # Build time: 4.468706846237183
    # Run time: 9.48699402809143

    x_error = state.x - gains  # (source, ant, chan)
    x_abs_error = jnp.abs(state.x) - jnp.abs(gains)
    x_angle_error = jnp.angle(state.x) - jnp.angle(gains)

    import pylab as plt
    plt.hist(x_abs_error.reshape(-1), bins='auto')
    plt.show()

    plt.hist(x_angle_error.reshape(-1), bins='auto')
    plt.show()

    plt.hist(jnp.real(x_error).reshape(-1), bins='auto')
    plt.show()

    plt.hist(jnp.imag(x_error).reshape(-1), bins='auto')
    plt.show()


if __name__ == '__main__':
    main()
