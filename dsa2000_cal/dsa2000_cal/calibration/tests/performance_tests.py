import itertools
import time

import jax
from jax import numpy as jnp

from dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardt
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.mixed_precision_utils import mp_policy


def test_multi_step_lm_performance():
    ant = 2048
    chan = 1
    source = 1

    gains = mp_policy.cast_to_gain(
        jnp.ones((source, ant, chan)) + 1j * 0.1 * jax.random.normal(jax.random.PRNGKey(1),
                                                                     (source, ant, chan))
    )
    antenna1, antenna2 = jnp.asarray(list(itertools.combinations_with_replacement(range(ant), 2)),
                                     dtype=mp_policy.index_dtype).T
    row = len(antenna1)

    vis_per_source = jnp.ones((source, row, chan), mp_policy.vis_dtype)

    def forward(gains, antenna1, antenna2, vis_per_source):
        g1 = gains[:, antenna1, :]  # (row, source, chan)
        g2 = gains[:, antenna2, :]
        vis = jnp.sum(g1 * vis_per_source * g2.conj(), axis=0)
        return vis

    data = forward(gains, antenna1, antenna2, vis_per_source) + 1e-3 * jax.random.normal(jax.random.PRNGKey(3),
                                                                                         (row, chan),
                                                                                         dtype=mp_policy.vis_dtype) + 1e-3 * 1j * jax.random.normal(
        jax.random.PRNGKey(4), (row, chan), dtype=mp_policy.vis_dtype)

    def run(antenna1, antenna2, vis_per_source, data):
        def residuals(gains):
            residuals = data - forward(gains, antenna1, antenna2, vis_per_source)
            return residuals

        x = jnp.ones((source, ant, chan), mp_policy.gain_dtype)

        lm = MultiStepLevenbergMarquardt(
            residual_fn=residuals,
            num_iterations=4,
            num_approx_steps=1,
            verbose=True
        )
        state = lm.create_initial_state(x)
        state, diagnostics = lm.solve(state)
        #
        # state = lm.update_initial_state(state)
        # state, diagnostics = lm.solve(state)
        return state, diagnostics

    t0 = time.time()
    run = jax.jit(run).lower(antenna1, antenna2, vis_per_source, data).compile()
    build_time = time.time() - t0
    print(f"Build time: {build_time}")
    t0 = time.time()
    block_until_ready(run(antenna1, antenna2, vis_per_source, data))
    run_time = time.time() - t0
    print(f"Run time: {run_time}")


def test_multi_step_lm_performance_defined_inside():
    ant = 2048
    chan = 1
    source = 1

    gains = mp_policy.cast_to_gain(
        jnp.ones((source, ant, chan)) + 1j * 0.1 * jax.random.normal(jax.random.PRNGKey(1),
                                                                     (source, ant, chan))
    )

    def forward(gains):
        antenna1, antenna2 = jnp.asarray(list(itertools.combinations_with_replacement(range(ant), 2)),
                                         dtype=mp_policy.index_dtype).T
        row = len(antenna1)

        vis_per_source = jnp.ones((source, row, chan), mp_policy.vis_dtype)

        g1 = gains[:, antenna1, :]  # (row, source, chan)
        g2 = gains[:, antenna2, :]
        vis = jnp.sum(g1 * vis_per_source * g2.conj(), axis=0)
        return vis

    data = forward(gains)
    noise = 1e-2 * mp_policy.cast_to_vis(
        jax.lax.complex(
            jax.random.normal(jax.random.PRNGKey(3), data.shape),
            jax.random.normal(jax.random.PRNGKey(4), data.shape)
        )
    )
    data += noise

    def run(data):
        def residuals(gains):
            residuals = data - forward(gains)
            return residuals

        x = jnp.ones((source, ant, chan), mp_policy.gain_dtype)

        lm = MultiStepLevenbergMarquardt(
            residual_fn=residuals,
            num_iterations=4,
            num_approx_steps=1,
            verbose=True
        )
        state = lm.create_initial_state(x)
        state, diagnostics = lm.solve(state)
        #
        # state = lm.update_initial_state(state)
        # state, diagnostics = lm.solve(state)
        return state, diagnostics

    t0 = time.time()
    run = jax.jit(run).lower(data).compile()
    build_time = time.time() - t0
    print(f"Build time: {build_time}")
    t0 = time.time()
    block_until_ready(run(data))
    run_time = time.time() - t0
    print(f"Run time: {run_time}")
