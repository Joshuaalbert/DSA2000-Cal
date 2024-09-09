import itertools
import time

import jax
from jax import numpy as jnp

from dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardt
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.types import mp_policy


def test_multi_step_lm():
    ant = 2048
    chan = 1
    source = 3

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
        def residuals(params):
            gains_real, gains_imag = params
            gains = mp_policy.cast_to_gain(gains_real + 1j * gains_imag)
            residuals = data - forward(gains, antenna1, antenna2, vis_per_source)
            return residuals

        x = jnp.ones((source, ant, chan), mp_policy.gain_dtype)
        x = (x.real, x.imag)

        lm = MultiStepLevenbergMarquardt(residual_fn=residuals,
                                         num_approx_steps=1,
                                         verbose=True)
        state = lm.create_initial_state(x)
        state, diagnostics = lm.solve(state)

        state = lm.update_initial_state(state)
        state, diagnostics = lm.solve(state)
        return state, diagnostics

    t0 = time.time()
    run = jax.jit(run).lower(antenna1, antenna2, vis_per_source, data).compile()
    build_time = time.time() - t0
    print(f"Build time: {build_time}")
    t0 = time.time()
    block_until_ready(run(antenna1, antenna2, vis_per_source, data))
    run_time = time.time() - t0
    print(f"Run time: {run_time}")
