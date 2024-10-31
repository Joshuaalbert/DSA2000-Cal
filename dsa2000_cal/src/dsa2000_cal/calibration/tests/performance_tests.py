import itertools
import time

import jax
import numpy as np
from jax import numpy as jnp

from src.dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardt
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.mixed_precision_utils import mp_policy


def test_multi_step_lm_performance():
    ant = 2048
    chan = 1
    source = 7

    gains = mp_policy.cast_to_gain(
        jnp.ones((source, ant, chan)) + 1j * 0.2 * jax.random.normal(jax.random.PRNGKey(1),
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

    data = forward(gains, antenna1, antenna2, vis_per_source) + 0.89 * mp_policy.cast_to_vis(jax.lax.complex(
        jax.random.normal(jax.random.PRNGKey(3), (row, chan)),
        jax.random.normal(jax.random.PRNGKey(4), (row, chan))))

    def run(antenna1, antenna2, vis_per_source, data):
        def residuals(gains):
            residuals = data - forward(gains, antenna1, antenna2, vis_per_source)
            return residuals

        x = jnp.ones((source, ant, chan), mp_policy.gain_dtype)

        lm = MultiStepLevenbergMarquardt(
            residual_fn=residuals,
            num_iterations=10,
            num_approx_steps=0,
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


def test_multi_step_lm_performance_better_memory_contiguous():
    ant = 2048
    chan = 1
    source = 7

    gains_true = mp_policy.cast_to_gain(
        jax.lax.complex(
            jnp.ones((ant, chan, source)),
            0.2 * jax.random.normal(jax.random.PRNGKey(1), (ant, chan, source))
        )
    )
    antenna1, antenna2 = jnp.asarray(list(itertools.combinations_with_replacement(range(ant), 2)),
                                     dtype=mp_policy.index_dtype).T
    row = len(antenna1)

    vis_per_source = mp_policy.cast_to_vis(
        jax.lax.complex(
            jnp.ones((row, chan, source)),
            0.2 * jax.random.normal(jax.random.PRNGKey(1), (row, chan, source))
        )
    )

    def forward(gains, antenna1, antenna2, vis_per_source):
        g1 = gains[antenna1, :, :]  # (row, chan, source)
        g2 = gains[antenna2, :, :]  # (row, chan, source)
        vis = jnp.sum(g1 * vis_per_source * g2.conj(), axis=-1)  # (row, chan)
        return vis

    data = forward(gains_true, antenna1, antenna2, vis_per_source) + 0.89 * mp_policy.cast_to_vis(jax.lax.complex(
        jax.random.normal(jax.random.PRNGKey(3), (row, chan)),
        jax.random.normal(jax.random.PRNGKey(4), (row, chan))))

    def run(antenna1, antenna2, vis_per_source, data):
        def residuals(gains):
            residuals = data - forward(gains, antenna1, antenna2, vis_per_source)
            return residuals / 0.89

        x = jnp.ones((ant, chan, source), mp_policy.gain_dtype)

        lm = MultiStepLevenbergMarquardt(
            residual_fn=residuals,
            num_iterations=100,
            num_approx_steps=0,
            verbose=True,
            gtol=1e-6
        )
        # gtol = 1e-8, Iter: 47
        # Run time: 116.39145541191101
        # Mean absolute error: 0.0848182961344719
        # Root mean square error: 0.09568079560995102
        # Mean error: (2.3129035980673507e-05+0.0001123386318795383j)
        # gtol = 1e-6, Iter: 16
        # Run time: 32.003414154052734
        # Mean absolute error: 0.08480879664421082
        # Root mean square error: 0.09566979110240936
        # Mean error: (2.3161173885455355e-05+0.0001139110536314547j)

        state = lm.create_initial_state(x)
        state, diagnostics = lm.solve(state)
        #
        # state = lm.update_initial_state(state)
        # state, diagnostics = lm.solve(state)
        mean_error = jnp.mean(state.x - gains_true)
        ma_error = jnp.mean(jnp.abs(state.x - gains_true))
        rms_error = jnp.sqrt(jnp.mean(jnp.abs(state.x - gains_true) ** 2))
        return state, diagnostics, ma_error, rms_error, mean_error

    t0 = time.time()
    run = jax.jit(run).lower(antenna1, antenna2, vis_per_source, data).compile()
    build_time = time.time() - t0
    print(f"Build time: {build_time}")
    t0 = time.time()
    state, diagnostics, ma_error, rms_error, mean_error = block_until_ready(
        run(antenna1, antenna2, vis_per_source, data))
    run_time = time.time() - t0
    print(f"Run time: {run_time}")
    print(f"Mean absolute error: {ma_error}")
    print(f"Root mean square error: {rms_error}")
    print(f"Mean error: {mean_error}")
    error = np.abs(state.x - gains_true)
    import pylab as plt
    plt.hist(error.flatten(), bins='auto')
    plt.show()
    phase_error = np.angle(state.x) - np.angle(gains_true)
    plt.hist(phase_error.flatten(), bins='auto')
    plt.show()
    print(diagnostics)
    print(state)
