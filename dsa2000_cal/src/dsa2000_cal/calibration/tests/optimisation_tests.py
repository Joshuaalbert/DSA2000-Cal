import dataclasses
import itertools
import time
from functools import partial
from typing import NamedTuple

import jax
from jax import numpy as jnp

from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.array_types import FloatArray

jax.config.update("jax_explain_cache_misses", True)

import tensorflow_probability.substrates.jax as tfp
from jaxns import Prior, Model

from dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardt

tfpd = tfp.distributions


class CalibrationParams(NamedTuple):
    # Improvement threshold
    p_any_improvement: FloatArray = 0.1  # p0 > 0
    p_less_newton: FloatArray = 0.25  # p2 -- less than sufficient improvement
    p_sufficient_improvement: FloatArray = 0.5  # p1 > p0
    p_more_newton: FloatArray = 0.75  # p3 -- more than sufficient improvement

    # Damping alteration factors 0 < c_more_newton < 1 < c_less_newton
    c_more_newton: FloatArray = 0.1
    c_less_newton: FloatArray = 2.
    # Damping factor = mu1 * ||F(x)||^delta, 1 <= delta <= 2
    delta: FloatArray = 2
    # mu1 > mu_min > 0
    mu1: FloatArray = 1.
    mu_min: FloatArray = 1e-3


@dataclasses.dataclass(eq=False)
class ForwardModel:
    ant: int
    source: int
    chan: int

    @property
    def gain_shape(self):
        return (self.ant, self.chan, self.source)

    def create_forward_data(self):
        antenna1, antenna2 = jnp.asarray(list(itertools.combinations_with_replacement(range(self.ant), 2)),
                                         dtype=jnp.int32).T
        row = len(antenna1)
        vis_per_source = jnp.ones((row, self.chan, self.source), jnp.complex64)
        return antenna1, antenna2, vis_per_source

    def forward(self, gains, antenna1, antenna2, vis_per_source):
        g1 = gains[antenna1, :, :]  # (row,  chan, source,)
        g2 = gains[antenna2, :, :]
        vis = jnp.sum(g1 * vis_per_source * g2.conj(), axis=-1)  # (row, chan)
        return vis


@dataclasses.dataclass(eq=False)
class OptimisationProblem:
    forward_model: ForwardModel
    num_iterations: int
    num_approx_steps: int

    def create_data(self, key, data_noise_scale: jax.Array, phase_error_scale: jax.Array, gain_noise_scale: jax.Array):
        keys = jax.random.split(key, 5)
        # Create true gains
        phases = phase_error_scale * jax.random.normal(
            keys[0],
            self.forward_model.gain_shape
        )
        gains_true = jnp.exp(1j * phases)
        gains_noise = gain_noise_scale * jax.lax.complex(
            jax.random.normal(keys[1], self.forward_model.gain_shape),
            jax.random.normal(keys[2], self.forward_model.gain_shape)
        )

        gains_true += gains_noise

        antenna1, antenna2, vis_per_source = self.forward_model.create_forward_data()

        data = self.forward_model.forward(gains_true, antenna1, antenna2, vis_per_source)
        noise = data_noise_scale * jax.lax.complex(
            jax.random.normal(keys[3], data.shape),
            jax.random.normal(keys[4], data.shape)
        )

        data += noise
        return gains_true, data, (antenna1, antenna2, vis_per_source)

    @partial(jax.jit, static_argnums=(0,))
    def multi_step_lm_solve(self, data, forward_args, params: CalibrationParams):
        def residuals(gains):
            residuals = data - self.forward_model.forward(gains, *forward_args)
            return residuals

        x = jnp.ones(self.forward_model.gain_shape, jnp.complex64)

        lm = MultiStepLevenbergMarquardt(
            residual_fn=residuals,
            num_iterations=self.num_iterations,
            num_approx_steps=self.num_approx_steps,
            p_any_improvement=params.p_any_improvement,
            p_less_newton=params.p_less_newton,
            p_sufficient_improvement=params.p_sufficient_improvement,
            p_more_newton=params.p_more_newton,
            c_more_newton=params.c_more_newton,
            c_less_newton=params.c_less_newton,
            delta=params.delta,
            mu1=params.mu1,
            mu_min=params.mu_min,
            verbose=True
        )
        state = lm.create_initial_state(x)
        state, diagnostics = lm.solve(state)
        return diagnostics.error[-1]


def run_optimisation(num_iterations: int,
                     num_approx_steps: int,
                     data_noise_scale: jax.Array,
                     phase_error_scale: jax.Array,
                     gain_noise_scale: jax.Array, num_search=1000):
    forward_model = ForwardModel(ant=2048, source=7, chan=1)
    optimisation_problem = OptimisationProblem(
        forward_model=forward_model,
        num_iterations=num_iterations,
        num_approx_steps=num_approx_steps
    )

    key = jax.random.PRNGKey(0)
    gains_true, data, forward_args = optimisation_problem.create_data(
        key,
        data_noise_scale,
        phase_error_scale,
        gain_noise_scale
    )

    def run(key, data, forward_args):
        def prior_model():
            p_any_improvement = yield Prior(
                tfpd.Uniform(low=0.0, high=1.),
                name='p_any_improvement'
            )
            p_less_newton = yield Prior(
                tfpd.Uniform(low=p_any_improvement, high=1.),
                name='p_less_newton'
            )
            p_sufficient_improvement = yield Prior(
                tfpd.Uniform(low=p_less_newton, high=1.),
                name='p_sufficient_improvement'
            )
            p_more_newton = yield Prior(
                tfpd.Uniform(low=p_sufficient_improvement, high=1.),
                name='p_more_newton'
            )
            c_more_newton = yield Prior(
                tfpd.Uniform(low=0.0, high=1.),
                name='c_more_newton'
            )
            c_less_newton = yield Prior(
                tfpd.Uniform(low=1., high=5.),
                name='c_less_newton'
            )
            delta = yield Prior(
                tfpd.Uniform(low=1., high=2.),
                name='delta'
            )

            mu_min = yield Prior(
                tfpd.Uniform(low=1e-5, high=1.),
                name='mu_min'
            )
            mu1 = yield Prior(
                tfpd.Uniform(low=mu_min, high=10.),
                name='mu1'
            )
            return CalibrationParams(
                p_any_improvement=p_any_improvement,
                p_less_newton=p_less_newton,
                p_sufficient_improvement=p_sufficient_improvement,
                p_more_newton=p_more_newton,
                c_more_newton=c_more_newton,
                c_less_newton=c_less_newton,
                delta=delta,
                mu1=mu1,
                mu_min=mu_min
            )

        def log_likelihood(params: CalibrationParams):
            return -optimisation_problem.multi_step_lm_solve(data, forward_args, params)

        model = Model(
            prior_model=prior_model,
            log_likelihood=log_likelihood
        )

        # brute-force check
        def body(state, key):
            U = model.sample_U(key)
            X = model.transform(U)
            log_likelihood = model.forward(U)
            return state, (U, X, log_likelihood)

        _, (U, X, log_likelihoods) = jax.lax.scan(body, None, jax.random.split(key, num_search))

        return U, X, log_likelihoods

        # # model.sanity_check(jax.random.PRNGKey(0), 5)
        # go = DefaultGlobalOptimisation(
        #     model=model
        # )
        # term_cond = GlobalOptimisationTerminationCondition(
        #     max_likelihood_evaluations=10
        # )
        # return go(key, term_cond)

    key = jax.random.PRNGKey(0)
    t0 = time.time()
    run_jit = jax.jit(run).lower(key, data, forward_args).compile()
    build_time = time.time() - t0
    print(f"Build time: {build_time}")
    t0 = time.time()
    # go_results = block_until_ready(run_jit(key, data, forward_args))
    _, X, log_L = block_until_ready(run_jit(key, data, forward_args))
    run_time = time.time() - t0
    print(f"Run time: {run_time}")

    # summary(go_results)
    #
    # save_pytree(go_results, 'go_results.json')
    #
    # go_results = load_pytree('go_results.json')
    # summary(go_results)


if __name__ == '__main__':
    run_optimisation(num_iterations=2, num_approx_steps=1, data_noise_scale=jnp.asarray(1e-3),
                     phase_error_scale=jnp.asarray(0.1), gain_noise_scale=jnp.asarray(0.1))
