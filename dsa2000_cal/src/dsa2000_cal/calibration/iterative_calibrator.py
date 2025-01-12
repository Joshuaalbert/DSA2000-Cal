import dataclasses
from functools import partial
from typing import Generator, NamedTuple, List

import jax
from dsa2000_cal.common.vec_utils import kron_product

from dsa2000_cal.common.mixed_precision_utils import mp_policy

from dsa2000_cal.common.jax_utils import simple_broadcast

from dsa2000_cal.common.array_types import IntArray, FloatArray, BoolArray, ComplexArray

from dsa2000_cal.calibration.solvers.multi_step_lm import MultiStepLevenbergMarquardtState

from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import AbstractGainPriorModel
from essm_jax.essm import ExtendedStateSpaceModel, IncrementalFilterState

import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp
import numpy as np
tfpd = tfp.distributions
class Data(NamedTuple):
    """
    Data structure that contains the data that is used in the calibration algorithm.
    D - number of directions
    T - number of times
    B - number of baselines
    C - number of channels
    num_coh - number of coherence products
    """
    coherencies: List[str] # list of coherencies of length num_coh
    vis_model: ComplexArray # [D, T, B, C, num_coh]
    vis_data: ComplexArray # [T, B, C, num_coh]
    weights: FloatArray # [T, B, C, num_coh]
    flags: BoolArray # [T, B, C, num_coh]
    freqs: FloatArray # [C]
    times: FloatArray # [T]
    antenna1: IntArray # [B]
    antenna2: IntArray # [B]

class SubtractedData(NamedTuple):
    residuals: ComplexArray # [T, B, C, num_coh]

class State(NamedTuple):
    state: MultiStepLevenbergMarquardtState | None = None

@dataclasses.dataclass(eq=False)
class IterativeCalibrator:
    """
    Implements a streaming iterative calibration algorithm, that uses several techniques to make calibration realtime:
    1. Uses an extended Kalman filter to forcast the solutions, which means fewer iterations are required.
    2. Imposes a temporal and spectral regularization on the solutions, which means that the solutions are more stable.
    3. Piecewise-constant solutions: uses 1 solution per Ts time steps, and Cs channels. This decreases the degrees of freedom.
    4. Averaging: Averages data (and model) down to Tm and Cm, which must be divisors of Ts and Cs. This reduces the number of computations.
    """

    gain_probabilistic_model: AbstractGainPriorModel
    full_stokes: bool
    num_ant: int
    verbose: bool = False

    def run(self, gen: Generator[Data, SubtractedData]):
        gen_response: SubtractedData | None = None
        while True:
            try:
                data = gen.send(gen_response)
            except StopIteration as e:
                break

            # Run calibration

    def filter(self, data: Data, filter_state: IncrementalFilterState):


        def transition_fn(z, t, t_next, map_estimate):
            dt = t_next - t
            sigma = map_estimate['sigma']
            x_mean = z # [n]
            x_std = sigma * jnp.sqrt(dt)
            return tfpd.MultivariateNormalDiag(loc=x_mean, scale_diag=x_std)

        def observation_fn(z, t, map_estimate):
            # Convert z into gains, and then compute the likelihood
            n = np.shape(z)[0]
            gains = jax.lax.complex(z[:n//2], z[n//2:])
            # compute visibilities from vis model
            vis_model = apply_gains_to_model_vis(data.vis_model, gains, data.antenna1, data.antenna2) # [Tm, B, Cm, num_coh]
            vis_model = jnp.ravel(vis_model) # [Tm*B*Cm*num_coh]
            x = jnp.concatenate([jnp.real(vis_model), jnp.imag(vis_model)], axis=-1) # [N]
            uncert = map_estimate['uncert']
            return tfpd.MultivariateNormalDiag(loc=x, scale_diag=uncert)


        initial_state_prior = tfpd.MultivariateNormalDiag(loc=jnp.zeros((1,)), scale_diag=jnp.ones((1,)))

        essm = ExtendedStateSpaceModel(
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            initial_state_prior=initial_state_prior
        )

        if filter_state is None:
            filter_state = essm.create_initial_filter_state(t0=data.times.min())

        filter_state = essm.incremental_predict(filter_state, map_estimate, t_next=data.times.mean())
        obs = jnp.ravel(data.vis_data) # [T*B*C*num_coh]
        obs = jnp.concatenate([jnp.real(obs), jnp.imag(obs)], axis=-1) # [N]
        filter_state, obs_dist = essm.incremental_update(filter_state, obs, map_estimate)


def apply_gains_to_model_vis(vis_model, gains, antenna1, antenna2):
    """
    Compute the residual between the model visibilities and the observed visibilities.

    Args:
        vis_model: [D, Tm, B, Cm[,2,2]] the model visibilities per direction
        gains: [D, Tm, A, Cm[,2,2]] the gains
        antenna1: [B] the antenna1
        antenna2: [B] the antenna2

    Returns:
        [Tm, B, Cm[, 2, 2]] the residuals
    """

    def body_fn(accumulate, x):
        vis_model, gains = x

        g1 = gains[:, antenna1, :, ...]  # [Tm, B, Cm[, 2, 2]]
        g2 = gains[:, antenna2, :, ...]  # [Tm, B, Cm[, 2, 2]]

        @partial(
            simple_broadcast,  # [Tm,B,Cm,...]
            leading_dims=3
        )
        def apply_gains(g1, g2, vis):
            if np.shape(g1) != np.shape(g1):
                raise ValueError("Gains must have the same shape.")
            if np.shape(vis) != np.shape(g1):
                raise ValueError("Gains and visibilities must have the same shape.")
            if np.shape(g1) == (2, 2):
                return mp_policy.cast_to_vis(kron_product(g1, vis, g2.conj().T))
            elif np.shape(g1) == ():
                return mp_policy.cast_to_vis(g1 * vis * g2.conj())
            else:
                raise ValueError(f"Invalid shape: {np.shape(g1)}")

        delta_vis = apply_gains(g1, g2, vis_model)  # [Tm, B, Cm[, 2, 2]]
        return accumulate + delta_vis, ()

    accumulate = jnp.zeros(np.shape(vis_model)[1:], dtype=vis_model.dtype)
    accumulate, _ = jax.lax.scan(body_fn, accumulate, (vis_model, gains))
    return accumulate
