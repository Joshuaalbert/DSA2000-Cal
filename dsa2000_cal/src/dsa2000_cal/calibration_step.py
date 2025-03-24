from functools import partial
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp
from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map

from dsa2000_cal.ops.residuals import compute_residual_TBC
from dsa2000_cal.probabilistic_models.gain_prior_models import AbstractGainPriorModel
from dsa2000_cal.solvers.multi_step_lm import lm_solver
from dsa2000_common.common.array_types import ComplexArray, FloatArray, IntArray
from dsa2000_common.common.jax_utils import create_mesh


@partial(
    jax.jit, static_argnames=['verbose', 'num_devices', 'backend', 'num_B_shards', 'num_C_shards']
)
def calibration_step(params: Any | None, vis_model: ComplexArray, vis_data: ComplexArray, weights: FloatArray,
                     antenna1: IntArray, antenna2: FloatArray,
                     gain_probabilistic_model: AbstractGainPriorModel, verbose: bool = False,
                     num_devices: int = 1, backend: str = 'cpu', num_B_shards: int = 1, num_C_shards: int = 1):
    """
    Perform a single calibration step.

    Args:
        params: Possible initial guesses.
        vis_model: [D, Tm, B, Cm[, 2, 2]]
        vis_data: [T, B, C[, 2, 2]] Tm and Cm divide T and C.
        weights: [T, B, C[, 2, 2]] flagged vis imply weights of zero.
        antenna1: [B] antenna 1
        antenna2: [B] antenna 2
        gain_probabilistic_model: the gain model
        verbose: whether to print
        num_devices: number of devices to use
        backend: the backend to use
        num_B_shards: size of B shard
        num_C_shards: size of C shard

    Requirement 1: B * C % num_devices == 0
    Requirement 2: B_shard_size * C_shard_size == num_devices

    Returns:
        params, gains, diagnostics
    """

    if np.shape(vis_data) != np.shape(weights):
        raise ValueError(
            f"Visibilities and weights must have the same shape, got {np.shape(vis_data)} and {np.shape(weights)}")

    if num_B_shards * num_C_shards != num_devices:
        raise ValueError(f"Sharding requirement not met: B_shard_size * C_shard_size == num_devices")

    T, B, C = np.shape(vis_data)[:3]
    if B < num_B_shards:
        raise ValueError(f"Sharding requirement not met: B < num_B_shards")
    if C < num_C_shards:
        raise ValueError(f"Sharding requirement not met: C < num_C_shards")

    D, Tm, B, Cm = np.shape(vis_model)[:4]
    if (gain_probabilistic_model.gain_shape()[0] != D or gain_probabilistic_model.gain_shape()[1] != Tm or
            gain_probabilistic_model.gain_shape()[3] != Cm):
        raise ValueError(f"gain shape doesn't match vis model")

    if T % Tm != 0:
        raise ValueError(f"Tm must be divide T.")
    if C % Cm != 0:
        raise ValueError(f"Cm must be divide c.")
    no_rep_needed = T == Tm and C == Cm

    if B * C % num_devices != 0:
        raise ValueError(f"The number of devices {num_devices} is not a multiple of B ({B}) * C ({C}) = {B * C}.")

    devices = jax.local_devices(backend=backend)[:num_devices]
    mesh = create_mesh((num_B_shards, num_C_shards), ('B', 'C'), devices)
    P = PartitionSpec
    gain_probabilistic_model_spec = gain_probabilistic_model.get_spec(
        freq_spec=P('C') if no_rep_needed else P(), time_spec=P())

    in_specs = (
        P(),
        gain_probabilistic_model_spec,
        P(None, None, 'B', 'C') if no_rep_needed else P(None, None, 'B'),
        P(None, 'B', 'C'),
        P(None, 'B', 'C'),
        P('B'),
        P('B')
    )
    out_specs = P(None, 'B', 'C')

    @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    # Create residual_fn
    def residual_fn(
            params: Any,
            gain_probabilistic_model,
            vis_model: ComplexArray,
            vis_data: ComplexArray,
            weights: FloatArray,
            antenna1: IntArray,
            antenna2: IntArray
    ):
        gains = gain_probabilistic_model.compute_gains(params)
        residuals = compute_residual_TBC(
            vis_model=vis_model,
            vis_data=vis_data,
            gains=gains,
            antenna1=antenna1,
            antenna2=antenna2
        )
        residuals *= jnp.sqrt(weights)  # [T, B, C[,2,2]]
        return residuals

    # Get initial parameters
    # Note: in some cases we could shard these, but since this is hard to know a priori we must replicate.
    # TODO: maybe there is a way to do this from the context.
    if params is None:
        in_specs = (
            P(),
            gain_probabilistic_model_spec
        )
        out_specs = P()

        @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
        def get_init_params(key, gain_probabilistic_model):
            params = gain_probabilistic_model.get_init_params(key)
            return params

        params = get_init_params(jax.random.PRNGKey(0), gain_probabilistic_model)

    params, diagnostics = lm_solver(
        residual_fn=residual_fn,
        x0=params,
        args=(gain_probabilistic_model, vis_model, vis_data, weights, antenna1, antenna2),
        gtol=1e-4,
        verbose=verbose
    )
    gains = gain_probabilistic_model.compute_gains(params)

    return params, gains, diagnostics
