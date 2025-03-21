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
    jax.jit, static_argnames=['verbose', 'num_devices', 'backend']
)
def calibration_step(params: Any | None, vis_model: ComplexArray, vis_data: ComplexArray, weights: FloatArray,
                     antenna1: IntArray, antenna2: FloatArray,
                     gain_probabilistic_model: AbstractGainPriorModel, verbose: bool = False,
                     num_devices: int = 1, backend: str = 'cpu'):
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

    Returns:
        params, gains, diagnostics
    """

    if np.shape(vis_data) != np.shape(weights):
        raise ValueError(
            f"Visibilities and weights must have the same shape, got {np.shape(vis_data)} and {np.shape(weights)}")

    devices = jax.local_devices(backend=backend)[:num_devices]
    mesh = create_mesh((len(devices),), ('B',), devices)
    P = PartitionSpec
    in_specs = (
        P(),
        P(None, None, 'B'),
        P(None, 'B'),
        P(None, 'B'),
        P('B'),
        P('B')
    )
    out_specs = P(None, 'B')

    @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    # Create residual_fn
    def residual_fn(
            params: Any,
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
        residuals *= jnp.sqrt(weights)  # [Tm, B, Cm[,2,2]]
        return residuals

    # Get solver state
    if params is None:
        params = gain_probabilistic_model.get_init_params(jax.random.PRNGKey(0))

    params, diagnostics = lm_solver(
        residual_fn=residual_fn,
        x0=params,
        args=(vis_model, vis_data, weights, antenna1, antenna2),
        gtol=1e-4,
        verbose=verbose
    )
    gains = gain_probabilistic_model.compute_gains(params)

    return params, gains, diagnostics
