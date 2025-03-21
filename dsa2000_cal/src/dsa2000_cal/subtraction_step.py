from functools import partial
from typing import Any

import jax
from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map

from dsa2000_cal.ops.residuals import compute_residual_TBC
from dsa2000_common.common.array_types import ComplexArray, FloatArray, IntArray
from dsa2000_common.common.jax_utils import create_mesh


@partial(
    jax.jit, static_argnames=['num_devices', 'backend']
)
def subtraction_step(gains: Any | None, vis_model: ComplexArray, vis_data: ComplexArray,
                     antenna1: IntArray, antenna2: FloatArray,
                     num_devices: int = 1, backend: str = 'cpu'):
    """
    Perform a single calibration step.

    Args:
        gains: [D, Tm, A, Cm[, 2, 2]]
        vis_model: [D, Tm, B, Cm[, 2, 2]]
        vis_data: [T, B, C[, 2, 2]] Tm and Cm divide T and C.
        antenna1: [B] antenna 1
        antenna2: [B] antenna 2

    Returns:
        params, gains, diagnostics
    """

    devices = jax.local_devices(backend=backend)[:num_devices]
    mesh = create_mesh((len(devices),), ('B',), devices)
    P = PartitionSpec
    in_specs = (
        P(),
        P(None, None, 'B'),
        P(None, 'B'),
        P('B'),
        P('B')
    )
    out_specs = P(None, 'B')

    @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    # Create residual_fn
    def subtraction_fn(
            gains: Any,
            vis_model: ComplexArray,
            vis_data: ComplexArray,
            antenna1: IntArray,
            antenna2: IntArray
    ):
        residuals = compute_residual_TBC(
            vis_model=vis_model,
            vis_data=vis_data,
            gains=gains,
            antenna1=antenna1,
            antenna2=antenna2
        )
        return residuals

    residuals = subtraction_fn(
        gains,
        vis_model,
        vis_data,
        antenna1,
        antenna2
    )

    return residuals
