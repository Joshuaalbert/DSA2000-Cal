from functools import partial
from typing import Any

import jax
import numpy as np
from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map

from dsa2000_cal.ops.residuals import compute_residual_TBC
from dsa2000_common.common.array_types import ComplexArray, FloatArray, IntArray
from dsa2000_common.common.jax_utils import create_mesh


@partial(
    jax.jit, static_argnames=['num_devices', 'backend', 'num_B_shards', 'num_C_shards']
)
def subtraction_step(gains: Any | None, vis_model: ComplexArray, vis_data: ComplexArray,
                     antenna1: IntArray, antenna2: FloatArray,
                     num_devices: int = 1, backend: str = 'cpu', num_B_shards: int = 1, num_C_shards: int = 1):
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

    if num_B_shards * num_C_shards != num_devices:
        raise ValueError(f"Sharding requirement not met: B_shard_size * C_shard_size == num_devices")

    T, B, C = np.shape(vis_data)[:3]
    if B < num_B_shards:
        raise ValueError(f"Sharding requirement not met: B < num_B_shards")
    if C < num_C_shards:
        raise ValueError(f"Sharding requirement not met: C < num_C_shards")
    D, Tm, B, Cm = np.shape(vis_model)[:4]
    if (np.shape(gains)[0] != D or np.shape(gains)[1] != Tm or np.shape(gains)[3] != Cm):
        raise ValueError(f"gains don't match model vis shape.")

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
    in_specs = (
        P(None, None, None, 'C') if no_rep_needed else P(),
        P(None, None, 'B', 'C') if no_rep_needed else P(None, None, 'B'),
        P(None, 'B', 'C'),
        P('B'),
        P('B')
    )
    out_specs = P(None, 'B', 'C')

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
