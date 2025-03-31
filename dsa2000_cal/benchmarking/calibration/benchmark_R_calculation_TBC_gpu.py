import os
from functools import partial

os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'

from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map
from dsa2000_common.common.fit_benchmark import fit_timings

from dsa2000_common.common.jax_utils import create_mesh
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.mixed_precision_utils import mp_policy
import itertools
import time
from typing import Dict, Any

import jax
import numpy as np

from dsa2000_cal.ops.residuals import compute_residual_TBC


def prepare_data(D: int, T, C, Ts, Tm, Cs, Cm) -> Dict[str, Any]:
    assert T % Tm == 0 and T % Ts == 0
    assert C % Cm == 0 and C % Cs == 0

    num_antennas = 2048
    baseline_pairs = np.asarray(list(itertools.combinations(range(num_antennas), 2)),
                                dtype=np.int32)
    antenna1 = baseline_pairs[:, 0]
    antenna2 = baseline_pairs[:, 1]  # [B]

    sort_idxs = np.lexsort((antenna1, antenna2))
    antenna1 = antenna1[sort_idxs]
    antenna2 = antenna2[sort_idxs]

    B = antenna1.shape[0]
    vis_model = np.zeros((D, T // Tm, B, C // Tm, 2, 2), dtype=mp_policy.vis_dtype)
    vis_data = np.zeros((T // Tm, B, C // Cm, 2, 2), dtype=mp_policy.vis_dtype)
    gains = np.zeros((D, T // Ts, num_antennas, C // Cs, 2, 2), dtype=mp_policy.gain_dtype)

    return dict(
        vis_model=vis_model,
        vis_data=vis_data,
        gains=gains,
        antenna1=antenna1,
        antenna2=antenna2
    )


def entry_point(data):
    return compute_residual_TBC(**data)


def build_sharded_entry_point(devices):
    mesh = create_mesh((len(devices),), ('B',), devices)
    P = PartitionSpec
    in_specs = dict(
        vis_model=P(None, None, 'B'),
        vis_data=P(None, 'B'),
        gains=P(),
        antenna1=P('B'),
        antenna2=P('B')
    )
    out_specs = P('B')

    @partial(shard_map, mesh=mesh, in_specs=(in_specs,), out_specs=out_specs)
    def entry_point_sharded(local_data):
        return entry_point(local_data)  # [ Tm, _B, Cm, 2, 2]

    return entry_point_sharded, mesh


def main():
    gpus = jax.devices("cuda")
    sharded_entry_point, mesh = build_sharded_entry_point(gpus)
    sharded_entry_point_jit = jax.jit(sharded_entry_point)
    # Run benchmarking over number of calibration directions
    shard_time_array = []
    d_array = []
    for D in range(1, 9):
        data = prepare_data(D, T=4, C=40, Ts=4, Tm=4, Cs=40, Cm=40)

        sharded_entry_point_jit_compiled = sharded_entry_point_jit.lower(data).compile()
        t0 = time.time()
        for _ in range(10):
            jax.block_until_ready(sharded_entry_point_jit_compiled(data))
        t1 = time.time()
        dt = (t1 - t0) / 10
        dsa_logger.info(f"TBC: Residual: GPU D={D}: {dt}")
        shard_time_array.append(dt)
        d_array.append(D)

    shard_time_array = np.array(shard_time_array)
    d_array = np.array(d_array)

    a, b, c = fit_timings(d_array, shard_time_array)
    dsa_logger.info(f"Fit GPU: t(n) = {a:.4f} * n ** {b:.2f} + {c:.4f}")

    cpus = jax.devices("cpu")
    sharded_entry_point, mesh = build_sharded_entry_point(cpus)
    sharded_entry_point_jit = jax.jit(sharded_entry_point)
    # Run benchmarking over number of calibration directions
    shard_time_array = []
    d_array = []
    for D in range(1, 9):
        data = prepare_data(D, T=4, C=40, Ts=4, Tm=4, Cs=40, Cm=40)

        sharded_entry_point_jit_compiled = sharded_entry_point_jit.lower(data).compile()
        t0 = time.time()
        for _ in range(10):
            jax.block_until_ready(sharded_entry_point_jit_compiled(data))
        t1 = time.time()
        dt = (t1 - t0) / 10
        dsa_logger.info(f"TBC: Residual: CPU D={D}: {dt}")
        shard_time_array.append(dt)
        d_array.append(D)

    shard_time_array = np.array(shard_time_array)
    d_array = np.array(d_array)

    a, b, c = fit_timings(d_array, shard_time_array)
    dsa_logger.info(f"Fit CPU: t(n) = {a:.4f} * n ** {b:.2f} + {c:.4f}")


if __name__ == '__main__':
    main()
