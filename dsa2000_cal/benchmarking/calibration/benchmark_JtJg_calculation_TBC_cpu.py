import os
from functools import partial

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={8}"

from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map
from dsa2000_common.common.fit_benchmark import fit_timings

from dsa2000_common.common.jax_utils import create_mesh
from dsa2000_common.common.jvp_linear_op import JVPLinearOp
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
    vis_model = np.zeros((D, T // Tm, B, C // Cm, 2, 2), dtype=mp_policy.vis_dtype)
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
    vis_model = data['vis_model']
    vis_data = data['vis_data']
    gains = data['gains']
    antenna1 = data['antenna1']
    antenna2 = data['antenna2']

    def fn(gains):
        res = compute_residual_TBC(vis_model=vis_model, vis_data=vis_data, gains=gains,
                                   antenna1=antenna1, antenna2=antenna2)
        return res

    J_bare = JVPLinearOp(fn, linearize=False)
    J = J_bare(gains)
    R = fn(gains)
    g = J.matvec(R, adjoint=True)
    return J.matvec(J.matvec(g), adjoint=True)


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


def run(T, C, Ts, Tm, Cs, Cm, backend, m=10, task='J^T.J.g(x)'):
    devices = jax.devices(backend)
    sharded_entry_point, mesh = build_sharded_entry_point(devices)
    sharded_entry_point_jit = jax.jit(sharded_entry_point)
    # Run benchmarking over number of calibration directions
    shard_time_array = []
    d_array = []
    for D in range(1, 9):
        data = prepare_data(D, T, C, Ts, Tm, Cs, Cm)
        sharded_entry_point_jit_compiled = sharded_entry_point_jit.lower(data).compile()
        t0 = time.time()
        for _ in range(m):
            jax.block_until_ready(sharded_entry_point_jit_compiled(data))
        t1 = time.time()
        dt = (t1 - t0) / m
        dsa_logger.info(f"{task}: {backend} D={D}: {dt}")
        shard_time_array.append(dt)
        d_array.append(D)

    shard_time_array = np.array(shard_time_array)
    d_array = np.array(d_array)

    a, b, c = fit_timings(d_array, shard_time_array)
    dsa_logger.info(f"{task}: {backend}: t(n) = {a:.4f} * n ** {b:.2f} + {c:.4f}")


def main():
    # run(T=4, C=40, Ts=4, Tm=4, Cs=40, Cm=40, backend='cpu', m=10, task='J^T.J.g(x) [all-GPU]')
    run(T=4, C=4, Ts=4, Tm=4, Cs=4, Cm=4, backend='cpu', m=10, task='J^T.J.g(x) [per-GPU]')


if __name__ == '__main__':
    main()
