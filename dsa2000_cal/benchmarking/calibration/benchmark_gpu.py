import os
from functools import partial

os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={8}"

from dsa2000_cal.calibration_step import calibration_step
from dsa2000_cal.probabilistic_models.gain_prior_models import GainPriorModel
from dsa2000_cal.solvers.cg import tree_add, tree_scalar_mul
from dsa2000_common.common.jvp_linear_op import JVPLinearOp
from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map
from dsa2000_common.common.fit_benchmark import fit_timings
import jax.numpy as jnp
from dsa2000_common.common.jax_utils import create_mesh
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.mixed_precision_utils import mp_policy
import itertools
import time
from typing import Dict, Any

import jax
import numpy as np

from dsa2000_cal.ops.residuals import compute_residual_TBC


def prepare_data_cal(D: int, T, C, BTs, BTm, BCs, BCm) -> Dict[str, Any]:
    num_antennas = 2048
    baseline_pairs = np.asarray(list(itertools.combinations(range(num_antennas), 2)),
                                dtype=np.int32)
    antenna1 = baseline_pairs[:, 0]
    antenna2 = baseline_pairs[:, 1]  # [B]

    sort_idxs = np.lexsort((antenna1, antenna2))
    antenna1 = antenna1[sort_idxs]
    antenna2 = antenna2[sort_idxs]

    B = antenna1.shape[0]
    vis_model = np.zeros((D, T // BTm, B, C // BCm, 2, 2), dtype=mp_policy.vis_dtype)
    vis_data = np.zeros((T // BTm, B, C // BCm, 2, 2), dtype=mp_policy.vis_dtype)
    weights = np.ones((T // BTm, B, C // BCm, 2, 2), dtype=mp_policy.weight_dtype)

    gain_probabilistic_model = GainPriorModel(
        num_source=D,
        num_ant=num_antennas,
        freqs=np.linspace(700e6, 800e6, C // BCs),
        times=np.linspace(0., 6, T // BTs),
        gain_stddev=2.,
        full_stokes=True,
        dd_type='phase',
        di_type='amplitude+clock',
        dd_dof=4,
        di_dof=1
    )
    return dict(
        vis_model=vis_model,
        vis_data=vis_data,
        weights=weights,
        antenna1=antenna1,
        antenna2=antenna2,
        gain_probabilistic_model=gain_probabilistic_model
    )


def build_sharded_entry_point_cal(backend):
    devices = jax.devices(backend)
    num_devices = len(devices)

    def entry_point(data):
        vis_model = data['vis_model']
        vis_data = data['vis_data']
        weights = data['weights']
        antenna1 = data['antenna1']
        antenna2 = data['antenna2']
        gain_probabilistic_model = data['gain_probabilistic_model']

        return calibration_step(
            params=None, vis_model=vis_model,
            vis_data=vis_data, weights=weights,
            antenna1=antenna1, antenna2=antenna2,
            gain_probabilistic_model=gain_probabilistic_model, verbose=False,
            backend=backend, num_B_shards=num_devices, num_C_shards=1, num_devices=num_devices,
            maxiter=1, maxiter_cg=1
        )

    return entry_point


def run_cal(T, C, BTs, BTm, BCs, BCm, backend, m, task, scheme):
    sharded_entry_point = build_sharded_entry_point_cal(backend)
    sharded_entry_point_jit = jax.jit(sharded_entry_point)
    # Run benchmarking over number of calibration directions
    shard_time_array = []
    d_array = []
    for D in range(1, 9):
        data = prepare_data_cal(D, T, C, BTs, BTm, BCs, BCm)
        sharded_entry_point_jit_compiled = sharded_entry_point_jit.lower(data).compile()
        t0 = time.time()
        for _ in range(m):
            jax.block_until_ready(sharded_entry_point_jit_compiled(data))
        t1 = time.time()
        dt = (t1 - t0) / m
        dsa_logger.info(f"{task} {scheme}: {backend} D={D}: {dt}")
        shard_time_array.append(dt)
        d_array.append(D)

    shard_time_array = np.array(shard_time_array)
    d_array = np.array(d_array)

    a, b, c = fit_timings(d_array, shard_time_array)
    dsa_logger.info(f"{task} {scheme}: {backend}: t(n) = {a:.4f} * n ** {b:.2f} + {c:.4f}")


def prepare_data(D: int, T, C, BTs, BTm, BCs, BCm) -> Dict[str, Any]:
    assert T % BTm == 0 and T % BTs == 0
    assert C % BCm == 0 and C % BCs == 0

    num_antennas = 2048
    baseline_pairs = np.asarray(list(itertools.combinations(range(num_antennas), 2)),
                                dtype=np.int32)
    antenna1 = baseline_pairs[:, 0]
    antenna2 = baseline_pairs[:, 1]  # [B]

    sort_idxs = np.lexsort((antenna1, antenna2))
    antenna1 = antenna1[sort_idxs]
    antenna2 = antenna2[sort_idxs]

    B = antenna1.shape[0]
    vis_model = np.zeros((D, T // BTm, B, C // BCm, 2, 2), dtype=mp_policy.vis_dtype)
    vis_data = np.zeros((T // BTm, B, C // BCm, 2, 2), dtype=mp_policy.vis_dtype)
    gains = np.zeros((D, T // BTs, num_antennas, C // BCs, 2, 2), dtype=mp_policy.gain_dtype)

    dsa_logger.info(f"Model size D * {vis_data.nbytes / D / 2 ** 20} MB, gain size D * {gains.nbytes / D / 2 ** 20} MB")

    return dict(
        vis_model=vis_model,
        vis_data=vis_data,
        gains=gains,
        antenna1=antenna1,
        antenna2=antenna2
    )


def entry_point_R(data):
    return compute_residual_TBC(**data)


def build_sharded_entry_point_R(devices):
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
        return entry_point_R(local_data)  # [ Tm, _B, Cm, 2, 2]

    return entry_point_sharded, mesh


def entry_point_JtR(data):
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
    return J.matvec(R, adjoint=True)


def build_sharded_entry_point_JtR(devices):
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
        return entry_point_JtR(local_data)  # [ Tm, _B, Cm, 2, 2]

    return entry_point_sharded, mesh


def entry_point_JtJp(data):
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
    p = jax.tree.map(jnp.ones_like, g)
    JTJv = J.matvec(J.matvec(p), adjoint=True)
    damping = jnp.asarray(1.)
    return tree_add(JTJv, tree_scalar_mul(damping, p))


def build_sharded_entry_point_JtJp(devices):
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
        return entry_point_JtJp(local_data)  # [ Tm, _B, Cm, 2, 2]

    return entry_point_sharded, mesh


def run(build_sharded_entry_point_fn, T, C, BTs, BTm, BCs, BCm, backend, m, task, scheme):
    devices = jax.devices(backend)
    sharded_entry_point, mesh = build_sharded_entry_point_fn(devices)
    sharded_entry_point_jit = jax.jit(sharded_entry_point)
    # Run benchmarking over number of calibration directions
    shard_time_array = []
    d_array = []
    for D in range(1, 9):
        data = prepare_data(D, T, C, BTs, BTm, BCs, BCm)
        sharded_entry_point_jit_compiled = sharded_entry_point_jit.lower(data).compile()
        t0 = time.time()
        for _ in range(m):
            jax.block_until_ready(sharded_entry_point_jit_compiled(data))
        t1 = time.time()
        dt = (t1 - t0) / m
        dsa_logger.info(f"{task} {scheme}: {backend} D={D}: {dt}")
        shard_time_array.append(dt)
        d_array.append(D)

    shard_time_array = np.array(shard_time_array)
    d_array = np.array(d_array)

    a, b, c = fit_timings(d_array, shard_time_array)
    dsa_logger.info(f"{task} {scheme}: {backend}: t(n) = {a:.4f} * n ** {b:.2f} + {c:.4f}")


def main():
    for scheme, (T, C, BTs, BTm, BCs, BCm) in zip(
            ['Full-Avg. per-GPU', 'Non-Avg. per-GPU', 'Non-Avg. all-GPU'],
            [
                (4, 4, 4, 4, 4, 4),
                (4, 4, 4, 1, 4, 1),
                (4, 40, 4, 1, 40, 1)
            ]
    ):
        for backend in ['cpu', 'cuda']:
            for task, build_sharded_entry_point_fn in zip(
                    ['R(x)', 'J^T.R(x)', 'J^T.J.p'],
                    [build_sharded_entry_point_R, build_sharded_entry_point_JtR, build_sharded_entry_point_JtJp]):
                run(build_sharded_entry_point_fn, T=T, C=C, BTs=BTs, BTm=BTm, BCs=BCs, BCm=BCm, backend=backend, m=10,
                    task=task, scheme=scheme)
            run_cal(
                T=T,
                C=C,
                BTs=BTs,
                BCs=BCs,
                BTm=BTm,
                BCm=BCm,
                backend=backend,
                m=10,
                task='LM-Solver 1-iter 1-CG-iter',
                scheme=scheme
            )


if __name__ == '__main__':
    main()
