import os

os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={8}"

from dsa2000_common.common.fit_benchmark import fit_timings

from dsa2000_cal.calibration_step import calibration_step
from dsa2000_cal.probabilistic_models.gain_prior_models import GainPriorModel
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.mixed_precision_utils import mp_policy
import itertools
import time
from typing import Dict, Any

import jax
import numpy as np


def prepare_data(D: int, T, C, Ts, Tm, Cs, Cm) -> Dict[str, Any]:
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
    weights = np.ones((T // Tm, B, C // Cm, 2, 2), dtype=mp_policy.weight_dtype)

    gain_probabilistic_model = GainPriorModel(
        num_source=D,
        num_ant=num_antennas,
        freqs=np.linspace(700e6, 800e6, Cm),
        times=np.linspace(0., 6, Tm),
        gain_stddev=2.,
        full_stokes=True,
        dd_type='unconstrained',
        di_type='unconstrained',
        dd_dof=1,
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


def build_sharded_entry_point(backend):
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
            params=None, vis_model=vis_model, vis_data=vis_data, weights=weights,
            antenna1=antenna1, antenna2=antenna2,
            gain_probabilistic_model=gain_probabilistic_model, verbose=False,
            backend=backend, num_B_shards=num_devices, num_C_shards=1, num_devices=num_devices,
            maxiter=1, maxiter_cg=1
        )

    return entry_point


def run(T, C, Ts, Tm, Cs, Cm, backend, m=10, task='LMSolver-1iter-1CG'):
    sharded_entry_point = build_sharded_entry_point(backend)
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
    # run(T=4, C=40, Ts=4, Tm=4, Cs=40, Cm=40, backend='cuda', m=10, task='LMSolver-1iter-1CG [all-GPU]')
    # run(T=4, C=40, Ts=4, Tm=4, Cs=40, Cm=40, backend='cpu', m=10, task='LMSolver-1iter-1CG [all-GPU]')
    run(T=4, C=4, Ts=4, Tm=4, Cs=4, Cm=4, backend='cuda', m=10, task='LMSolver-1iter-1CG [Full.Avg. per-GPU]')
    run(T=4, C=4, Ts=4, Tm=4, Cs=4, Cm=4, backend='cpu', m=10, task='LMSolver-1iter-1CG [Full.Avg. per-GPU]')


if __name__ == '__main__':
    main()
