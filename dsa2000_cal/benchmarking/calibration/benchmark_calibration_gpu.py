import os

os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'

from dsa2000_cal.calibration_step import calibration_step
from dsa2000_cal.probabilistic_models.gain_prior_models import GainPriorModel
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.mixed_precision_utils import mp_policy
import itertools
import time
from typing import Dict, Any

import jax
import numpy as np


def prepare_data(D: int, Ts, Tm, Cs, Cm) -> Dict[str, Any]:
    num_antennas = 2048
    baseline_pairs = np.asarray(list(itertools.combinations(range(num_antennas), 2)),
                                dtype=np.int32)
    antenna1 = baseline_pairs[:, 0]
    antenna2 = baseline_pairs[:, 1]  # [B]

    sort_idxs = np.lexsort((antenna1, antenna2))
    antenna1 = antenna1[sort_idxs]
    antenna2 = antenna2[sort_idxs]

    B = antenna1.shape[0]
    vis_model = np.zeros((D, Tm, B, Cm, 2, 2), dtype=mp_policy.vis_dtype)
    vis_data = np.zeros((Ts, B, Cs, 2, 2), dtype=mp_policy.vis_dtype)
    weights = np.ones((Ts, B, Cs, 2, 2), dtype=mp_policy.weight_dtype)

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
        backend='cuda', num_B_shards=1, num_C_shards=1, num_devices=1,
        maxiter=1, maxiter_cg=1
    )


def main():
    gpus = jax.devices("cuda")
    print(f"Found {len(gpus)} GPUs, using the first one.")
    gpu = gpus[0]

    entry_point_jit = jax.jit(entry_point)
    # Run benchmarking over number of calibration directions
    time_array = []
    d_array = []
    for D in range(1, 9):
        data = prepare_data(D, Ts=1, Tm=1, Cs=1, Cm=1)

        data = jax.device_put(data, device=gpu)
        entry_point_jit_compiled = entry_point_jit.lower(data).compile()
        t0 = time.time()
        for _ in range(3):
            jax.block_until_ready(entry_point_jit_compiled(data))
        t1 = time.time()
        dt = (t1 - t0) / 3
        dsa_logger.info(f"Calibration Single Iteration Single CG: CPU D={D}: {dt}")
        time_array.append(dt)
        d_array.append(D)

    # Fit line to data using scipy
    time_array = np.array(time_array)
    d_array = np.array(d_array)
    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(lambda x, a, b: a * x ** b, d_array, time_array)
    dsa_logger.info(f"Fit: {popt}")


if __name__ == '__main__':
    main()
