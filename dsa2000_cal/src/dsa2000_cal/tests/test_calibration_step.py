import os

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={4}"

import jax
from jax import numpy as jnp

from dsa2000_cal.calibration_step import calibration_step
from dsa2000_cal.probabilistic_models.gain_prior_models import GainPriorModel
from dsa2000_common.common.mixed_precision_utils import mp_policy


def test_calibration_step():
    D = 1

    for (T, C) in [(1, 1), (2, 2), (4, 4)]:
        for (BTm, BCm) in [(1, 1), (2, 2), (4, 4)]:
            for (BTs, BCs) in [(1, 1), (2, 2), (4, 4)]:
                if BTs < BTm or BCs < BCm:
                    continue
                if T % BTm != 0 or C % BCm != 0:
                    continue
                if T % BTs != 0 or C % BCs != 0:
                    continue
                if T // BTm == 0 or C // BCm == 0:
                    continue
                if T // BTs == 0 or C // BCs == 0:
                    continue
                print(f"T={T}, C={C}, BTm={BTm}, BCm={BCm}, BTs={BTs}, BCs={BCs}")
                gain_model = GainPriorModel(
                    num_source=D,
                    num_ant=10,
                    freqs=jnp.linspace(700e6, 800e6, C // BCs),
                    times=jnp.linspace(0., 6, T // BTs),
                    gain_stddev=2.,
                    full_stokes=True,
                    dd_type='unconstrained',
                    di_type='unconstrained',
                    dd_dof=1,
                    di_dof=1
                )

                A = 352

                B = A * (A - 1) // 2
                vis_data = jnp.ones((T // BTm, B, C // BCm, 2, 2), dtype=mp_policy.vis_dtype)
                vis_model = jnp.ones((D, T // BTm, B, C // BCm, 2, 2), dtype=mp_policy.vis_dtype)
                weights = jnp.ones((T // BTm, B, C // BCm, 2, 2), dtype=mp_policy.weight_dtype)
                antenna1 = jnp.ones((B,), dtype=mp_policy.index_dtype)
                antenna2 = jnp.ones((B,), dtype=mp_policy.index_dtype)
                params, gains, diagnostics = jax.block_until_ready(
                    calibration_step(params=None, vis_model=vis_model, vis_data=vis_data, weights=weights,
                                     antenna1=antenna1, antenna2=antenna2, gain_probabilistic_model=gain_model,
                                     verbose=True, num_devices=4, maxiter=1, maxiter_cg=1)
                )

                params, gains, diagnostics = jax.block_until_ready(
                    calibration_step(params=params, vis_model=vis_model, vis_data=vis_data, weights=weights,
                                     antenna1=antenna1, antenna2=antenna2, gain_probabilistic_model=gain_model,
                                     verbose=True, num_devices=4, maxiter=1, maxiter_cg=1)
                )
