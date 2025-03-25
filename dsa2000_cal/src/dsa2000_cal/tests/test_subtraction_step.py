import os

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import jax
from jax import numpy as jnp

from dsa2000_cal.probabilistic_models.gain_prior_models import GainPriorModel
from dsa2000_cal.subtraction_step import subtraction_step
from dsa2000_common.common.mixed_precision_utils import mp_policy


def test_subtraction_step():
    T = 4
    C = 4
    Tm = 4
    Cm = 4

    gain_model = GainPriorModel(
        num_source=1,
        num_ant=10,
        freqs=jnp.linspace(700e6, 800e6, Cm),
        times=jnp.linspace(0., 6, Tm),
        gain_stddev=2.,
        full_stokes=True,
        dd_type='unconstrained',
        di_type='unconstrained',
        dd_dof=1,
        di_dof=1
    )

    init_params = gain_model.get_init_params(jax.random.PRNGKey(0))
    gains = gain_model.compute_gains(init_params)

    B = 2048 * 2047 // 2
    vis_data = jnp.ones((T, B, C, 2, 2), dtype=mp_policy.vis_dtype)
    vis_model = jnp.ones((1, Tm, B, Cm, 2, 2), dtype=mp_policy.vis_dtype)
    antenna1 = jnp.ones((B,), dtype=mp_policy.index_dtype)
    antenna2 = jnp.ones((B,), dtype=mp_policy.index_dtype)
    residuals = jax.block_until_ready(
        subtraction_step(
            gains=gains,
            vis_model=vis_model,
            vis_data=vis_data,
            antenna1=antenna1,
            antenna2=antenna2,
            num_devices=8,
            num_B_shards=4,
            num_C_shards=2
        )
    )

    T = 4
    C = 4
    Tm = 1
    Cm = 1

    gain_model = GainPriorModel(
        num_source=1,
        num_ant=10,
        freqs=jnp.linspace(700e6, 800e6, Cm),
        times=jnp.linspace(0., 6, Tm),
        gain_stddev=2.,
        full_stokes=True,
        dd_type='unconstrained',
        di_type='unconstrained',
        dd_dof=1,
        di_dof=1
    )

    init_params = gain_model.get_init_params(jax.random.PRNGKey(0))
    gains = gain_model.compute_gains(init_params)

    B = 2048 * 2047 // 2
    vis_data = jnp.ones((T, B, C, 2, 2), dtype=mp_policy.vis_dtype)
    vis_model = jnp.ones((1, Tm, B, Cm, 2, 2), dtype=mp_policy.vis_dtype)
    antenna1 = jnp.ones((B,), dtype=mp_policy.index_dtype)
    antenna2 = jnp.ones((B,), dtype=mp_policy.index_dtype)
    residuals = jax.block_until_ready(
        subtraction_step(
            gains=gains,
            vis_model=vis_model,
            vis_data=vis_data,
            antenna1=antenna1,
            antenna2=antenna2,
            num_devices=8,
            num_B_shards=4,
            num_C_shards=2
        )
    )

    T = 1
    C = 1
    Tm = 1
    Cm = 1

    gain_model = GainPriorModel(
        num_source=1,
        num_ant=10,
        freqs=jnp.linspace(700e6, 800e6, Cm),
        times=jnp.linspace(0., 6, Tm),
        gain_stddev=2.,
        full_stokes=True,
        dd_type='unconstrained',
        di_type='unconstrained',
        dd_dof=1,
        di_dof=1
    )

    init_params = gain_model.get_init_params(jax.random.PRNGKey(0))
    gains = gain_model.compute_gains(init_params)

    B = 2048 * 2047 // 2
    vis_data = jnp.ones((T, B, C, 2, 2), dtype=mp_policy.vis_dtype)
    vis_model = jnp.ones((1, Tm, B, Cm, 2, 2), dtype=mp_policy.vis_dtype)
    antenna1 = jnp.ones((B,), dtype=mp_policy.index_dtype)
    antenna2 = jnp.ones((B,), dtype=mp_policy.index_dtype)
    residuals = jax.block_until_ready(
        subtraction_step(
            gains=gains,
            vis_model=vis_model,
            vis_data=vis_data,
            antenna1=antenna1,
            antenna2=antenna2,
            num_devices=1,
            num_B_shards=1,
            num_C_shards=1
        )
    )
