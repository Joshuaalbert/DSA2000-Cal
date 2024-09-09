import itertools
import time

import jax
from jax import numpy as jnp

from dsa2000_cal.common.types import complex_type
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.visibility_model.rime_model import RIMEModel


def test_apply_gains_benchmark_performance():
    num_source = 10
    num_chan = 2
    num_ant = 2048
    num_time = 1

    antennas = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (num_ant, 3))
    antenna_1, antenna_2 = jnp.asarray(
        list(itertools.combinations_with_replacement(range(num_ant), 2))).T

    num_rows = len(antenna_1)

    uvw = antennas[antenna_2] - antennas[antenna_1]
    uvw = uvw.at[:, 2].mul(1e-3)

    times = jnp.arange(num_time) * 1.5
    time_idx = jnp.zeros((num_rows,), jnp.int64)
    time_obs = times[time_idx]

    visibility_coords = VisibilityCoords(
        uvw=uvw,
        time_obs=time_obs,
        antenna_1=antenna_1,
        antenna_2=antenna_2,
        time_idx=time_idx
    )

    vis = jnp.zeros((num_source, num_rows, num_chan, 2, 2), dtype=complex_type)
    gains = jnp.zeros((num_source, num_time, num_ant, num_chan, 2, 2), dtype=complex_type)

    f = jax.jit(RIMEModel.apply_gains).lower(gains=gains, vis=vis, visibility_coords=visibility_coords).compile()
    t0 = time.time()
    f(gains=gains, vis=vis, visibility_coords=visibility_coords).block_until_ready()
    t1 = time.time()
    print(f"Apply gains for sources {num_source} ant {num_ant} freqs {num_chan} took {t1 - t0:.2f} s")
    # Apply gains for sources 10 ant 2048 freqs 16 took 7.98 seconds 1.10 s | 1.00 s | 1.28 s | 1.45 s | 1.00 s
