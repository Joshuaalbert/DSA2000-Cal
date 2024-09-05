import time as time_mod

import jax
import jax.numpy as jnp
import pytest

from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model


@pytest.mark.parametrize('array_name', ['dsa2000W_small', 'dsa2000W', 'lwa'])
def test_compute_beam_at_data_points(array_name: str):
    t0 = time_mod.time()
    beam_gain_model = build_beam_gain_model(array_name=array_name)
    print(f"Built in {time_mod.time() - t0} seconds.")

    geodesics = beam_gain_model.lmn_data[:, None, None, :]
    args = dict(
        freqs=quantity_to_jnp(beam_gain_model.model_freqs[0:1]),
        times=jnp.asarray([0.]),
        geodesics=geodesics
    )
    t0 = time_mod.time()
    compute_gains = jax.jit(beam_gain_model.compute_gain).lower(**args).compile()
    print(f"Compiled in {time_mod.time() - t0} seconds.")

    t0 = time_mod.time()
    gain_screen = compute_gains(
        **args
    )  # [s, t, a, f, ...]
    jax.block_until_ready(gain_screen)
    print(f"Computed in {time_mod.time() - t0} seconds.")
