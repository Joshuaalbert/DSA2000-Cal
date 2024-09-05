import time as time_mod

import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
import pytest

from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_cal.gain_models.spherical_interpolator import phi_theta_from_lmn, lmn_from_phi_theta


@pytest.mark.parametrize('array_name', ['dsa2000W_small'])
def test_beam_gain_model_factory(array_name: str):
    t0 = time_mod.time()
    beam_gain_model = build_beam_gain_model(array_name=array_name)
    print(f"Built in {time_mod.time() - t0} seconds.")

    assert not np.any(np.isnan(beam_gain_model.model_gains))

    phi, theta = phi_theta_from_lmn(
        beam_gain_model.lmn_data[..., 0], beam_gain_model.lmn_data[..., 1], beam_gain_model.lmn_data[..., 2]
    )

    lmn_data_rec = np.stack(lmn_from_phi_theta(phi, theta), axis=-1)

    np.testing.assert_allclose(np.asarray(beam_gain_model.lmn_data), np.asarray(lmn_data_rec), atol=2e-5)

    beam_gain_model.plot_regridded_beam()

    select = jnp.logical_not(jnp.isnan(beam_gain_model.lmn_data[..., 2]))
    geodesics = beam_gain_model.lmn_data[select, None, None, :]
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

    plt.scatter(geodesics[:, 0, 0, 0], geodesics[:, 0, 0, 1], c=np.log10(np.abs(gain_screen[:, 0, 0, 0, 0, 0])),
                cmap='PuOr', s=1)
    plt.colorbar()
    plt.show()
    plt.scatter(geodesics[:, 0, 0, 0], geodesics[:, 0, 0, 1], c=np.angle(gain_screen[:, 0, 0, 0, 0, 0]), cmap='hsv',
                vmin=-np.pi, vmax=np.pi, s=1)
    plt.colorbar()
    plt.show()

    np.testing.assert_allclose(
        np.abs(gain_screen[:, 0, 0, 0, ...]),
        np.abs(quantity_to_jnp(beam_gain_model.model_gains[0, select, 0, ...])),
        atol=0.02
    )

    np.testing.assert_allclose(
        np.angle(gain_screen[:, 0, 0, 0, ...]),
        np.angle(quantity_to_jnp(beam_gain_model.model_gains[0, select, 0, ...])),
        atol=0.02
    )
