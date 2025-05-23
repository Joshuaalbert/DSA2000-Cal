import time as time_mod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsa2000_common.common.jax_utils import block_until_ready
from dsa2000_common.gain_models.base_spherical_interpolator import lmn_from_phi_theta, phi_theta_from_lmn
from dsa2000_common.gain_models.beam_gain_model import build_beam_gain_model


def test_pytree_serialisation():
    beam_gain_model = build_beam_gain_model(array_name='lwa_mock')

    @jax.jit
    def f(model):
        return model

    beam_gain_model = block_until_ready(f(beam_gain_model))


# @pytest.mark.parametrize('array_name', ['lwa_mock', 'dsa2000W_small'])
@pytest.mark.parametrize('array_name', ['dsa2000_optimal_v1'])
def test_beam_gain_model(array_name: str):
    t0 = time_mod.time()
    beam_gain_model = build_beam_gain_model(array_name=array_name)
    print(f"Built in {time_mod.time() - t0} seconds.")

    assert not np.any(np.isnan(beam_gain_model.model_gains))

    L, M = np.meshgrid(beam_gain_model.lvec, beam_gain_model.mvec, indexing='ij')
    N = np.sqrt(1. - L ** 2 - M ** 2)
    phi, theta = phi_theta_from_lmn(
        L, M, N
    )  # [num_l, num_m, 2]

    L_rec, M_rec, N_rec = lmn_from_phi_theta(phi, theta)

    mask = L ** 2 + M ** 2 <= 1  # [num_l, num_m]

    np.testing.assert_allclose(L_rec[mask], L[mask], atol=2e-5)
    np.testing.assert_allclose(M_rec[mask], M[mask], atol=2e-5)
    np.testing.assert_allclose(N_rec[mask], N[mask], atol=2e-5)

    beam_gain_model.plot_regridded_beam()

    # Only select n>=0 geodesics

    lmn_data = np.stack([L, M, N], axis=-1).reshape((-1, 3))
    geodesics = lmn_data[None, None, :, :]  # [1, 1, num_l * num_m, 3]
    args = dict(
        freqs=beam_gain_model.model_freqs[0:1],
        times=jnp.asarray([0.]),
        lmn_geodesic=geodesics
    )
    t0 = time_mod.time()
    compute_gains = jax.jit(beam_gain_model.compute_gain).lower(**args).compile()
    print(f"Compiled in {time_mod.time() - t0} seconds.")

    t0 = time_mod.time()
    reconstructed_model_gains = block_until_ready(compute_gains(
        **args
    ))  # [num_times, num_ant, num_freq, num_sources,  [, 2, 2]]
    print(f"Computed in {time_mod.time() - t0} seconds.")

    reconstructed_model_gains = reconstructed_model_gains[0, 0, 0, :, :, :].reshape(
        L.shape + (2, 2))  # [num_l, num_m, 2, 2]

    mask = np.tile(mask[:, :, None, None], (1, 1, 2, 2))
    np.testing.assert_allclose(
        reconstructed_model_gains.real[mask],
        # [num_model_times, lres, mres, [num_ant,] num_model_freqs[, 2, 2]] -> [num_l, num_m, 2, 2]
        beam_gain_model.model_gains[0, :, :, 0, :, :].real[mask],
        atol=0.05
    )
    np.testing.assert_allclose(
        reconstructed_model_gains.imag[mask],
        beam_gain_model.model_gains[0, :, :, 0, :, :].imag[mask],
        atol=0.05
    )

    aperture_beam = beam_gain_model.to_aperture()
    aperture_beam.plot_regridded_beam(is_aperture=True)
