import time as time_mod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.gain_models.base_spherical_interpolator import lmn_from_phi_theta, phi_theta_from_lmn
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model, select_interpolation_points


def test_pytree_serialisation():
    beam_gain_model = build_beam_gain_model(array_name='lwa_mock')

    @jax.jit
    def f(model):
        return model

    beam_gain_model = block_until_ready(f(beam_gain_model))


@pytest.mark.parametrize('array_name', ['lwa_mock', 'dsa2000W_small'])
def test_beam_gain_model_factory(array_name: str):
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


def test_select_interpolation_points():
    desired_freqs = np.asarray([1.0])
    model_freqs = np.asarray([1.0, 2.0, 3.0])
    expected_select_idxs = np.asarray([0, 1])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)

    desired_freqs = np.asarray([1.0])
    model_freqs = np.asarray([1.0, 1.0, 2.0, 3.0])
    expected_select_idxs = np.asarray([1, 2])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)

    desired_freqs = np.asarray([3.0])
    model_freqs = np.asarray([1.0, 2.0, 3.0])
    expected_select_idxs = np.asarray([2])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)

    desired_freqs = np.asarray([3.0])
    model_freqs = np.asarray([1.0, 2.0, 3.0, 3.0])
    expected_select_idxs = np.asarray([3])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)

    desired_freqs = np.asarray([1.0, 2.0, 3.0])
    model_freqs = np.asarray([0.5, 1.5, 2.5, 3.5])
    expected_select_idxs = np.asarray([0, 1, 2, 3])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)

    desired_freqs = np.asarray([1.0, 2.0, 3.0])
    model_freqs = np.asarray([0.5, 1.5, 1.5, 1.5, 1.75, 2.5, 3.5])
    expected_select_idxs = np.asarray([0, 1, 4, 5, 6])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)
