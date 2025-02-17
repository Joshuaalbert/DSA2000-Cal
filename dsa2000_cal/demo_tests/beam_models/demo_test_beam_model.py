import time as time_mod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsa2000_common.common.jax_utils import block_until_ready
from dsa2000_common.gain_models.base_spherical_interpolator import lmn_from_phi_theta, phi_theta_from_lmn
from dsa2000_common.gain_models.beam_gain_model import build_beam_gain_model


# @pytest.mark.parametrize('array_name', ['lwa', 'dsa2000W', 'dsa2000_31b'])
@pytest.mark.parametrize('array_name', ['dsa2000_31b'])
def test_beam_gain_model_factory(array_name: str):
    t0 = time_mod.time()
    beam_gain_model = build_beam_gain_model(array_name=array_name)
    print(f"Built in {time_mod.time() - t0} seconds.")

    assert not np.any(np.isnan(beam_gain_model.model_gains))

    L, M = np.meshgrid(beam_gain_model.lvec, beam_gain_model.mvec, indexing='ij')
    N = np.sqrt(1. - L ** 2 - M ** 2)
    phi, theta = phi_theta_from_lmn(
        L, M, N
    )

    L_rec, M_rec, N_rec = lmn_from_phi_theta(phi, theta)

    mask = L ** 2 + M ** 2 <= 1

    np.testing.assert_allclose(L_rec[mask], L[mask], atol=2e-5)
    np.testing.assert_allclose(M_rec[mask], M[mask], atol=2e-5)
    np.testing.assert_allclose(N_rec[mask], N[mask], atol=2e-5)

    beam_gain_model.plot_regridded_beam(freq_idx=-1)

    # Only select n>=0 geodesics

    lmn_data = np.stack([L, M, N], axis=-1).reshape((-1, 3))
    geodesics = lmn_data[:, None, None, :]
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
    ))  # [num_sources, num_times, num_ant, num_freq[, 2, 2]]
    print(f"Computed in {time_mod.time() - t0} seconds.")

    reconstructed_model_gains = reconstructed_model_gains[:, 0, 0, 0, :, :].reshape(L.shape + (2, 2))

    mask = np.tile(mask[:, :, None, None], (1, 1, 2, 2))
    np.testing.assert_allclose(
        reconstructed_model_gains.real[mask],
        beam_gain_model.model_gains[0, :, :, 0, :, :].real[mask],
        atol=0.05
    )
    np.testing.assert_allclose(
        reconstructed_model_gains.imag[mask],
        beam_gain_model.model_gains[0, :, :, 0, :, :].imag[mask],
        atol=0.05
    )

# @pytest.mark.parametrize('array_name', ['lwa', 'dsa2000W', 'dsa2000_31b'])
@pytest.mark.parametrize('array_name', ['dsa2000_31b'])
def test_beam_gain_model_transforms(array_name: str):
    t0 = time_mod.time()
    beam_gain_model = build_beam_gain_model(array_name=array_name)
    print(f"Built in {time_mod.time() - t0} seconds.")
    beam_gain_model.plot_regridded_beam()

    ap_beam = beam_gain_model.to_aperture()
    ap_beam.plot_regridded_beam()

    im_beam = ap_beam.to_image()
    im_beam.plot_regridded_beam()

    np.testing.assert_allclose(im_beam.lvec, beam_gain_model.lvec, atol=1e-6)
    np.testing.assert_allclose(im_beam.mvec, beam_gain_model.mvec, atol=1e-6)
    np.testing.assert_allclose(im_beam.model_gains.real, beam_gain_model.model_gains.real, atol=1e-6)
    np.testing.assert_allclose(im_beam.model_gains.imag, beam_gain_model.model_gains.imag, atol=1e-6)
