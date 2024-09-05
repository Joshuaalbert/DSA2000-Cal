import time as time_mod

import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
import pytest

from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_cal.gain_models.spherical_interpolator import phi_theta_from_lmn, lmn_from_phi_theta


@pytest.mark.parametrize('array_name', ['dsa2000W_small', 'dsa2000W', 'lwa'])
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

    beam_gain_model.plot_beam()
    beam_gain_model.plot_regridded_beam()

    # Only select n>=0 geodesics
    select = beam_gain_model.lmn_data[:, 2] >= 0

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
    reconstructed_model_gains = compute_gains(
        **args
    )  # [s, t, a, f, ...]
    jax.block_until_ready(reconstructed_model_gains)
    print(f"Computed in {time_mod.time() - t0} seconds.")

    print(beam_gain_model.model_gains.shape)  # [num_model_times, num_model_dir, num_model_freqs, 2, 2]
    print(reconstructed_model_gains.shape)  # [num_sources, num_times, num_ant, num_freq[, 2, 2]]

    # Plot all on fig
    lvec, mvec = beam_gain_model.lvec_jax, beam_gain_model.mvec_jax
    gain_screen = beam_gain_model.model_gains_jax[0, :, :, 0, ...]  # [lres, mres, 2,2]
    model_gains = quantity_to_np(beam_gain_model.model_gains[0, select, 0, ...])  # [num_model_dir, 2, 2]
    reconstructed_model_gains = reconstructed_model_gains[:, 0, 0, 0, ...]  # [num_model_dir, 2, 2]
    l, m = geodesics[:, 0, 0, 0], geodesics[:, 0, 0, 1]
    reconstruct_diff = model_gains - reconstructed_model_gains
    reconstruct_diff = np.where(np.abs(reconstruct_diff) < 0.1, np.nan, reconstruct_diff)

    # Print out the l,m where we see the bad residuals
    # for p in range(2):
    #     for q in range(2):
    #         print(reconstruct_diff[~np.isnan(reconstruct_diff[:, p, q]), p, q])
    #         print(list(zip(l[~np.isnan(reconstruct_diff[:, p, q])], m[~np.isnan(reconstruct_diff[:, p, q])])))

    # Row 1 model_gains (scatter)
    # Row 2 reconstructed_model_gains (scatter)
    # Row 3 model_gains - reconstructed_model_gains (scatter)
    # Row 4 gain_screen (imshow)
    # Col 1 Abs
    # Col 2 Phase

    for p, q in [(0, 0), (0, 1)]:
        fig, axs = plt.subplots(4, 2, figsize=(6, 12), sharex=True, sharey=True)

        for i, (data, title) in enumerate(zip(
                [model_gains[:, p, q],
                 reconstructed_model_gains[:, p, q],
                 reconstruct_diff[:, p, q],
                 gain_screen[:, :, p, q]],
                [f'Model Gains({p},{q})', f'Reconstructed Model Gains({p},{q})', f'Bad Residuals({p},{q})',
                 f'Gain Screen({p},{q})']
        )):
            for j, (quantity, ylabel) in enumerate(zip([np.abs(data), np.angle(data)], ['Abs', 'Phase'])):
                if i == 3:
                    im = axs[i, j].imshow(quantity.T,
                                          origin='lower',
                                          extent=[lvec[0], lvec[-1], mvec[0], mvec[-1]],
                                          cmap='jet',
                                          interpolation='none'
                                          )
                    fig.colorbar(im, ax=axs[i, j])
                else:
                    sc = axs[i, j].scatter(l, m, c=quantity, s=1, cmap='jet', alpha=0.5)
                    fig.colorbar(sc, ax=axs[i, j])
                axs[i, j].set_title(f'{title} {ylabel}')
                axs[i, j].set_xlabel('l (proj.rad)')
                axs[i, j].set_ylabel('m (proj.rad)')

        plt.show()

    np.testing.assert_allclose(
        np.abs(reconstructed_model_gains - model_gains),
        np.zeros(model_gains.shape),
        atol=0.05
    )
    np.testing.assert_allclose(
        np.abs(reconstructed_model_gains),
        np.abs(model_gains),
        atol=0.05
    )

    # np.testing.assert_allclose(
    #     np.angle(reconstructed_model_gains),
    #     np.angle(model_gains),
    #     atol=0.05
    # )
