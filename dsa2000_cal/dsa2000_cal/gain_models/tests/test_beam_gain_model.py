import time as time_mod

import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
import pytest
from astropy import units as au, coordinates as ac

from dsa2000_cal.common.coord_utils import lmn_to_icrs
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_cal.gain_models.spherical_interpolator import phi_theta_from_lmn, lmn_from_phi_theta


@pytest.mark.parametrize('array_name, freq, zenith', [['lwa', 50e6, True], ['dsa2000W', 700e6, False]])
def test_beam_gain_model_real_data(array_name, freq, zenith):
    freqs = au.Quantity([freq], unit=au.Hz)
    beam_gain_model = build_beam_gain_model(array_name=array_name)

    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=45 * au.deg)
    array_location = beam_gain_model.antennas[0]
    time = beam_gain_model.model_times[0]

    # Test meshgrid
    lvec = np.linspace(-1, 1, 100) * au.dimensionless_unscaled
    mvec = np.linspace(-1, 1, 100) * au.dimensionless_unscaled
    M, L = np.meshgrid(mvec, lvec, indexing='ij')
    lmn = np.stack([L, M, np.sqrt(1. - L ** 2 - M ** 2)], axis=-1)  # [100, 100, 3]
    sources = lmn_to_icrs(lmn=lmn, phase_tracking=phase_tracking)
    gains = beam_gain_model.compute_gain(
        freqs=freqs, sources=sources, array_location=array_location,
        time=time,
        pointing=None if zenith else phase_tracking)  # [100, 100, num_ant, num_freq, 2, 2]
    assert gains.shape == (len(mvec), len(lvec), len(beam_gain_model.antennas), len(freqs), 2, 2)
    gains = gains[..., 0, 0, 0, 0]  # [100, 100]

    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True, sharey=True, squeeze=False)
    im = axs[0, 0].imshow(
        np.abs(gains),
        extent=(lvec[0].value, lvec[-1].value, mvec[0].value, mvec[-1].value),
        origin='lower',
        cmap='PuOr',
        vmin=0.
    )
    fig.colorbar(im, ax=axs[0, 0])
    axs[0, 0].set_xlabel('l')
    axs[0, 0].set_ylabel('m')
    axs[0, 0].set_title('Amplitude')

    im = axs[1, 0].imshow(
        np.angle(gains),
        extent=(lvec[0].value, lvec[-1].value, mvec[0].value, mvec[-1].value),
        origin='lower',
        cmap='hsv',
        vmin=-np.pi,
        vmax=np.pi
    )
    fig.colorbar(im, ax=axs[1, 0])
    axs[1, 0].set_xlabel('l')
    axs[1, 0].set_ylabel('m')
    axs[1, 0].set_title('phase')

    fig.tight_layout()
    plt.show()


@pytest.mark.parametrize('array_name', ['lwa', 'dsa2000W'])
def test_beam_gain_model_factory(array_name: str):
    beam_gain_model = build_beam_gain_model(array_name=array_name)
    assert not np.any(np.isnan(beam_gain_model.model_gains))

    phi, theta = phi_theta_from_lmn(
        beam_gain_model.lmn_data[..., 0], beam_gain_model.lmn_data[..., 1], beam_gain_model.lmn_data[..., 2]
    )

    lmn_data_rec = np.stack(lmn_from_phi_theta(phi, theta), axis=-1)

    np.testing.assert_allclose(np.asarray(beam_gain_model.lmn_data), np.asarray(lmn_data_rec), atol=2e-5)

    beam_gain_model.plot_beam()

    select = (0 < beam_gain_model.lmn_data[..., 2]) & (~jnp.isnan(beam_gain_model.lmn_data[..., 2]))
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
    gain_screen.block_until_ready()
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
