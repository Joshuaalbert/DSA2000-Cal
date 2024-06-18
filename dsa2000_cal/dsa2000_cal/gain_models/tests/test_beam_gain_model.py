import numpy as np
import pylab as plt
import pytest
from astropy import units as au, coordinates as ac

from dsa2000_cal.common.coord_utils import lmn_to_icrs
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory


@pytest.mark.parametrize('array_name, freq, zenith', [['lwa', 50e6, True], ['dsa2000W', 700e6, False]])
def test_beam_gain_model_real_data(array_name, freq, zenith):
    freqs = au.Quantity([freq], unit=au.Hz)
    beam_gain_model = beam_gain_model_factory(array_name=array_name)

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
    beam_gain_model = beam_gain_model_factory(array_name=array_name)
    select = beam_gain_model.lmn_data[:, 2] >= 0.  # Select only positive N
    sc = plt.scatter(beam_gain_model.lmn_data[select, 0], beam_gain_model.lmn_data[select, 1], s=1, alpha=0.5,
                     c=np.log10(np.abs(beam_gain_model.model_gains[select, 0, 0, 0])))
    plt.colorbar(sc)
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('log10(Amplitude)')
    plt.show()

    sc = plt.scatter(beam_gain_model.lmn_data[select, 0], beam_gain_model.lmn_data[select, 1], s=1, alpha=0.5,
                     c=np.angle(beam_gain_model.model_gains[select, 0, 0, 0]),
                     cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(sc)
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('Phase')
    plt.show()
