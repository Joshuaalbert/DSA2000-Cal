import os

import pylab as plt
import pytest

from dsa2000_cal.antenna_model.antenna_model_utils import get_dish_model_beam_widths
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_assets.rfi.lte_rfi.lwa_cell_tower import LWACellTower
from dsa2000_assets.source_models.cyg_a.source_model import CygASourceModel


@pytest.mark.parametrize('array_name', ['dsa2000W_small', 'dsa2000W', 'lwa'])
def test_array_layouts(array_name: str):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))
    antennas = array.get_antennas()

    import pylab as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(antennas.itrs.cartesian.x.to('km'),
               antennas.itrs.cartesian.y.to('km'),
               antennas.itrs.cartesian.z.to('km'))
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(array_name)
    plt.show()


@pytest.mark.parametrize('array_name', ['dsa2000W_small', 'dsa2000W', 'lwa'])
def test_dsa2000_antenna_beam(array_name: str):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))
    antenna_model = array.get_antenna_model()

    # amplitude = antenna_model.get_amplitude()[..., 0, 0]
    # circular_mean = np.mean(amplitude, axis=1)
    # phi = antenna_model.get_phi()
    # theta = antenna_model.get_theta()
    # for i, freq in enumerate(antenna_model.get_freqs()):
    #     plt.plot(theta, amplitude[:, i], label=freq)
    # plt.legend()
    # plt.show()
    # for i, freq in enumerate(antenna_model.get_freqs()):
    #     for k, th in enumerate(theta):
    #         if circular_mean[k, i] < 0.01:
    #             break
    #     print(f"Freq: {i}, {freq}, theta: {k}, {th}")

    freqs, beam_widths = get_dish_model_beam_widths(
        antenna_model=antenna_model,
        threshold=0.5
    )
    plt.plot(freqs, beam_widths.to('deg').value)
    plt.ylabel('Beam Half Power Width (deg)')
    plt.xlabel('Frequency (Hz)')
    plt.show()


def test_lwa_cell_tower():
    lwa_cell_tower = LWACellTower(seed='abc')
    lwa_cell_tower.plot_acf()


def test_model():
    for file in CygASourceModel(seed='abc').get_wsclean_fits_files():
        assert os.path.isfile(file)
