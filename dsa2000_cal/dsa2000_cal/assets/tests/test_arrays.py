import os

import pylab as plt
import pytest
from astropy import units as au

from dsa2000_cal.antenna_model.utils import get_dish_model_beam_widths
from dsa2000_cal.assets.arrays.array import extract_itrs_coords
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry


def test_extract_itrs_coords():
    with open('test_array.txt', 'w') as f:
        f.write('#X Y Z dish_diam station mount\n'
                '-1601614.0612 -5042001.67655 3554652.4556 25 vla-00 ALT-AZ\n'
                '-1602592.82353 -5042055.01342 3554140.65277 25 vla-01 ALT-AZ\n'
                '-1604008.70191 -5042135.83581 3553403.66677 25 vla-02 ALT-AZ\n'
                '-1605808.59818 -5042230.07046 3552459.16736 25 vla-03 ALT-AZ')
    stations, antenna_coords = extract_itrs_coords('test_array.txt')
    assert len(stations) == 4
    assert len(antenna_coords) == 4
    assert antenna_coords[0].x == -1601614.0612 * au.m
    assert antenna_coords[0].y == -5042001.67655 * au.m
    assert antenna_coords[0].z == 3554652.4556 * au.m
    assert stations == ['vla-00', 'vla-01', 'vla-02', 'vla-03']
    os.remove('test_array.txt')


@pytest.mark.parametrize('array_name', ['dsa2000W_small', 'dsa2000W'])
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
