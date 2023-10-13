import numpy as np

from dsa2000_cal.noise import calc_noise


def test_calc_noise_full_observation():
    num_antennas = 2000
    system_equivalent_flux_density = 5022. / 2000.  # Jy
    chan_width_hz = 1300e6
    t_int_s = 900
    system_efficiency = 0.7
    assert np.isclose(
        num_antennas * calc_noise(system_equivalent_flux_density, chan_width_hz, t_int_s, system_efficiency), 0.00407,
        atol=1e-3)
