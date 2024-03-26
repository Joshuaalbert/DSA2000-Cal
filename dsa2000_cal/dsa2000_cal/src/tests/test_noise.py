import numpy as np

from dsa2000_cal.noise import calc_baseline_noise, calc_image_noise


def test_calc_noise_full_observation():
    num_antennas = 2000
    system_equivalent_flux_density = 5022. / 2000.  # Jy
    chan_width_hz = 1300e6
    t_int_s = 900
    system_efficiency = 0.7
    assert np.isclose(
        num_antennas * calc_baseline_noise(system_equivalent_flux_density, chan_width_hz, t_int_s), 0.00407,
        atol=1e-3)

def test_calc_noise_8000chan_1hour():
    num_antennas = 2048
    system_equivalent_flux_density = 5022.  # Jy
    chan_width_hz = 162500.0  # Hz
    t_int_s = 1.5  # s
    num_channels = 8000
    num_integrations = 3600 // 15
    flag_frac = 0.35
    image_noise = calc_image_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        bandwidth_hz=chan_width_hz * num_channels,
        t_int_s=t_int_s * num_integrations,
        num_antennas=num_antennas,
        flag_frac=flag_frac
    )
    print(f"Image noise (1h 35% flagged): {image_noise} Jy / beam")
    assert np.isclose(image_noise, 4.447e-6, atol=1e-1)

    baseline_noise = calc_baseline_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        chan_width_hz=chan_width_hz,
        t_int_s=t_int_s
    )
    print(f"Baseline noise: {baseline_noise} Jy")
    assert np.isclose(baseline_noise, 7.19, atol=1e-1)
