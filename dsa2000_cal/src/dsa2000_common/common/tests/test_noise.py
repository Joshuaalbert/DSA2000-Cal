import numpy as np
import pytest

from dsa2000_common.common.noise import calc_baseline_noise, calc_image_noise


def test_calc_noise_10000chan_1hour():
    num_antennas = 2048
    system_equivalent_flux_density = 5022.  # Jy
    chan_width_hz = 1300e6 / 10000  # Hz
    t_int_s = 1.5  # s
    num_channels = 10000
    num_integrations = 3600. / t_int_s
    flag_frac = 0.35
    image_noise = calc_image_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        bandwidth_hz=chan_width_hz * num_channels,
        t_int_s=t_int_s * num_integrations,
        num_antennas=num_antennas,
        flag_frac=flag_frac,
        num_pol=2
    )
    print(f"Image noise (1h 35% flagged): {image_noise} Jy / beam")
    assert np.isclose(image_noise, 1e-6, atol=1e-6)

    baseline_noise = calc_baseline_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        chan_width_hz=chan_width_hz,
        t_int_s=t_int_s
    )
    print(f"Baseline noise: {baseline_noise} Jy")
    assert np.isclose(baseline_noise, 11.37, atol=1e-1)


@pytest.mark.parametrize('num_chan', [1, 1920])
@pytest.mark.parametrize('frac_flagged', [0., 0.33])
def test_calc_noise_1920chan_10s_lwa(num_chan: int, frac_flagged: float):
    # 352 5570.0 Jy
    num_antennas = 352
    system_equivalent_flux_density = 1707324.  # full band snapshow has 160mJy noise
    chan_width_hz = 23913.3199056  # Hz
    t_int_s = 10.  # s
    num_channels = num_chan
    num_integrations = 1
    flag_frac = frac_flagged
    image_noise = calc_image_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        bandwidth_hz=chan_width_hz * num_channels,
        t_int_s=t_int_s * num_integrations,
        num_antennas=num_antennas,
        flag_frac=flag_frac,
        num_pol=2
    )
    print(f"Image noise (10s {num_chan} chans {frac_flagged}% flagged): {image_noise} Jy / beam")
    # assert np.isclose(image_noise, 4.447e-6, atol=1e-1)

    baseline_noise = calc_baseline_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        chan_width_hz=chan_width_hz,
        t_int_s=t_int_s
    )
    print(f"Baseline noise: {baseline_noise} Jy")
    # assert np.isclose(baseline_noise, 7.19, atol=1e-1)


def test_calc_noise_full_observation():
    num_antennas = 2048
    system_equivalent_flux_density = 5022. / 2000.  # Jy
    chan_width_hz = 1300e6  # Hz
    t_int_s = 10.3 * 60  # s
    system_efficiency = 0.7
    assert np.isclose(
        num_antennas * calc_baseline_noise(system_equivalent_flux_density, chan_width_hz, t_int_s), 0.00407,
        atol=1e-3)


def test_calc_noise_8000chan_1hour():
    num_antennas = 2048
    system_equivalent_flux_density = 5022.  # Jy
    chan_width_hz = 130000.0  # Hz
    t_int_s = 1.5  # s
    num_channels = 10000
    num_integrations = 3600 / t_int_s
    flag_frac = 0.33
    image_noise = calc_image_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        bandwidth_hz=chan_width_hz * num_channels,
        t_int_s=t_int_s * num_integrations,
        num_antennas=num_antennas,
        flag_frac=flag_frac,
        num_pol=2
    )
    print(chan_width_hz * num_channels)
    print(f"Image noise (1h 33% flagged): {image_noise} Jy / beam")
    assert np.isclose(image_noise, 1e-6, atol=1e-6)

    baseline_noise = calc_baseline_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        chan_width_hz=chan_width_hz,
        t_int_s=t_int_s
    )
    print(f"Baseline noise: {baseline_noise} Jy")
    assert np.isclose(baseline_noise, 11.37, atol=1e-1)


def test_calc_noise_fullband_obs():
    num_antennas = 2048
    system_equivalent_flux_density = 5022.  # Jy
    chan_width_hz = 130000.0  # Hz
    t_int_s = 1.5  # s
    num_channels = 10000
    num_integrations = 10.3 * 60 / t_int_s
    flag_frac = 0.33
    image_noise = calc_image_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        bandwidth_hz=chan_width_hz * num_channels,
        t_int_s=t_int_s * num_integrations,
        num_antennas=num_antennas,
        flag_frac=flag_frac,
        num_pol=2
    )
    print(chan_width_hz * num_channels)
    print(f"Image noise (10.3min 33% flagged): {image_noise} Jy / beam")

    baseline_noise = calc_baseline_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        chan_width_hz=chan_width_hz,
        t_int_s=t_int_s
    )
    print(f"Baseline noise: {baseline_noise} Jy")


def test_calc_noise_40chan_6s_dsa():
    num_antennas = 2048
    system_equivalent_flux_density = 5022.  # Jy
    chan_width_hz = 130000.0  # Hz
    t_int_s = 1.5  # s
    num_channels = 40
    num_integrations = 4
    flag_frac = 0.35
    image_noise = calc_image_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        bandwidth_hz=chan_width_hz * num_channels,
        t_int_s=t_int_s * num_integrations,
        num_antennas=num_antennas,
        flag_frac=flag_frac,
        num_pol=2
    )
    print(f"Image noise (6s 40chans 35% flagged): {image_noise} Jy / beam")
    # assert np.isclose(image_noise, 4.447e-6, atol=1e-1)

    baseline_noise = calc_baseline_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        chan_width_hz=chan_width_hz,
        t_int_s=t_int_s
    )
    print(f"Baseline noise: {baseline_noise} Jy")
    # assert np.isclose(baseline_noise, 7.19, atol=1e-1)


@pytest.mark.parametrize('num_chan', [1, 1920])
@pytest.mark.parametrize('frac_flagged', [0., 0.33])
def test_calc_noise_1920chan_10s_lwa(num_chan: int, frac_flagged: float):
    # 352 5570.0 Jy
    num_antennas = 352
    system_equivalent_flux_density = 1707324.  # full band snapshow has 160mJy noise
    chan_width_hz = 23913.3199056  # Hz
    t_int_s = 10.  # s
    num_channels = num_chan
    num_integrations = 1
    flag_frac = frac_flagged
    image_noise = calc_image_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        bandwidth_hz=chan_width_hz * num_channels,
        t_int_s=t_int_s * num_integrations,
        num_antennas=num_antennas,
        flag_frac=flag_frac,
        num_pol=2
    )
    print(f"Image noise (10s {num_chan} chans {frac_flagged}% flagged): {image_noise} Jy / beam")
    # assert np.isclose(image_noise, 4.447e-6, atol=1e-1)

    baseline_noise = calc_baseline_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        chan_width_hz=chan_width_hz,
        t_int_s=t_int_s
    )
    print(f"Baseline noise: {baseline_noise} Jy")
    # assert np.isclose(baseline_noise, 7.19, atol=1e-1)
