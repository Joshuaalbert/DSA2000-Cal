import os.path
import sys
from typing import Union

import astropy.coordinates as ac
import astropy.units as au
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.assets.rfi.rfi_data import RFIData


def calculate_free_space_path_loss(rfi_sim_config: RFISimConfig, ms_data: MSData) -> np.ndarray:
    """
    Computes free space path loss for each telescope antenna given the RFI transmitter location.

    Args:
        rfi_sim_config: Configuration for the RFI simulation.
        ms_data: Data from an MS file.

    Returns:
        Free space path loss for each telescope antenna [num_ant]
    """
    lte_location = ac.SkyCoord(
        east=rfi_sim_config.lte_east * au.m,
        north=rfi_sim_config.lte_north * au.m,
        up=rfi_sim_config.lte_up * au.m,
        frame=ms_data.enu_frame
    )
    antennas_enu = ms_data.antennas.transform_to(ms_data.enu_frame)
    antennas_enu_xyz = antennas_enu.cartesian.xyz.T  # [num_ant, 3]
    lte_location_xyz = lte_location.cartesian.xyz  # [3]
    dist = np.linalg.norm((antennas_enu_xyz - lte_location_xyz).to(au.m).value, axis=-1)  # [num_ant]
    free_space_path_loss = C / (4 * np.pi * dist * rfi_sim_config.lte_frequency_hz) ** 2  # [num_ant]
    return free_space_path_loss


def calculate_side_lobes_attenuation(rfi_sim_config: RFISimConfig, ms_data: MSData) -> np.ndarray:
    """
    Computes side lobes attenuation for each telescope antenna given the RFI transmitter location.

    Args:
        rfi_sim_config: Configuration for the RFI simulation.
        ms_data: Data from an MS file.

    Returns:
        Side lobes attenuation for each telescope antenna [num_ant]
    """
    antenna_model = ms_data.array.get_antenna_model()

    lte_location = ac.SkyCoord(
        east=rfi_sim_config.lte_east * au.m,
        north=rfi_sim_config.lte_north * au.m,
        up=rfi_sim_config.lte_up * au.m,
        frame=ms_data.enu_frame
    )

    antennas_enu_xyz = ms_data.antennas.transform_to(ms_data.enu_frame).cartesian.xyz.T  # [num_ant, 3]
    lte_location_xyz = lte_location.cartesian.xyz  # [3]

    line_of_sight = antennas_enu_xyz - lte_location_xyz  # [num_ant, 3]
    line_of_sight /= np.linalg.norm(line_of_sight, axis=-1, keepdims=True)  # [num_ant, 3] (normed)
    source = ac.SkyCoord(east=line_of_sight[:, 0], north=line_of_sight[:, 1], up=line_of_sight[:, 2],
                         frame=ms_data.enu_frame).transform_to(ac.ICRS())  # [num_ant]

    return antenna_model.compute_amplitude(
        pointing=ms_data.pointing,
        source=source,
        freq_hz=rfi_sim_config.lte_frequency_hz,
        enu_frame=ms_data.enu_frame,
        pol='X'
    )  # [num_ant]


def calculate_geometric_delays(rfi_sim_config: RFISimConfig, ms_data: MSData) -> np.ndarray:
    """
    Computes geometric delays for each visibility given the RFI transmitter location.

    Args:
        rfi_sim_config: Configuration for the RFI simulation.
        ms_data: Data from an MS file.

    Returns:
        Geometric delays for each visibility [num_vis]
    """
    lte_location = ac.SkyCoord(
        east=rfi_sim_config.lte_east * au.m,
        north=rfi_sim_config.lte_north * au.m,
        up=rfi_sim_config.lte_up * au.m,
        frame=ms_data.enu_frame
    )
    lte_location_xyz = lte_location.cartesian.xyz  # [3]
    antennas_enu_xyz = ms_data.antennas.transform_to(ms_data.enu_frame).cartesian.xyz.T  # [num_ant, 3]
    dist_m = np.linalg.norm((antennas_enu_xyz - lte_location_xyz).to(au.m).value, axis=-1)  # [num_ant] (m)
    delays = (dist_m[ms_data.antenna1] - dist_m[ms_data.antenna2]) / C  # [num_vis]
    return delays


def calculate_tracking_delays(rfi_sim_config: RFISimConfig, ms_data: MSData) -> np.ndarray:
    """
    Computes tracking delays for each visibility given the RFI transmitter location.

    Args:
        ms_data: Data from an MS file.

    Returns:
        Tracking delays for each visibility [num_vis]
    """
    antennas_enu_xyz = ms_data.antennas.transform_to(ms_data.enu_frame).cartesian.xyz.T  # [num_ant, 3]
    pointing_enu_xyz = ms_data.pointing.transform_to(ms_data.enu_frame).cartesian.xyz.T  # [num_ant, 3] (normed)

    antenna_tracking_delays = np.sum((pointing_enu_xyz * antennas_enu_xyz).to(au.m).value, axis=-1) / C  # [num_ant]
    track_delay = antenna_tracking_delays[ms_data.antenna1] - antenna_tracking_delays[ms_data.antenna2]  # [num_vis]
    return track_delay


def calculate_visibilities(free_space_path_loss, side_lobes_attenuation, geometric_delays, tracking_delays,
                           time_acf, acf,
                           rfi_sim_config: RFISimConfig, ms_data: MSData) -> np.ndarray:
    """
    Computes visibilities for each visibility given the RFI transmitter location.

    Args:
        free_space_path_loss: The free space path loss for each telescope antenna [num_ant]
        side_lobes_attenuation: The side lobes attenuation for each telescope antenna [num_ant]
        geometric_delays: The geometric delays for each visibility [num_vis]
        tracking_delays: The tracking delays for each visibility [num_vis]
        time_acf: The time axis of the ACF [num_acf]
        acf: The ACF [num_acf]
        rfi_sim_config: Configuration for the RFI simulation.
        ms_data: Data from an MS file.

    Returns:
        Visibilities for each visibility [num_vis, num_channels, 4]
    """
    (num_vis,) = geometric_delays.shape
    vis = np.zeros((num_vis, 1, 4), dtype=np.complex64)
    tot_att = (
                      1e26 * rfi_sim_config.lte_power_W_Hz / ((C / rfi_sim_config.lte_frequency_hz) ** 2 / (4 * np.pi))
              ) * np.sqrt(
        free_space_path_loss[ms_data.antenna1] * free_space_path_loss[ms_data.antenna2]
    ) * np.sqrt(
        side_lobes_attenuation[ms_data.antenna1] * side_lobes_attenuation[ms_data.antenna2]
    )  # [num_vis]
    total_delay = geometric_delays + tracking_delays  # [num_vis]
    # Get the ACF for times that are within the delay range (performance improvement?)
    min_delay = np.min(total_delay)
    max_delay = np.max(total_delay)
    # (delidx,) = np.where((time_acf >= min_delay) & (time_acf <= max_delay))
    # time_acf = time_acf[delidx]
    # acf = acf[delidx]
    print(f"Delay range: {min_delay:.2e} - {max_delay:.2e}")
    print(f"Total attenuation range: {np.min(tot_att):.2e} - {np.max(tot_att):.2e}")
    # Calculate visibilities, find the delay index for each visibility, and multiply by the ACF at that index
    select_idx = np.clip(np.searchsorted(time_acf, total_delay), 0, len(time_acf) - 1)
    print("Select idx range:", np.min(select_idx), np.max(select_idx))

    # We put into the destined channel
    (select_freq_idx,) = np.where(np.bitwise_and(
        ms_data.freqs_hz >= rfi_sim_config.lte_frequency_hz - rfi_sim_config.lte_bandwidth_hz / 2,
        ms_data.freqs_hz <= rfi_sim_config.lte_frequency_hz + rfi_sim_config.lte_bandwidth_hz / 2
    ))
    vis[:, select_freq_idx, :] = np.reshape(tot_att * acf[select_idx], (-1, 1, 1))  # [num_vis, num_channels, 4]
    # Now we need to rotate the correlations to the correct polarization angle
    lte_polarization_rad = np.deg2rad(rfi_sim_config.lte_polarization_deg)
    vis[:, :, 0] *= np.cos(lte_polarization_rad) ** 2
    vis[:, :, 1] *= np.cos(lte_polarization_rad) * np.sin(lte_polarization_rad)
    vis[:, :, 2] *= np.cos(lte_polarization_rad) * np.sin(lte_polarization_rad)
    vis[:, :, 3] *= np.sin(lte_polarization_rad) ** 2
    return vis


def run_rfi_simulation(array_name: str,
                       ms_file: str,
                       rfi_sim_config: RFISimConfig,
                       overwrite: bool = True):
    """
    Inject an RFI signal (from LTE cell towers) within an MS data set.

    Args:
        array_name: Name of the array.
        ms_file: MS file where the RFI will be injected.
        rfi_sim_config: Configuration for the RFI simulation.
        overwrite: Option to overwrite visibilities in input MS file instead of adding them. Default is True.
    """
    if not os.path.exists(ms_file):
        raise ValueError(f"MS file {ms_file} does not exist.")
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))

    logger.info("Extracting data from MS file.")
    gen = iter_ms_data(ms_file=ms_file, array=array, overwrite=overwrite)
    gen_response: Union[np.ndarray, None] = None
    pbar = tqdm(file=sys.stdout, dynamic_ncols=True)

    pbar.set_description("Loading RFI data.")
    rfi_acf_data = loadmat(RFIData().rfi_injection_model())
    time_acf = rfi_acf_data['t_acf'][0]
    acf = rfi_acf_data['acf'][0]

    while True:
        try:
            ms_data = gen.send(gen_response)
        except StopIteration:
            break

        pbar.set_description("Calculating free space path loss.")
        free_space_path_loss = calculate_free_space_path_loss(rfi_sim_config=rfi_sim_config, ms_data=ms_data)
        if np.count_nonzero(free_space_path_loss) == 0:
            raise ValueError("Free space path loss is zero, this indicates an problem.")

        pbar.set_description("Calculating side lobes attenuation.")
        side_lobes_attenuation = calculate_side_lobes_attenuation(rfi_sim_config=rfi_sim_config, ms_data=ms_data)
        if np.count_nonzero(side_lobes_attenuation) == 0:
            raise ValueError("Side lobes attenuation is zero, this indicates an problem.")

        pbar.set_description("Calculating geometric delays.")
        geometric_delays = calculate_geometric_delays(rfi_sim_config=rfi_sim_config, ms_data=ms_data)

        pbar.set_description("Calculating tracking delays.")
        tracking_delays = calculate_tracking_delays(rfi_sim_config=rfi_sim_config, ms_data=ms_data)

        pbar.set_description("Calculating visibilities.")
        visibilities = calculate_visibilities(
            free_space_path_loss=free_space_path_loss,
            side_lobes_attenuation=side_lobes_attenuation,
            geometric_delays=geometric_delays,
            tracking_delays=tracking_delays,
            time_acf=time_acf,
            acf=acf,
            rfi_sim_config=rfi_sim_config,
            ms_data=ms_data
        )  # [num_vis, 1, 4]
        if np.any(np.isnan(visibilities)):
            raise ValueError("NaNs in visibilities.")
        logger.info(f"Injected {visibilities.shape[0]} visibilities. Mean {np.mean(visibilities):.2e}.")
        gen_response = visibilities
        pbar.update(1)
