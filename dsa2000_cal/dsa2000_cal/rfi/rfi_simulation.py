import logging
import os.path

import astropy.time as at
import numpy as np
import utm
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from casacore.tables import table
from pydantic import Field
from scipy.io import loadmat
from tqdm import tqdm

from dsa2000_cal.assets.rfi.rfi_data import RFIData
from dsa2000_cal.utils import SerialisableBaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Constants
C = 299792458.  # Speed of light [m/s]


def extract_ms_data(msfile):
    """Extracts relevant data from an MS file."""
    with table(msfile) as t:
        uvw = t.getcol('UVW')
        ant1 = t.getcol('ANTENNA1')
        ant2 = t.getcol('ANTENNA2')
    visidx = np.zeros((len(ant1), 2), dtype=int)
    visidx[:, 0] = ant1
    visidx[:, 1] = ant2
    with table(msfile + '/ANTENNA') as t:
        antspos = t.getcol('POSITION')

    antspos = EarthLocation.from_geocentric(x=antspos[:, 0], y=antspos[:, 1], z=antspos[:, 2]).to_geodetic()
    arr_center = [np.mean(antspos[1]), np.mean(antspos[0])]
    antspos = utm.from_latlon(np.array(antspos.lat), np.array(antspos.lon))
    antspos = np.array([antspos[0] - np.mean(antspos[0]), antspos[1] - np.mean(antspos[1])])
    with table(msfile + '/POINTING') as t:
        time = t.getcol('TIME')[0]
        time = at.Time(time / 86400, format='mjd')
        point = t.getcol('DIRECTION')[0][0]

    observing_location = EarthLocation(lat=arr_center[0], lon=arr_center[1], height=5)
    observing_time = time
    aa = AltAz(location=observing_location, obstime=observing_time)
    coord = SkyCoord(point[0], point[1], unit="deg")
    azel = coord.transform_to(aa)
    azel = [azel.az, azel.alt]
    return uvw, antspos.T, azel, visidx


def calculate_free_space_path_loss(LTEang, LTEdist, LTEheight, LTEfreq, antspos):
    """Calculates free space path loss between RFI transmitter and individual antennas."""
    FSPL = np.zeros(len(antspos))
    LTE = [LTEdist * np.cos(LTEang * 2 * np.pi / 360 + np.pi / 2),
           LTEdist * np.sin(LTEang * 2 * np.pi / 360 + np.pi / 2),
           LTEheight]
    for k in range(len(antspos)):
        d = np.sqrt((antspos[k, 0] - LTE[0]) ** 2 + (antspos[k, 1] - LTE[1]) ** 2 + (5 - LTE[2]) ** 2)
        FSPL[k] = (C / (4 * np.pi * d * LTEfreq * 1e6)) ** 2
    return FSPL


def calculate_side_lobes_attenuation(LTEang, LTEdist, LTEheight, LTEfreq, antspos, azel):
    """Computes side lobe attenuation for each telescope antenna given the RFI transmitter location."""
    ant_model = loadmat(RFIData().dsa2000_antenna_model())
    I = np.argmin(np.abs(ant_model["freqListGHz"] - LTEfreq * 1e-3))
    SLA = np.zeros(len(antspos))
    az = np.deg2rad(azel[0].value + 90)
    el = np.deg2rad(azel[1].value)
    ArrSteer = np.array([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)])
    LTE = [LTEdist * np.cos(LTEang * 2 * np.pi / 360 + np.pi / 2),
           LTEdist * np.sin(LTEang * 2 * np.pi / 360 + np.pi / 2),
           LTEheight]
    normArrSteer = np.sqrt(np.dot(ArrSteer, ArrSteer))
    for k in range(len(antspos)):
        vec = [(antspos[k, 0] - LTE[0]), (antspos[k, 1] - LTE[1]), 5 - LTE[2]]
        ang = np.dot(ArrSteer, vec) / np.sqrt(np.dot(vec, vec)) / normArrSteer
        ang = np.arccos(ang)
        idx = np.argmin(np.abs(ant_model["ThetaDeg"] - np.rad2deg(ang)))
        SLA[k] = 10 ** (ant_model["coPolPattern_dBi_Freqs_15DegConicalShield"][idx, 0, I] / 10)
    return SLA


def calculate_geometric_delays(LTEang, LTEdist, LTEheight, antspos, visidx):
    """Calculates geometric delays between pairs of antennas given the RFI tx location."""
    LTE = [LTEdist * np.cos(LTEang * 2 * np.pi / 360 + np.pi / 2),
           LTEdist * np.sin(LTEang * 2 * np.pi / 360 + np.pi / 2),
           LTEheight]
    dist_LTE_rx = np.sqrt((antspos[:, 0] - LTE[0]) ** 2 + (antspos[:, 1] - LTE[1]) ** 2 + (5 - LTE[2]) ** 2)
    delays = (dist_LTE_rx[visidx[:, 0]] - dist_LTE_rx[visidx[:, 1]]) / C
    return delays


def calculate_tracking_delays(azel, antspos, visidx):
    """Calculates tracking delays for each visibility given the tracking position."""
    az = np.deg2rad(azel[0].value + 90)
    el = np.deg2rad(azel[1].value)
    target_vec = np.array([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)])
    antstrackdelays = np.dot(antspos, target_vec[:2]) / C
    trackdelays = antstrackdelays[visidx[:, 0]] - antstrackdelays[visidx[:, 1]]
    return trackdelays


def calculate_visibilities(FSPL, SLA, visidx, geodelays, trackdelays, t_acf, acf, polAngle, LTEpow):
    """Computes actual visibilities with all considerations."""
    vis = np.zeros((len(geodelays), 1, 4), dtype=np.complex64)
    tot_att = FSPL[visidx[:, 0]] * FSPL[visidx[:, 1]] * SLA[visidx[:, 0]] * SLA[visidx[:, 1]]
    tot_del = geodelays + trackdelays
    delmin = np.min(tot_del)
    delmax = np.max(tot_del)
    delidx = np.where((t_acf >= delmin) & (t_acf <= delmax))[0]
    t_acf = t_acf[delidx]
    acf = acf[delidx]
    t_acf = t_acf[::200]
    acf = acf[::200]
    for k in tqdm(range(len(geodelays))):
        delayidx = np.argmin(np.abs(t_acf - tot_del[k]))
        vis[k, 0, :] = tot_att[k] * acf[delayidx]
    vis[:, :, 0] *= np.cos(np.deg2rad(polAngle)) ** 2
    vis[:, :, 1] *= np.cos(np.deg2rad(polAngle)) * np.sin(np.deg2rad(polAngle))
    vis[:, :, 2] *= np.cos(np.deg2rad(polAngle)) * np.sin(np.deg2rad(polAngle))
    vis[:, :, 3] *= np.sin(np.deg2rad(polAngle)) ** 2
    vis = vis * LTEpow / 13 * 10 ** 26
    vis = vis.astype(np.complex64)
    return vis


def write_visibilities_to_ms(vis_data, msfile, overwrite):
    """Writes visibility data to the MS table."""
    with table(msfile, readonly=False) as t:
        dtype = t.getcol('DATA').dtype
        vis_data = vis_data.astype(dtype)
        if not overwrite:  # Then we add to existing
            vis = t.getcol('DATA')
            vis_data += vis
        t.putcol('DATA', vis_data)


class RFISimConfig(SerialisableBaseModel):
    lte_distance: float = Field(
        desciption="Distance between RFI transmitter and center of the telescope [m].",
        default=46800.
    )
    lte_angle: float = Field(
        description="Angle between RFI transmitter and center of the telescope [deg., 0=N, 90=W].",
        default=160.
    )
    lte_height: float = Field(
        description="Height of RFI transmitter [m].",
        default=20.
    )
    lte_frequency: float = Field(
        description="Frequency of RFI signal [MHz].",
        default=770.
    )
    lte_polarization: float = Field(
        description="Polarization angle of RFI [deg, 0=full XX, 90=full YY].",
        default=10.
    )
    lte_power: float = Field(
        description="Power of RFI transmitter at the source [W/Hz].",
        default=6.4e-4
    )


def run_rfi_simulation(ms_file: str,
                       rfi_sim_config: RFISimConfig,
                       overwrite: bool = True):
    """
    Inject an RFI signal (from LTE cell towers) within an MS data set.

    Args:
        ms_file: MS file where the RFI will be injected.
        rfi_sim_config: Configuration for the RFI simulation.
        overwrite: Option to overwrite visibilities in input MS file instead of adding them. Default is True.
    """
    if not os.path.exists(ms_file):
        raise ValueError(f"MS file {ms_file} does not exist.")

    logger.info("Extracting data from MS file.")
    uvw, antenna_positions, azel, visibility_index = extract_ms_data(ms_file)

    logger.info("Calculating free space path loss.")
    fspl = calculate_free_space_path_loss(rfi_sim_config.lte_angle, rfi_sim_config.lte_distance,
                                          rfi_sim_config.lte_height, rfi_sim_config.lte_frequency, antenna_positions)

    logger.info("Calculating side lobes attenuation.")
    sla = calculate_side_lobes_attenuation(rfi_sim_config.lte_angle, rfi_sim_config.lte_distance,
                                           rfi_sim_config.lte_height, rfi_sim_config.lte_frequency, antenna_positions,
                                           azel)

    logger.info("Calculating geometric delays.")
    geometric_delays = calculate_geometric_delays(rfi_sim_config.lte_angle, rfi_sim_config.lte_distance,
                                                  rfi_sim_config.lte_height, antenna_positions, visibility_index)

    logger.info("Calculating tracking delays.")
    tracking_delays = calculate_tracking_delays(azel, antenna_positions, visibility_index)

    logger.info("Loading RFI data.")
    rfi_acf_data = loadmat(RFIData().rfi_injection_model())
    time_acf = rfi_acf_data['t_acf'][0]
    acf = rfi_acf_data['acf'][0]

    logger.info("Calculating visibilities.")
    visibilities = calculate_visibilities(fspl, sla, visibility_index, geometric_delays, tracking_delays, time_acf, acf,
                                          rfi_sim_config.lte_polarization, rfi_sim_config.lte_power)

    logger.info("Writing visibilities to MS.")
    write_visibilities_to_ms(visibilities, ms_file, overwrite)
