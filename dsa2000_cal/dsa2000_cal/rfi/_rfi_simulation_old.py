import argparse

import astropy.time
import astropy.units as u
import numpy as np
import utm
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from casacore.tables import table
from scipy.io import loadmat
from tqdm import tqdm

'''
Here I temporarily define input parameters
Note : I am assuming the polarization order is XX*, XY*, YX*, YY* -- TO BE VERIFIED
msfile = '/home/ghellbourg/rfi_sim/1-chan-template.ms'
SITE_LOCATION = EarthLocation(lat='38.4433', lon='-116.2731', height=1563) #generic for hot creek valley, can specify exact later
LTEdist = 46800 #aggressor antenna distance from center of array
LTEang = 160    #angle of antenna (cardinal) from N of array
LTEheight = 80  #height of aggressor antenna relative to height of array (assumes array zero reference)
LTEfreq = 770   # RFI center frequency in MHz
polAngle = 10;  # polarization angle in degrees : 0 = full XX, 90 = full YY
LTEpow = 6.4e-4;   # power in W/Hz
'''

'''
constants
'''
c = 299792458.;
antenna_model = '/home/ghellbourg/RFISim/DSA2000_antenna_model.mat';
rfi_acf = '/home/ghellbourg/RFISim/rfi_inject.mat';


def get_ms_data(msfile):
    '''
    This function extracts all relevant data from an MS file.
    Assumes an MS file containing an individual frequency channel and a single 1.5s integration
    input is path to MS file
    outputs are:
    * uvw : coordinates of uvw points in m
    * antspos : coordinates of individual antennas in m
    * azel : AzEl pointing of the antennas assumed fixed over the integration time
    * visidx : gives antennas indexes for each visibility
    '''
    t = table(msfile);
    uvw = t.getcol('UVW');
    ant1 = t.getcol('ANTENNA1');
    ant2 = t.getcol('ANTENNA2');
    t.close();
    visidx = np.zeros((len(ant1), 2), dtype=int);
    visidx[:, 0] = ant1;
    visidx[:, 1] = ant2;
    t = table(msfile + '/ANTENNA');
    antspos = t.getcol('POSITION');
    t.close();
    antspos = EarthLocation.from_geocentric(x=antspos[:, 0] * u.m, y=antspos[:, 1] * u.m,
                                            z=antspos[:, 2] * u.m).to_geodetic();
    arr_center = [np.mean(antspos[1]), np.mean(antspos[0])];
    antspos = utm.from_latlon(np.array(antspos.lat), np.array(antspos.lon));
    antspos = np.array([antspos[0] - np.mean(antspos[0]), antspos[1] - np.mean(antspos[1])]);
    t = table(msfile + '/POINTING');
    tim = t.getcol('TIME')[0];
    tim = astropy.time.Time(tim / 86400, format='mjd');
    point = t.getcol('DIRECTION')[0][0];
    t.close();

    observing_location = EarthLocation(lat=arr_center[0], lon=arr_center[1], height=5 * u.m);
    observing_time = tim;
    aa = AltAz(location=observing_location, obstime=observing_time);
    coord = SkyCoord(point[0], point[1], unit="deg");
    azel = coord.transform_to(aa);
    azel = [azel.az, azel.alt];
    return uvw, antspos.T, azel, visidx


def vis_to_ms(vis_data, msfile, ow):
    '''
    writes visibility data in MS table.
    '''
    vis_data = vis_data.astype(np.complex64);
    with table(msfile, readonly=False) as t:
        if ow:
            vis = t.getcol('DATA');
            vis_data += vis;
        t.putcol('DATA', vis_data);


def FreeSpacePathLoss(LTEang, LTEdist, LTEheight, LTEfreq, antspos):
    '''
    calculates free space path loss between the RFI transmitter and individual antennas.
    Returns an array of path attenuation for each antenna.
    '''
    FSPL = np.zeros(len(antspos));
    LTE = [LTEdist * np.cos(LTEang * 2 * np.pi / 360 + np.pi / 2), \
           LTEdist * np.sin(LTEang * 2 * np.pi / 360 + np.pi / 2), \
           LTEheight];
    for k in range(len(antspos)):
        d = np.sqrt((antspos[k, 0] - LTE[0]) ** 2 + (antspos[k, 1] - LTE[1]) ** 2 + (
                5 - LTE[2]) ** 2);  # 5 m altitude for each antenna
        FSPL[k] = (c / (4 * np.pi * d * LTEfreq * 1e6)) ** 2;
    return FSPL;


def SideLobesAtt(LTEang, LTEdist, LTEheight, LTEfreq, antspos, azel):
    '''
    computes side lobe attenuation for each telescope antenna given the location of the RFI transmitter.
    Based on antenna_model file.
    '''
    ant_model = loadmat(antenna_model);
    I = np.argmin(np.abs(ant_model["freqListGHz"] - LTEfreq * 1e-3));
    SLA = np.zeros(len(antspos));
    az = np.deg2rad(azel[0].value + 90);
    el = np.deg2rad(azel[1].value);
    ArrSteer = np.array([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)]);
    LTE = [LTEdist * np.cos(LTEang * 2 * np.pi / 360 + np.pi / 2), \
           LTEdist * np.sin(LTEang * 2 * np.pi / 360 + np.pi / 2), \
           LTEheight];
    normArrSteer = np.sqrt(np.dot(ArrSteer, ArrSteer));
    for k in range(len(antspos)):
        vec = [(antspos[k, 0] - LTE[0]), (antspos[k, 1] - LTE[1]), 5 - LTE[2]];
        ang = np.dot(ArrSteer, vec) / np.sqrt(np.dot(vec, vec)) / normArrSteer;
        ang = np.arccos(ang);
        idx = np.argmin(np.abs(ant_model["ThetaDeg"] - np.rad2deg(ang)));
        SLA[k] = 10 ** (ant_model["coPolPattern_dBi_Freqs_15DegConicalShield"][idx, 0, I] / 10);
    return SLA;


def calc_geodelays(LTEang, LTEdist, LTEheight, antspos, visidx):
    '''
    calculates geometrical delays between pairs of antennas given the location of the RFI tx
    '''
    LTE = [LTEdist * np.cos(LTEang * 2 * np.pi / 360 + np.pi / 2), \
           LTEdist * np.sin(LTEang * 2 * np.pi / 360 + np.pi / 2), \
           LTEheight];
    dist_LTE_rx = np.sqrt((antspos[:, 0] - LTE[0]) ** 2 + (antspos[:, 1] - LTE[1]) ** 2 + (5 - LTE[2]) ** 2);
    delays = (dist_LTE_rx[visidx[:, 0]] - dist_LTE_rx[visidx[:, 1]]) / c;
    return delays


def calc_trackdelays(azel, antspos, visidx):
    '''
    calculates tracking delays for each visibilitiy given the tracking position indicated in the MS file
    '''
    az = np.deg2rad(azel[0].value + 90);
    el = np.deg2rad(azel[1].value);
    target_vec = np.array([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)]);
    antstrackdelays = np.dot(antspos, target_vec[:2]) / c;
    trackdelays = antstrackdelays[visidx[:, 0]] - antstrackdelays[visidx[:, 1]];
    return trackdelays


def calc_vis(FSPL, SLA, visidx, geodelays, trackdelays, t_acf, acf, polAngle, LTEpow):
    '''
    computes the actual visibilities, including geometric delays, tracking delays, propagation loss, side lobes attenuation, and polarization angle
    '''
    vis = np.zeros((len(geodelays), 1, 4), dtype=np.complex64);
    tot_att = FSPL[visidx[:, 0]] * FSPL[visidx[:, 1]] * SLA[visidx[:, 0]] * SLA[
        visidx[:, 1]];  # total attenuation path loss + side lobes
    tot_del = geodelays + trackdelays;  # total delay on each baseline
    delmin = np.min(tot_del);
    delmax = np.max(tot_del);
    delidx = np.where((t_acf >= delmin) & (t_acf <= delmax))[0];
    t_acf = t_acf[delidx];
    acf = acf[delidx];
    t_acf = t_acf[::200];
    acf = acf[::200];
    for k in tqdm(range(len(geodelays))):
        delayidx = np.argmin(np.abs(t_acf - tot_del[k]));
        vis[k, 0, :] = tot_att[k] * acf[delayidx]
        # print(k);

    vis[:, :, 0] *= np.cos(np.deg2rad(polAngle)) ** 2;
    vis[:, :, 1] *= np.cos(np.deg2rad(polAngle)) * np.sin(np.deg2rad(polAngle));
    vis[:, :, 2] *= np.cos(np.deg2rad(polAngle)) * np.sin(np.deg2rad(polAngle));
    vis[:, :, 3] *= np.sin(np.deg2rad(polAngle)) ** 2;

    vis = vis * LTEpow / 13 * 10 ** 26;  # convert in Jy, assuming 13m2 collecting area
    vis = vis.astype(np.complex64);
    return vis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inject_rfi.py',
                                     description='This program inject an RFI signal within an MS data set.')
    parser.add_argument('MSfile', help='MS file where the RFI will be injected in');
    parser.add_argument('--LTEdist', type=float,
                        help='Distance between RFI transmitter and center of the telescope [m]', default=46800);
    parser.add_argument('--LTEang', type=float,
                        help='Angle between RFI transmitter and center of the telescope [deg., 0=N, 90=W]',
                        default=160);
    parser.add_argument('--LTEheight', type=float, help='Height of RFI transmitter [m]', default=20);
    parser.add_argument('--LTEfreq', type=float, help='Frequency of RFI signal [MHz]', default=770);
    parser.add_argument('--LTEpol', type=float, help='Polarization angle of RFI [deg, 0=full XX, 90=full YY]',
                        default=10);
    parser.add_argument('--LTEpow', type=float, help='Power of RFI transmitter at the sources [W/Hz]', default=6.4e-4);
    parser.add_argument('--TelLat', type=float, help='Latitude of telescope center', default=38.4433);
    parser.add_argument('--TelLon', type=float, help='Longitude of telescope center', default=-116.2731);
    parser.add_argument('--OverWrite', action='store_true',
                        help='Option to overwrite visibilities in input MS file as opposed to add them');
    args = parser.parse_args();

    msfile = args.MSfile;
    SITE_LOCATION = EarthLocation(lat=str(args.TelLat), lon=str(args.TelLon),
                                  height=1563);  # generic for hot creek valley, can specify exact later
    LTEdist = float(args.LTEdist);  # aggressor antenna distance from center of array
    LTEang = float(args.LTEang);  # angle of antenna (cardinal) from N of array
    LTEheight = float(
        args.LTEheight);  # height of aggressor antenna relative to height of array (assumes array zero reference)
    LTEfreq = float(args.LTEfreq);  # RFI center frequency in MHz
    polAngle = float(args.LTEpol);  # polarization angle in degrees : 0 = full XX, 90 = full YY
    LTEpow = float(args.LTEpow);  # power in W/Hz
    ow = args.OverWrite;  # overwriting flag -- false by default

    uvw, antspos, azel, visidx = get_ms_data(msfile);
    print('read MS file and extracted all relevant information.');
    FSPL = FreeSpacePathLoss(LTEang, LTEdist, LTEheight, LTEfreq, antspos);
    print('computed path loss between RFI Tx and telescops Rx');
    SLA = SideLobesAtt(LTEang, LTEdist, LTEheight, LTEfreq, antspos, azel);
    print('computed side lobe attenuation');
    geodelays = calc_geodelays(LTEang, LTEdist, LTEheight, antspos, visidx);
    print('computed geometric delays');
    trackdelays = calc_trackdelays(azel, antspos, visidx);
    print('computed tracking delays');
    rfi_acf = loadmat(rfi_acf);
    t_acf = rfi_acf['t_acf'][0];
    acf = rfi_acf['acf'][0];
    print('RFI correlation function loaded');
    print('computing visbilities...');
    vis = calc_vis(FSPL, SLA, visidx, geodelays, trackdelays, t_acf, acf, polAngle, LTEpow);
    print('Visibilities computed');
    vis_to_ms(vis, msfile, ow);
    print('Visibilities injected in ' + msfile);
