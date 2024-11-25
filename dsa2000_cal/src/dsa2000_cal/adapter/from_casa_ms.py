import os.path
from typing import Literal

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pyrap.tables as pt

from dsa2000_cal.adapter.utils import CASA_CORR_TYPES
from dsa2000_cal.common.astropy_utils import mean_itrs
from dsa2000_cal.measurement_sets.measurement_set import  MeasurementSet, MeasurementSetMeta, MeasurementSetMeta, \
    VisibilityData


def create_ms_meta(casa_ms: str, field_idx: int | None = None,
                   spectral_window_idx: int | None = None,
                   convention: Literal['engineering', 'physical'] = 'physical') -> MeasurementSetMeta:
    """
    Create a MeasurementSetMeta object from a CASA Measurement Set file.

    Args:
        casa_ms: the name of CASA Measurement Set file
        field_idx: the index of the field to extract, if None, the first field is used
        spectral_window_idx: the index of the spectral window to extract, if None, the first spectral window is used

    Returns:
        MeasurementSetMeta object
    """

    with pt.table(os.path.join(casa_ms, 'ANTENNA')) as t:
        antenna_position_m = t.getcol('POSITION')  # [num_ant, 3]
        # The exact frame should be specified in the MEASURE_REFERENCE keyword (ITRF or WGS84)
        if 'MEASURE_REFERENCE' in t.keywordnames():
            measure_reference = t.getkeyword('MEASURE_REFERENCE')
        else:
            measure_reference = 'ITRF'
        if measure_reference == 'ITRF':
            # Use ITRS frame
            antennas = ac.ITRS(x=antenna_position_m[:, 0] * au.m,
                               y=antenna_position_m[:, 1] * au.m,
                               z=antenna_position_m[:, 2] * au.m).earth_location
        elif measure_reference == 'WGS84':
            # Use EarthLocation frame
            antennas = ac.EarthLocation.from_geocentric(
                x=antenna_position_m[:, 0] * au.m,
                y=antenna_position_m[:, 1] * au.m,
                z=antenna_position_m[:, 2] * au.m
            )
        array_name = t.getcol('STATION')[0]  # [num_ant]
        antenna_names = list(t.getcol('NAME')[:])  # [num_ant]
        antenna_diameter_m = t.getcol('DISH_DIAMETER')  # [num_ant]
        antenna_diameters = antenna_diameter_m * au.m  # [num_ant]
        mount_types = list(t.getcol('MOUNT')[:])  # [num_ant]

    array_location = mean_itrs(antennas.get_itrs()).earth_location
    # array_location = antennas[0]

    with pt.table(os.path.join(casa_ms, 'FIELD')) as t:
        phase_center_rad = t.getcol('PHASE_DIR')  # [num_field, 1, 2]
        num_field, _, _ = phase_center_rad.shape
        if num_field > 1 and field_idx is None:
            raise ValueError("Multiple fields found, please specify field_idx.")
        if field_idx is None:
            field_idx = 0
        phase_center = ac.ICRS(ra=phase_center_rad[field_idx, 0, 0] * au.rad,
                                 dec=phase_center_rad[field_idx, 0, 1] * au.rad)

    with pt.table(os.path.join(casa_ms, 'SPECTRAL_WINDOW')) as t:
        freqs_hz = t.getcol('CHAN_FREQ')  # [num_spectral_windows, num_freqs]
        num_spectral_windows, num_freqs = freqs_hz.shape
        if num_spectral_windows > 1 and spectral_window_idx is None:
            raise ValueError("Multiple spectral windows found, please specify spectral_window_idx.")
        if spectral_window_idx is None:
            spectral_window_idx = 0
        freqs = freqs_hz[spectral_window_idx, :] * au.Hz
        channel_width_hz = t.getcol('CHAN_WIDTH')  # [num_spectral_windows, num_freqs]
        channel_width_hz = channel_width_hz[spectral_window_idx, :]
        # Only support single channel width for now
        if not np.all(channel_width_hz == channel_width_hz[0]):
            raise ValueError("Only support single channel width for now.")
        channel_width = channel_width_hz[0] * au.Hz

    with pt.table(os.path.join(casa_ms, 'POLARIZATION')) as t:
        corr_type = t.getcol('CORR_TYPE')  # [_, num_corrs]
        if corr_type.shape[0] > 1:
            raise ValueError("Multiple coherency types found.")
        coherencies = list(CASA_CORR_TYPES[x] for x in corr_type[0, :])  # [num_corrs]

    with pt.table(os.path.join(casa_ms, 'POINTING')) as t:
        if t.nrows() == 0:
            # Assuming there are no pointings, so all antennas point at zenith
            pointings = None
        else:
            pointing_rad = t.getcol('DIRECTION')  # [num_ant, 1, 2]
            pointings = ac.ICRS(ra=pointing_rad[:, 0, 0] * au.rad, dec=pointing_rad[:, 0, 1] * au.rad)  # [num_ant]

    with pt.table(casa_ms, readonly=True) as ms:
        # Get the shape of the antenna1
        antenna1 = ms.getcol('ANTENNA1')
        antenna2 = ms.getcol('ANTENNA2')
        if antenna1[0] == antenna2[0]:
            with_autocorr = True
        else:
            with_autocorr = False

        # Get the times where UVW is defined (We take not on the effective interval)
        times_tai_mjs = ms.getcol('TIME')[:]
        times = at.Time(np.unique(times_tai_mjs) / 86400., format='mjd', scale='utc')  # [num_rows]

        # Get integration time, before averaging and flagging
        interval = ms.getcol('INTERVAL')[:]  # [num_rows]
        if not np.all(interval == interval[0]):
            raise ValueError("Integration time is not constant.")
        integration_time = interval[0] * au.s

    system_equivalent_flux_density = None

    meta = MeasurementSetMeta(
        array_name=array_name,
        array_location=array_location,
        phase_center=phase_center,
        channel_width=channel_width,
        integration_time=integration_time,
        coherencies=coherencies,
        pointings=pointings,
        times=times,
        freqs=freqs,
        antennas=antennas,
        antenna_names=antenna_names,
        antenna_diameters=antenna_diameters,
        mount_types=mount_types,
        with_autocorr=with_autocorr,
        system_equivalent_flux_density=system_equivalent_flux_density,
        convention=convention
    )
    return meta


def transfer_from_casa(ms_folder: str,
                       casa_ms: str,
                       field_idx: int | None = None,
                       spectral_window_idx: int | None = None,
                       convention: Literal['engineering', 'physical'] = 'physical') -> MeasurementSet:
    """
    Transfer visibilities from the MeasurementSet to the output CASA Measurement Set.

    Args:
        casa_ms: the name of CASA Measurement Set file
    """
    # Create new MS file
    print(f"Creating new MeasurementSet {ms_folder} from CASA MS {casa_ms}")
    meta = create_ms_meta(casa_ms=casa_ms, field_idx=field_idx, spectral_window_idx=spectral_window_idx,
                          convention=convention)
    ms = MeasurementSet.create_measurement_set(ms_folder=ms_folder, meta=meta)

    # Populate the new MS file with visibilities
    print(f"Populating measurement set {ms_folder} with visibilities from {casa_ms}.")
    with pt.table(casa_ms, readonly=True) as output_ms:
        uvw_casa = output_ms.getcol('UVW')  # [rows, 3]

        uvw_internal = ms.get_uvw(row_slice=slice(None, None, None))
        uvw_offset = (uvw_casa - uvw_internal) * au.m
        print(f"UVW offset: {np.mean(uvw_offset, axis=0)} +- {np.std(uvw_offset, axis=0)}")

        start_row = 0
        num_rows = output_ms.nrows()
        while start_row < num_rows:
            end_row = min(start_row + ms.block_size, num_rows)
            # Times are MJS in tai scale
            time_mjs = output_ms.getcol('TIME', startrow=start_row, nrow=end_row - start_row)
            times = at.Time(time_mjs / 86400., format='mjd', scale='utc')
            antenna1 = output_ms.getcol('ANTENNA1', startrow=start_row, nrow=end_row - start_row)
            antenna2 = output_ms.getcol('ANTENNA2', startrow=start_row, nrow=end_row - start_row)

            if 'WEIGHT_SPECTRUM' not in output_ms.colnames():
                # Use WEIGHT and broadcast to all coherencies
                print("Using WEIGHT column for weights.")
                weights = output_ms.getcol('WEIGHT', startrow=start_row, nrow=end_row - start_row)  # [rows, num_corrs]
                weights = np.repeat(weights[:, None, :], len(meta.freqs), axis=1)  # [rows, num_freqs, num_corrs]
            else:
                weights = output_ms.getcol('WEIGHT_SPECTRUM', startrow=start_row,
                                           nrow=end_row - start_row)  # [rows, num_freqs, num_corrs]
            data = VisibilityData(
                vis=output_ms.getcol(
                    'DATA', startrow=start_row, nrow=end_row - start_row
                ),  # [rows, num_freqs, num_corrs]
                weights=weights,  # [rows, num_freqs, num_corrs]
                flags=output_ms.getcol(
                    'FLAG', startrow=start_row, nrow=end_row - start_row
                )  # [rows, num_freqs, num_corrs]
            )
            ms.put(
                data=data,
                antenna1=antenna1,
                antenna2=antenna2,
                times=times
            )
            start_row = end_row
    return ms
