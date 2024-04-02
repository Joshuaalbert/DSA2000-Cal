import os.path
import shutil
import subprocess
from datetime import datetime

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pyrap.tables as pt

from dsa2000_cal.assets.content_registry import fill_registries

fill_registries()

from dsa2000_cal.assets.arrays.array import AbstractArray
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.assets.templates.templates import Templates
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet


def create_makems_config(array_name: str,
                         ms_name: str,
                         start_freq: float,
                         step_freq: float,
                         start_time: datetime,
                         step_time: float,
                         phase_tracking: ac.ICRS,
                         num_freqs: int,
                         num_times: int) -> str:
    """
    Create a makems config file for the given array.

    Args:
        array_name: name of the array
        ms_name: name of the measurement set
        start_freq: start frequency in Hz
        step_freq: frequency step in Hz
        start_time: start time
        step_time: time step in seconds
        phase_tracking: pointing direction
        num_freqs: number of frequencies
        num_times: number of times

    Returns:
        path to the makems config file
    """
    fill_registries()
    print(array_registry.entries)
    array: AbstractArray = array_registry.get_instance(array_registry.get_match(array_name))
    antennas_itrs = array.get_antennas().get_itrs()
    antenna_names = array.get_antenna_names()
    array_table = f"{array_name}_ANTENNA"
    # Make table with the 8 colums: ['OFFSET', 'POSITION', 'TYPE', 'DISH_DIAMETER', 'FLAG_ROW', 'MOUNT', 'NAME', 'STATION']
    # Copy template table to array_table
    shutil.copytree(Templates().template_antenna_table(), array_table)
    # Add rows to array_table
    with pt.table(array_table, readonly=False) as f:
        nrows = f.nrows()
        diff = len(antennas_itrs) - nrows
        if diff > 0:
            f.addrows(diff)
        elif diff < 0:
            f.removerows([0] * (-diff))
        for i, antenna in enumerate(antennas_itrs):
            f.putcell('OFFSET', i, np.zeros(3))
            f.putcell('POSITION', i, antenna.cartesian.xyz.to(au.m).value)
            f.putcell('TYPE', i, 'GROUND-BASED')
            f.putcell('DISH_DIAMETER', i, array.get_antenna_diameter())
            f.putcell('FLAG_ROW', i, False)
            f.putcell('MOUNT', i, array.get_mount_type())
            f.putcell('NAME', i, antenna_names[i])
            f.putcell('STATION', i, array.get_station_name())

    ra = phase_tracking.ra.to_string(unit='hour', sep=':', pad=True)
    dec = phase_tracking.dec.to_string(unit='deg', sep='.', pad=True, alwayssign=True)
    start_time_str = start_time.strftime('%Y/%m/%d/%H:%M:%S')
    makems_config = [
        f'StartFreq={start_freq}',
        f'StepFreq={step_freq}',
        f'StartTime={start_time_str}',
        f'StepTime={step_time}',
        f'RightAscension={ra}',
        f'Declination={dec}',
        'NBands=1',
        f'NFrequencies={num_freqs}',
        f'NTimes={num_times}',
        'NParts=1',
        'WriteImagerColumns=F',
        'WriteAutoCorr=T',
        f'AntennaTableName={array_table}',
        f'MSName={ms_name}',
        'MSDesPath=.'
    ]
    config_file = os.path.abspath("makems.cfg")
    with open(config_file, 'w') as f:
        f.write('\n'.join(makems_config))

    # Run `makems ${makems_config_file}`
    completed_process = subprocess.run(["makems", config_file])
    if completed_process.returncode != 0:
        raise RuntimeError(f"makems failed with return code {completed_process.returncode}")

    return config_file


def transfer_visibilities(ms: MeasurementSet, ms_file_name: str):
    """
    Transfer visibilities from the MeasurementSet to the output CASA Measurement Set.

    Args:
        ms: the MeasurementSet object
        ms_file_name: the name of CASA Measurement Set file
    """
    if len(ms.meta.times) == 1:
        step_time = ms.meta.integration_time.to('s').value
    else:
        step_time = (ms.meta.times[1].tai - ms.meta.times[0].tai).sec
    # Create new MS file
    create_makems_config(
        array_name=ms.meta.array_name,
        ms_name=ms_file_name,
        start_freq=ms.meta.freqs[0].to('Hz').value,
        step_freq=ms.meta.channel_width.to('Hz').value,
        start_time=ms.meta.times[0].tai.mjd * 86400.,
        step_time=step_time,
        phase_tracking=ms.meta.pointings,
        num_freqs=len(ms.meta.freqs),
        num_times=len(ms.meta.times)
    )

    # Populate the new MS file with visibilities
    with pt.table(ms_file_name, readonly=False) as output_ms:
        # shape = output_ms.getcol('DATA').shape
        # dtype = output_ms.getcol('DATA').dtype

        # Make WEIGHT_SPECTRUM if it doesn't exist
        if 'WEIGHT_SPECTRUM' not in output_ms.colnames():
            desc = pt.makecoldesc('WEIGHT_SPECTRUM', output_ms.getcoldesc('DATA'))
            dminfo = output_ms.getdminfo('DATA')
            output_ms.addcols(desc=desc, dminfo=dminfo)

        start_row = 0
        num_rows = output_ms.nrows()
        while start_row < num_rows:
            end_row = min(start_row + ms.block_size, num_rows)
            # Times are MJS in tai scale
            time_mjs = output_ms.getcol('TIME', startrow=start_row, nrow=end_row - start_row)
            times = at.Time(time_mjs / 86400., format='mjd', scale='tai')
            antenna_1 = output_ms.getcol('ANTENNA1', startrow=start_row, nrow=end_row - start_row)
            antenna_2 = output_ms.getcol('ANTENNA2', startrow=start_row, nrow=end_row - start_row)

            data = ms.match(antenna_1=antenna_1, antenna_2=antenna_2, times=times)
            output_ms.putcol(
                'DATA', data.vis, startrow=start_row, nrow=end_row - start_row
            )  # [num_freqs, num_corrs]
            output_ms.putcol(
                'WEIGHT_SPECTRUM', data.weights, startrow=start_row, nrow=end_row - start_row
            )  # [num_freqs, num_corrs]
            output_ms.putcol(
                'FLAG', data.flags, startrow=start_row, nrow=end_row - start_row
            )  # [num_freqs, num_corrs]
            start_row = end_row
