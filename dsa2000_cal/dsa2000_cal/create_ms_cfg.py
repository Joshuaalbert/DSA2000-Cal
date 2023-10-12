import os.path
import shutil
from datetime import datetime

import astropy.coordinates as ac
import astropy.units as au
import numpy as np
import pyrap.tables as pt

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.assets.templates.templates import Templates

fill_registries()
from dsa2000_cal.assets.arrays.array import AbstractArray


def create_makems_config(array_name: str,
                         ms_name: str,
                         start_freq: float,
                         step_freq: float,
                         start_time: datetime,
                         step_time: float,
                         pointing_direction: ac.ICRS,
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
        pointing_direction: pointing direction
        num_freqs: number of frequencies
        num_times: number of times

    Returns:
        path to the makems config file
    """
    array: AbstractArray = array_registry.get_instance(array_registry.get_match(array_name))
    antennas = array.get_antennas()
    antenna_names = array.get_antenna_names()
    array_table = f"{array_name}_ANTENNA"
    # Make table with the 8 colums: ['OFFSET', 'POSITION', 'TYPE', 'DISH_DIAMETER', 'FLAG_ROW', 'MOUNT', 'NAME', 'STATION']
    # Copy template table to array_table
    shutil.copytree(Templates().template_antenna_table(), array_table)
    # Add rows to array_table
    with pt.table(array_table, readonly=False) as f:
        nrows = f.nrows()
        diff = len(antennas) - nrows
        if diff > 0:
            f.addrows(diff)
        elif diff < 0:
            f.removerows([0] * (-diff))
        for i, antenna in enumerate(antennas):
            f.putcell('OFFSET', i, np.zeros(3))
            f.putcell('POSITION', i, antenna.cartesian.xyz.to(au.m).value)
            f.putcell('TYPE', i, 'GROUND-BASED')
            f.putcell('DISH_DIAMETER', i, array.get_antenna_diameter())
            f.putcell('FLAG_ROW', i, False)
            f.putcell('MOUNT', i, array.get_mount_type())
            f.putcell('NAME', i, antenna_names[i])
            f.putcell('STATION', i, array.get_station_name())

    ra = pointing_direction.ra.to_string(unit='hour', sep=':', pad=True)
    dec = pointing_direction.dec.to_string(unit='deg', sep='.', pad=True, alwayssign=True)
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

    return config_file
