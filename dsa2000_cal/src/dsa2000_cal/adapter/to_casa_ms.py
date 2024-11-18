import os.path
import shutil
import subprocess

import astropy.time as at
import astropy.units as au
import numpy as np
import pyrap.tables as pt

from dsa2000_cal.assets.templates.templates import Templates
from dsa2000_cal.measurement_sets.measurement_set import  MeasurementSet, MeasurementSetMeta


def create_makems_config(casa_ms: str,
                         meta: MeasurementSetMeta) -> str:
    """
    Create a makems config file for the given array.

    Args:
        casa_ms: name of the measurement set
        meta: the MeasurementSetMeta object

    Returns:
        path to the makems config file
    """

    antennas = meta.antennas
    antenna_names = meta.antenna_names
    mount_types = meta.mount_types
    start_freq = (meta.freqs[0] - 0.5 * meta.channel_width).to('Hz').value
    step_freq = meta.channel_width.to('Hz').value
    start_time = (meta.times[0] - 0.5 * meta.integration_time).datetime
    step_time = meta.integration_time.to('s').value
    phase_center = meta.pointings
    num_freqs = len(meta.freqs)
    num_times = len(meta.x)

    antennas_itrs = antennas.get_itrs()
    array_table = f"{meta.array_name}_ANTENNA"
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
            f.putcell('TYPE', i, 'GROUND-BASED')  # only kind of antenna supported
            f.putcell('DISH_DIAMETER', i, meta.antenna_diameters[i].to('m').value)
            f.putcell('FLAG_ROW', i, False)
            f.putcell('MOUNT', i, mount_types[i])
            f.putcell('NAME', i, antenna_names[i])
            f.putcell('STATION', i, meta.array_name)

    ra = phase_center.ra.to_string(unit='hour', sep=':', pad=True)
    dec = phase_center.dec.to_string(unit='deg', sep='.', pad=True, alwayssign=True)
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
        f'MSName={casa_ms}',
        'MSDesPath=.'
    ]
    config_file = os.path.abspath("makems.cfg")
    with open(config_file, 'w') as f:
        f.write('\n'.join(makems_config))

    return config_file


def create_empty_casa_ms(ms: MeasurementSet, casa_ms: str):
    config_file = create_makems_config(casa_ms=casa_ms,
                                       meta=ms.meta)
    # Run `makems ${makems_config_file}`
    completed_process = subprocess.run(["makems", config_file])
    if completed_process.returncode != 0:
        raise RuntimeError(f"makems failed with return code {completed_process.returncode}")

    # Rename from *_p0 to *
    os.rename(f"{casa_ms}_p0", casa_ms)


def transfer_to_casa(ms: MeasurementSet, casa_ms: str):
    """
    Transfer visibilities from the MeasurementSet to the output CASA Measurement Set.

    Args:
        ms: the MeasurementSet object
        casa_ms: the name of CASA Measurement Set file
    """
    # Create new MS file
    print(f"Creating new MS file {casa_ms}")
    if not ms.meta.with_autocorr:
        raise ValueError("Autocorrelations must be present in the MeasurementSet to map to CASA MS.")

    create_makems_config(
        meta=ms.meta,
        casa_ms=casa_ms,
    )

    # Populate the new MS file with visibilities
    print(f"Populating {casa_ms} with visibilities")
    with pt.table(casa_ms, readonly=False) as output_ms:
        # shape = output_ms.getcol('DATA').shape
        # dtype = output_ms.getcol('DATA').dtype

        # Make WEIGHT_SPECTRUM if it doesn't exist
        if 'WEIGHT_SPECTRUM' not in output_ms.colnames():
            data_desc = output_ms.getcoldesc('DATA')

            desc = pt.makecoldesc('WEIGHT_SPECTRUM', data_desc)
            desc['valueType'] = 'float'
            desc['shape'] = data_desc['shape'][::-1]  # [num_corrs, num_freqs] # reverse necessary
            desc['ndim'] = data_desc['ndim']

            dminfo = output_ms.getdminfo('DATA')
            dminfo['NAME'] = 'TiledWeightSpectrum'  # Use a unique data manager name
            output_ms.addcols(desc=desc, dminfo=dminfo)

        start_row = 0
        num_rows = output_ms.nrows()
        while start_row < num_rows:
            end_row = min(start_row + ms.block_size, num_rows)
            # Times are MJS in tai scale
            time_mjs = output_ms.getcol('TIME', startrow=start_row, nrow=end_row - start_row)
            times = at.Time(time_mjs / 86400., format='mjd', scale='utc')
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
