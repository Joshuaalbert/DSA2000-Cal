import os

from dsa2000_fm.forward_models.streaming.calibrator import Calibration

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

from typing import NamedTuple, List

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.numpy as jnp
import jax.random
import numpy as np
import pyrap.tables as pt

from dsa2000_common.common.corr_utils import CASA_CORR_TYPES, broadcast_translate_corrs
from dsa2000_common.common.array_types import ComplexArray, FloatArray, BoolArray, IntArray
from dsa2000_cal.common.astropy_utils import mean_itrs
from dsa2000_common.common.quantity_utils import time_to_jnp, quantity_to_jnp


class MSData(NamedTuple):
    vis_data: ComplexArray  # [T, B, chan, 2, 2]
    weights: FloatArray  # [T, B, chan, 2, 2]
    flags: BoolArray  # [T, B, chan, 2, 2]
    antenna1: IntArray  # [B]
    antenna2: IntArray  # [B]
    freqs: FloatArray  # [chan]
    times: FloatArray  # [T]
    num_antennas: int
    original_coherencies: List[str | int] = None


def read_casa_ms(casa_ms, times_per_chunk: int, data_column: str = 'DATA', field_idx=None, spectral_window_idx=None,
                 store_response: bool = False):
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
    num_antennas = len(antenna_names)

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
            pointing_rad = t.getcol('DIRECTION')  # [num_ant, 2, 1] # Sometimes [num_ant, 1, 2]
            print(pointing_rad.shape)
            pointings = ac.ICRS(ra=pointing_rad[:, 0, 0] * au.rad, dec=pointing_rad[:, 1, 0] * au.rad)  # [num_ant]

    with pt.table(casa_ms, readonly=False) as ms:
        num_rows = ms.nrows()

        ref_time = at.Time(ms.getcol('TIME', startrow=0, nrow=1)[0] / 86400., format='mjd', scale='tt')  # [1]
        # Get the shape of the antenna1
        antenna10 = ms.getcol('ANTENNA1', startrow=0, nrow=1)
        antenna20 = ms.getcol('ANTENNA2', startrow=0, nrow=1)
        if antenna10[0] == antenna20[0]:
            with_autocorr = True
        else:
            with_autocorr = False

        if with_autocorr:
            num_baselines = num_antennas * (num_antennas + 1) // 2
        else:
            num_baselines = num_antennas * (num_antennas - 1) // 2

        rows_per_chunk = times_per_chunk * num_baselines
        for row_idx in range(0, num_rows, rows_per_chunk):

            # Get the shape of the antenna1
            antenna1 = ms.getcol('ANTENNA1', startrow=row_idx, nrow=rows_per_chunk)
            antenna2 = ms.getcol('ANTENNA2', startrow=row_idx, nrow=rows_per_chunk)

            antenna1 = np.reshape(antenna1, (times_per_chunk, num_baselines))[0]  # [B]
            antenna2 = np.reshape(antenna2, (times_per_chunk, num_baselines))[0]  # [B]
            B = np.shape(antenna1)[0]

            # Get the times where UVW is defined (We take not on the effective interval)
            times_tai_mjs = ms.getcol('TIME', startrow=row_idx, nrow=rows_per_chunk)
            times = np.reshape(times_tai_mjs, (times_per_chunk, num_baselines))[:, 0]  # [T]
            times = at.Time(times / 86400., format='mjd', scale='tt')  # [T]

            # Get integration time, before averaging and flagging
            interval = ms.getcol('INTERVAL', startrow=row_idx, nrow=rows_per_chunk)[:]  # [num_rows]
            if not np.all(interval == interval[0]):
                raise ValueError("Integration time is not constant.")
            integration_time = interval[0] * au.s

            times = time_to_jnp(times, ref_time)  # [T]
            T = len(times)

            vis_data = ms.getcol(data_column)  # [num_rows, num_chan, coh ]
            flags = ms.getcol('FLAG', startrow=row_idx, nrow=rows_per_chunk)  # [num_rows, num_chan, coh]

            if 'WEIGHT_SPECTRUM' not in ms.colnames():
                # Use WEIGHT and broadcast to all coherencies
                print("Using WEIGHT column for weights.")
                weights = ms.getcol('WEIGHT', startrow=row_idx, nrow=rows_per_chunk)  # [rows, num_corrs]
                weights = np.repeat(weights[:, None, :], len(freqs), axis=1)  # [rows, num_freqs, num_corrs]
            else:
                weights = ms.getcol('WEIGHT_SPECTRUM', startrow=row_idx,
                                    nrow=rows_per_chunk)  # [rows, num_freqs, num_corrs]

            print(vis_data.shape)
            print(weights.shape)
            print(flags.shape)

            vis_data = jnp.asarray(np.reshape(vis_data, (T, B, num_freqs, len(coherencies))))
            vis_data = broadcast_translate_corrs(vis_data, from_corrs=tuple(coherencies),
                                                 to_corrs=(("XX", "XY"), ("YX", "YY")))

            weights = jnp.asarray(np.reshape(weights, (T, B, num_freqs, len(coherencies))))
            weights = broadcast_translate_corrs(weights, from_corrs=tuple(coherencies),
                                                to_corrs=(("XX", "XY"), ("YX", "YY"))).astype(np.float32)

            flags = jnp.asarray(np.reshape(flags, (T, B, num_freqs, len(coherencies))))
            flags = broadcast_translate_corrs(flags.astype(np.float32), from_corrs=tuple(coherencies),
                                              to_corrs=(("XX", "XY"), ("YX", "YY"))).astype(np.bool_)
        response = yield MSData(
            vis_data=vis_data,
            weights=weights,
            flags=flags,
            antenna1=antenna1,
            antenna2=antenna2,
            times=times,
            freqs=quantity_to_jnp(freqs),
            num_antennas=num_antennas,
            original_coherencies=coherencies
        )
        # Set
        if store_response:
            new_vis, = response
            ms.putcol('DATA', new_vis, startrow=row_idx, nrow=rows_per_chunk)


def main(data_ms: str, subtract_ms_list: List[str], no_subtract_ms_list: List[str], times_per_chunk: int):
    if not os.path.exists(data_ms):
        raise ValueError(f"Data Measurement Set {data_ms} does not exist.")
    for ms in subtract_ms_list + no_subtract_ms_list:
        if not os.path.exists(ms):
            raise ValueError(f"Model measurement Set {ms} does not exist.")
    # Add a column to the data Measurement Set "DATA_RESIDUALS" to store the residuals
    # add_residual_column(data_ms)

    # Read the data
    data_gen = read_casa_ms(data_ms, data_column='DATA', times_per_chunk=times_per_chunk)
    subtract_gen_list = [read_casa_ms(ms, data_column='DATA', times_per_chunk=times_per_chunk) for ms in
                         subtract_ms_list]
    no_subtract_gen_list = [read_casa_ms(ms, data_column='DATA', times_per_chunk=times_per_chunk) for ms in
                            no_subtract_ms_list]

    response = None
    last_solver_state = None
    while True:
        try:
            ms_data = data_gen.send(response)
            subtract_ms_data = [next(gen) for gen in subtract_gen_list]
            no_subtract_ms_data = [next(gen) for gen in no_subtract_gen_list]
        except StopIteration:
            break

        vis_data = ms_data.vis_data
        weights = ms_data.weights
        flags = ms_data.flags
        antenna1 = ms_data.antenna1
        antenna2 = ms_data.antenna2
        freqs = ms_data.freqs
        times = ms_data.times
        num_antennas = ms_data.num_antennas
        original_coherencies = ms_data.original_coherencies
        del ms_data

        if len(freqs) % len(jax.devices()) != 0:
            raise ValueError("Number of channels must be divisible by the number of devices to shard.")

        T, B, F, _, _ = np.shape(vis_data)

        # Get model visibilities
        subtract_vis = [ms.vis_data for ms in subtract_ms_data]
        no_subtract_vis = [ms.vis_data for ms in no_subtract_ms_data]
        vis_model = jnp.stack(subtract_vis + no_subtract_vis, axis=0)  # [D, T, B, F, 2, 2]

        del subtract_vis
        del no_subtract_vis

        calibration = Calibration(
            full_stokes=True,
            num_ant=num_antennas,
            num_background_source_models=len(no_subtract_ms_list),
            verbose=True
        )
        calibrate_and_subtract_jit = jax.jit(calibration.step)

        device = jax.devices()[0]
        args = jax.tree.map(
            lambda x: jax.device_put(x, device),
            (vis_model, vis_data, weights, flags, freqs, times, antenna1, antenna2, last_solver_state)
        )

        gains, vis_data_residuals, solver_state, diagnostics = calibrate_and_subtract_jit(*args)

        last_solver_state = solver_state

        # Convert the residuals to the original coherencies
        vis_data_residuals = broadcast_translate_corrs(
            vis_data_residuals, from_corrs=(("XX", "XY"), ("YX", "YY")),
            to_corrs=tuple(original_coherencies)
        )  # [T, B, F, coh]

        vis_data_residuals = np.reshape(vis_data_residuals, (T * B, F, -1))

        response = (vis_data_residuals,)


def add_residual_column(data_ms):
    """
    Add a column to the data Measurement Set "DATA_RESIDUALS" to store the residuals

    Args:
        data_ms: The data Measurement Set to add the column to.
    """
    with pt.table(data_ms, readonly=False) as ms:
        if 'DATA_RESIDUALS' not in ms.colnames():
            # Get the column description of the 'DATA' column
            data_desc = ms.getcoldesc('DATA')

            # Modify the column name to 'DATA_RESIDUALS'
            data_residuals_desc = data_desc.copy()
            data_residuals_desc['name'] = 'DATA_RESIDUALS'

            # Create a table description with the new column
            newcols_desc = pt.maketabdesc([data_residuals_desc])

            # Add the new column to the table
            ms.addcols(newcols_desc)


if __name__ == '__main__':
    # create arg parser
    import argparse


    def parse_list_str(list_str):
        return list_str.split(',')


    parser = argparse.ArgumentParser(
        description='DD Calibrate CASA Measurement Set against several other model Measurement Sets, and subtract.')
    parser.add_argument('--data_ms', type=str, help='The data Measurement Set to calibrate.')
    parser.add_argument('--subtract_ms_list', type=parse_list_str,
                        help='The list of Measurement Sets to subtract.')
    parser.add_argument('--no_subtract_ms_list', type=parse_list_str,
                        help='The list of Measurement Sets to not subtract.')
    parser.add_argument('--times_per_chunk', type=int, default=100, help='The block size to process the data in.')

    # Example usage:
    # python main.py --data_ms /path/to/data.ms --subtract_ms_list /path/to/subtract1.ms /path/to/subtract2.ms --no_subtract_ms_list /path/to/no_subtract1.ms /path/to/no_subtract2.ms --block_size 100

    args = parser.parse_args()
    main(args.data_ms, args.subtract_ms_list, args.no_subtract_ms_list, args.times_per_chunk)
