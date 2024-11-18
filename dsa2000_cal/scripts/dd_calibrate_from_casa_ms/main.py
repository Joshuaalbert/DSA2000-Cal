import os

from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

from functools import partial
from typing import NamedTuple, List

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.numpy as jnp
import jax.random
import jaxns.framework.context as ctx
import numpy as np
import pyrap.tables as pt
from jaxns.framework.ops import simulate_prior_model
from tqdm import tqdm

from dsa2000_cal.adapter.utils import CASA_CORR_TYPES, broadcast_translate_corrs
from dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardt, MultiStepLevenbergMarquardtState
from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import UnconstrainedGain
from dsa2000_cal.common.array_types import ComplexArray, FloatArray, BoolArray, IntArray
from dsa2000_cal.common.astropy_utils import mean_itrs
from dsa2000_cal.common.jax_utils import multi_vmap, create_mesh
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import time_to_jnp, quantity_to_jnp
from dsa2000_cal.common.vec_utils import kron_product


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


def read_casa_ms(casa_ms, data_column: str = 'DATA', field_idx=None, spectral_window_idx=None):
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
        times = at.Time(np.unique(times_tai_mjs) / 86400., format='mjd', scale='utc')  # [T]

        # Get integration time, before averaging and flagging
        interval = ms.getcol('INTERVAL')[:]  # [num_rows]
        if not np.all(interval == interval[0]):
            raise ValueError("Integration time is not constant.")
        integration_time = interval[0] * au.s

        ref_time = times[0]

        times = time_to_jnp(times, ref_time)  # [T]
        T = len(times)
        antenna1 = np.reshape(antenna1, (T, -1))
        antenna2 = np.reshape(antenna2, (T, -1))
        _, B = np.shape(antenna1)

        vis_data = ms.getcol(data_column)  # [num_rows, num_chan, coh ]
        vis_data = np.reshape(vis_data, (T, B, num_freqs, len(coherencies)))
        vis_data = broadcast_translate_corrs(vis_data, from_corrs=tuple(coherencies),
                                             to_corrs=(("XX", "XY"), ("YX", "YY")))
        weights = ms.getcol('WEIGHT')  # [num_rows, num_chan, coh]
        weights = np.reshape(weights, (T, B, num_freqs, len(coherencies)))
        weights = broadcast_translate_corrs(weights, from_corrs=tuple(coherencies),
                                            to_corrs=(("XX", "XY"), ("YX", "YY"))).astype(np.float32)
        flags = ms.getcol('FLAG')  # [num_rows, num_chan, coh]
        flags = np.reshape(flags, (T, B, num_freqs, len(coherencies)))
        flags = broadcast_translate_corrs(flags.astype(np.float32), from_corrs=tuple(coherencies),
                                          to_corrs=(("XX", "XY"), ("YX", "YY"))).astype(np.bool_)
    return MSData(
        vis_data=vis_data,
        weights=weights,
        flags=flags,
        antenna1=antenna1,
        antenna2=antenna2,
        times=times,
        freqs=quantity_to_jnp(freqs),
        num_antennas=len(antenna_names),
        original_coherencies=coherencies
    )


# all devices work on channels
mesh = create_mesh((1, 1, len(jax.devices())), ('T', 'B', 'F'), devices=jax.devices())


@partial(
    shard_map,
    mesh=mesh,
    in_specs=(PartitionSpec(), PartitionSpec('T', 'B', 'F'), PartitionSpec('T', 'B', 'F'),
              PartitionSpec('T', 'B', 'F'), PartitionSpec('T', 'B', 'F'),
              PartitionSpec('B'), PartitionSpec('B')),
    out_specs=PartitionSpec('T', 'B', 'F')
)
def compute_residuals(
        gains: ComplexArray,
        vis_per_direction: ComplexArray,
        vis_data: ComplexArray,
        weights: FloatArray,
        flags: BoolArray,
        antenna1: IntArray,
        antenna2: IntArray,
        weighted: bool = True
):
    """
    Compute the residual between the model visibilities and the observed visibilities.

    Args:
        gains: [A, F, D, 2, 2]
        vis_per_direction: [T, B, F, D, 2, 2]
        vis_data: [T, B, F, 2, 2]
        weights: [T, B, F, 2, 2]
        flags: [T, B, F, 2, 2]
        antenna1: [B]
        antenna2: [B]

    Returns:
        residuals: [T, B, F, 2, 2]
    """

    if np.shape(weights) != np.shape(flags):
        raise ValueError(f"Visibilities shape {np.shape(vis_per_direction)} must match flags shape {np.shape(flags)}.")

    # Compute the model visibilities
    g1 = gains[antenna1]  # [B, F, D, 2, 2]
    g2 = gains[antenna2]  # [B, F, D, 2, 2]

    @partial(
        multi_vmap,
        in_mapping="[B,F,D,2,2],[B,F,D,2,2],[T,B,F,D,2,2]",
        out_mapping="[T,B,F,D,~P,~Q]",
        verbose=True
    )
    def apply_gains(g1, g2, vis):
        if np.shape(g1) != np.shape(g1):
            raise ValueError("Gains must have the same shape.")
        if np.shape(vis) != np.shape(g1):
            raise ValueError("Gains and visibilities must have the same shape.")
        if np.shape(g1) == (2, 2):
            return mp_policy.cast_to_vis(kron_product(g1, vis, g2.conj().T))
        elif np.shape(g1) == ():
            return mp_policy.cast_to_vis(g1 * vis * g2.conj())
        else:
            raise ValueError(f"Invalid shape: {np.shape(g1)}")

    model_vis = apply_gains(g1, g2, vis_per_direction)  # [T, B, F, D, 2, 2]
    model_vis = jnp.sum(model_vis, axis=-3)  # [T, B, F, 2, 2]
    residuals = model_vis - vis_data  # [T, B, F, 2, 2]
    if weighted:
        weights = jnp.where(flags, jnp.zeros_like(weights), weights)  # [T, B, F, 2, 2]
        residuals = residuals * weights  # [T, B, F, 2, 2]
    return residuals


def calibrate_and_subtract(state: MultiStepLevenbergMarquardtState | None, vis_per_direction: ComplexArray,
                           vis_data: ComplexArray,
                           weights: FloatArray,
                           flags: BoolArray,
                           antenna1: IntArray, antenna2: IntArray,
                           freqs: FloatArray, times: FloatArray, num_antennas: int, num_subtract: int):
    T, B, F, D, _, _ = np.shape(vis_per_direction)
    key = jax.random.PRNGKey(0)

    # Create gain prior model
    def get_gains():
        gain_probabilistic_model = UnconstrainedGain()
        mean_time = jnp.mean(times)
        prior_model = gain_probabilistic_model.build_prior_model(
            num_source=D,
            num_ant=num_antennas,
            freqs=freqs,
            times=mean_time[None]
        )
        (gains,), _ = simulate_prior_model(key, prior_model)  # [1, A, F, D, 2, 2]
        return gains[0]

    get_gains_transformed = ctx.transform(get_gains)

    # Create residual_fn
    def residual_fn(params: ComplexArray) -> ComplexArray:
        gains = get_gains_transformed.apply(params, key).fn_val
        return compute_residuals(gains, vis_per_direction, vis_data, weights, flags, antenna1, antenna2)

    solver: MultiStepLevenbergMarquardt = MultiStepLevenbergMarquardt(
        residual_fn=residual_fn,
        num_approx_steps=0,
        num_iterations=100,
        verbose=True,
        gtol=1e-4
    )

    # Get solver state
    if state is None:
        init_params = get_gains_transformed.init(key).params
        state = solver.create_initial_state(init_params)
    else:
        state = solver.update_initial_state(state)
    state, diagnostics = solver.solve(state)

    gains = get_gains_transformed.apply(state.x, key).fn_val

    vis_data_residuals = compute_residuals(gains, vis_per_direction[..., :num_subtract, :, :],
                                           vis_data, weights, flags, antenna1,
                                           antenna2, weighted=False)

    return gains, vis_data_residuals, state, diagnostics


def main(data_ms: str, subtract_ms_list: List[str], no_subtract_ms_list: List[str], block_size: int):
    if not os.path.exists(data_ms):
        raise ValueError(f"Data Measurement Set {data_ms} does not exist.")
    for ms in subtract_ms_list + no_subtract_ms_list:
        if not os.path.exists(ms):
            raise ValueError(f"Model measurement Set {ms} does not exist.")
    # Add a column to the data Measurement Set "DATA_RESIDUALS" to store the residuals
    add_residual_column(data_ms)
    # Read the data
    ms_data = read_casa_ms(data_ms, data_column='DATA')
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
    subtract_ms_data = [read_casa_ms(ms, data_column='DATA') for ms in subtract_ms_list]
    no_subtract_ms_data = [read_casa_ms(ms, data_column='DATA') for ms in no_subtract_ms_list]

    num_subtract = len(subtract_ms_data)
    vis_per_direction = jnp.stack([ms.vis_data for ms in no_subtract_ms_data], axis=-3)  # [T, B, F, D, 2, 2]
    del subtract_ms_data
    del no_subtract_ms_data

    calibrate_and_subtract_jit = jax.jit(calibrate_and_subtract, static_argnames=['num_antennas'])

    # Let's go one step at a time
    init_state = None
    for time_idx in tqdm(range(0, len(times), block_size)):
        time_slice = slice(time_idx, time_idx + block_size)
        # Calibrate
        gains, vis_data_residuals, state, diagnostics = calibrate_and_subtract_jit(
            init_state, vis_per_direction[time_slice],
            vis_data[time_slice], weights[time_slice],
            flags[time_slice], antenna1, antenna2,
            freqs, times[time_slice],
            num_antennas, num_subtract
        )

        init_state = state
        # Convert the residuals to the original coherencies
        vis_data_residuals = broadcast_translate_corrs(
            vis_data_residuals, from_corrs=(("XX", "XY"), ("YX", "YY")),
            to_corrs=tuple(original_coherencies)
        )  # [T, B, F, 4]

        # Write the residuals to the data Measurement Set
        with pt.table(data_ms, readonly=False) as ms:
            start_row = time_idx * B
            end_row = (time_idx + block_size) * B
            # set slice
            vis_data_residuals = np.reshape(vis_data_residuals, (T * B, F, -1))
            ms.putcol('DATA_RESIDUALS', vis_data_residuals, startrow=start_row, nrow=end_row - start_row)


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

    parser = argparse.ArgumentParser(
        description='DD Calibrate CASA Measurement Set against several other model Measurement Sets, and subtract.')
    parser.add_argument('--data_ms', type=str, help='The data Measurement Set to calibrate.')
    parser.add_argument('--subtract_ms_list', type=str, nargs='+', help='The list of Measurement Sets to subtract.')
    parser.add_argument('--no_subtract_ms_list', type=str, nargs='+',
                        help='The list of Measurement Sets to not subtract.')
    parser.add_argument('--block_size', type=int, default=100, help='The block size to process the data in.')

    # Example usage:
    # python main.py --data_ms /path/to/data.ms --subtract_ms_list /path/to/subtract1.ms /path/to/subtract2.ms --no_subtract_ms_list /path/to/no_subtract1.ms /path/to/no_subtract2.ms --block_size 100

    args = parser.parse_args()
    main(args.data_ms, args.subtract_ms_list, args.no_subtract_ms_list, args.block_size)
