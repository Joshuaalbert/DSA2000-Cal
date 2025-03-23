import os

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

from typing import Generator
import itertools

import numpy as np
from astropy import time as at, units as au, coordinates as ac
from jax import numpy as jnp

from dsa2000_cal.iterative_calibrator import create_data_input_gen, Data, IterativeCalibrator, DataGenInput
from dsa2000_common.common.mixed_precision_utils import mp_policy


def data_generator(input_gen: Generator[DataGenInput, None, None], num_ant: int):
    for input_data in input_gen:
        # Get the data corresponding to the input_data from somewhere and prepare the data

        # input_data is a DataGenInput object
        print(input_data)

        # ---- This section should come from your data (e.g. MS's) ----

        time_idxs = input_data.time_idxs  # [T] the time indices to get from the MS
        freq_idxs = input_data.freq_idxs  # [C] the frequency indices to get from the MS
        sol_int_time_idx = input_data.sol_int_time_idx
        # Note: in MS vis are shaped [rows, channels, coh]
        # Rows are time-major stacked, so to get the `sol_int_time_idx` block do:
        B = (num_ant * (num_ant - 1)) // 2  # N*(N+1)/2 with auto-correlations, or N*(N-1)/2 without auto-correlations
        block_size = B * len(time_idxs)
        # visibilities = ms.get_data('DATA', startrow=sol_int_time_idx * block_size, nrow=block_size)

        antenna1, antenna2 = jnp.asarray(list(itertools.combinations_with_replacement(range(num_ant), 2)),
                                         dtype=mp_policy.index_dtype).T
        B = len(antenna1)

        coherencies = ('XX', 'XY', 'YX', 'YY')
        vis_data_shape = (len(input_data.time_idxs), B, len(input_data.freq_idxs), len(coherencies))
        num_bright_sources = 3
        num_background_sources = 1

        # These are the high resolution visibilities from the data.
        # Shape: [T, B, C, coh]
        # T: number of times, unaveraged
        # B: number of baselines
        # C: number of channels, unaveraged
        # coh: number of coherencies

        vis_data = np.random.normal(size=vis_data_shape) + 1j * np.random.normal(size=vis_data_shape)
        weights = np.ones(vis_data_shape)
        flags = np.zeros(vis_data_shape)

        # There are two model visibilities:
        # 1. vis_bright_sources: visibilities from D bright sources, that will be subtracted.
        # 2. vis_background: visibilities from E background sources, which are not subtracted.

        # vis_bright_sources: visibilities from D bright sources, that will be subtracted.
        # Shape: [D, Tm, B, Cm, coh]
        # D: number of bright sources
        # Tm: number of model times, i.e. len(input_data.model_times)
        # B: number of baselines
        # Cm: number of model frequencies, i.e. len(input_data.model_freqs)
        # coh: number of coherencies
        vis_bright_sources = np.random.normal(
            size=(
                num_bright_sources,
                len(input_data.model_times),
                B,
                len(input_data.model_freqs),
                len(coherencies)
            )) + 1j * np.random.normal(
            size=(
                num_bright_sources,
                len(input_data.model_times),
                B,
                len(input_data.model_freqs),
                len(coherencies))
        )

        # vis_background: visibilities from E background sources, which are not subtracted.
        # Shape: [E, Tm, B, Cm, coh]
        # E: number of background sources
        # Tm: number of model times, i.e. len(input_data.model_times)
        # B: number of baselines
        # Cm: number of model frequencies, i.e. len(input_data.model_freqs)
        # coh: number of coherencies

        vis_background = np.random.normal(
            size=(
                num_background_sources,
                len(input_data.model_times),
                B,
                len(input_data.model_freqs),
                len(coherencies)
            )) + 1j * np.random.normal(
            size=(
                num_background_sources,
                len(input_data.model_times),
                B,
                len(input_data.model_freqs),
                len(coherencies))
        )

        # ---- End of data preparation ----

        return_data = yield Data(
            sol_int_time_idx=input_data.sol_int_time_idx,
            coherencies=coherencies,
            vis_data=vis_data,
            weights=weights,
            flags=flags,
            vis_bright_sources=vis_bright_sources,
            vis_background=vis_background,
            antenna1=antenna1,
            antenna2=antenna2,
            model_times=input_data.model_times,
            model_freqs=input_data.model_freqs,
            ref_time=input_data.ref_time
        )

        # ---- This section where you should store the return_data ----

        # The return_data is a ReturnData object
        # Shape: [T, B, C, coh]
        # The gains have been applied to vis_bright_sources, and tiled to match the raw visibilities, and subtracted.

        # Store return_data if desired
        print(f"Storing residuals for solution interval {input_data.sol_int_time_idx}")
        # print(return_data.vis_residuals)
        # print(return_data.gains)

        # ---- End of data storage ----


def main():
    # ---- This section is where you create the generator that will be used to drive calibration ----

    # Get these from your MS
    # ref_time: astropy Time object, the reference time for the observation, e.g. the start time of the observation
    # integration_time: astropy Quantity object, the integration time of the observation
    # obstimes: astropy Time object, the times you will calibrate over, unaveraged
    # obsfreqs: astropy Quantity object, the frequencies you will calibrate over, unaveraged
    # antennas: astropy EarthLocation object, the locations of the antennas used in the observation

    ref_time = at.Time.now()  # Use the right scale, likely TAI, UTC, or TT.
    integration_time = 1.5 * au.s
    obstimes = ref_time + np.arange(8) * integration_time
    obsfreqs = np.linspace(700, 2000, 40) * au.MHz
    antennas = ac.EarthLocation.from_geocentric(
        10e3 * np.random.uniform(size=352) * au.m,
        10e3 * np.random.uniform(size=352) * au.m,
        10e3 * np.random.uniform(size=352) * au.m
    )

    # Use this utility function to feed your data generator. Read it's docstring for more information.
    input_gen = create_data_input_gen(
        sol_int_freq_idx=0,
        T=4,
        C=40,
        Tm=1,
        Cm=1,
        obsfreqs=obsfreqs,
        obstimes=obstimes,
        ref_time=ref_time
    )

    your_generator = data_generator(input_gen, num_ant=len(antennas))

    # ---- End of generator creation ----

    # ---- This section is where you create the calibrator and run the calibration ----

    full_stokes = True

    calibrator = IterativeCalibrator(
        plot_folder='demo_plots',
        run_name='demo',
        gain_stddev=1.,
        dd_dof=1,
        di_dof=1,
        double_differential=True,
        dd_type='unconstrained',
        di_type='unconstrained',
        full_stokes=full_stokes,
        antennas=antennas,
        verbose=True,  # if using GPU's set to False
        num_devices=8,
        backend='cpu'
    )

    # Run the calibration
    # Ts and Cs are optional. You can fight decoherence by using larger Ts and Cs.
    # Ts: your solution interval data is averaged down to this many time chunks. default is 1
    # Cs: your solution interval data is averaged down to this many frequency chunks. default is 1
    calibrator.run(your_generator, Ts=None, Cs=None)

    # ---- End of calibration ----


if __name__ == '__main__':
    main()
