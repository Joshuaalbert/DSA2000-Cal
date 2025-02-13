import os
from typing import Generator

import jax

# TODO: Set environment variables for JAX at top of script before imports
# To use all GPU's on a single machine, set the following environment variable
# os.environ['JAX_PLATFORMS'] = 'cuda'  # Use GPU devices by default
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # On-demand GPU memory allocation
# devices = jax.devices(backend='gpu')
# To use all CPU's on a single machine for XLA, set the following environment variable
os.environ[
    "XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"  # Use all CPU's on a single machine
os.environ['JAX_PLATFORMS'] = 'cpu'  # Use CPUs by default

import itertools

import numpy as np
from astropy import time as at, units as au, coordinates as ac
from jax import numpy as jnp

from dsa2000_cal.calibration.iterative_calibrator import create_data_input_gen, Data, IterativeCalibrator, DataGenInput
from dsa2000_cal.common.mixed_precision_utils import mp_policy


def data_generator(input_gen: Generator[DataGenInput, None, None], num_ant: int):
    for input_data in input_gen:
        # Get the data corresponding to the input_data from somewhere and prepare the data

        print(input_data)

        antenna1, antenna2 = jnp.asarray(list(itertools.combinations_with_replacement(range(num_ant), 2)),
                                         dtype=mp_policy.index_dtype).T
        B = len(antenna1)

        coherencies = ('XX', 'XY', 'YX', 'YY')
        vis_data_shape = (len(input_data.time_idxs), B, len(input_data.freq_idxs), len(coherencies))
        num_bright_sources = 3
        num_background_sources = 1

        vis_data = np.random.normal(size=vis_data_shape) + 1j * np.random.normal(size=vis_data_shape)
        weights = np.ones(vis_data_shape)
        flags = np.zeros(vis_data_shape)

        # These are the model visibilities for the bright sources and background sources, computed at the model times and frequencies.
        # If they are precomputed at a different resolution, they should be interpolated to the model times and frequencies.
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

        # Store return_data if desired
        print(f"Storing residuals for solution interval {input_data.sol_int_time_idx}")
        # print(return_data.vis_residuals)


def main():
    # devices = jax.devices()
    devices = None # single process, single device

    # Get obs data from somewhere, like MS
    ref_time = at.Time.now()  # Can choose to be first time in obs, use the right scale, likely TAI, UTC, or TT.
    integration_time = 1.5 * au.s
    obstimes = ref_time + np.arange(8) * integration_time
    obsfreqs = np.linspace(700, 2000, 10000) * au.MHz
    antennas = ac.EarthLocation.from_geocentric(
        10e3 * np.random.uniform(size=10) * au.m,
        10e3 * np.random.uniform(size=10) * au.m,
        10e3 * np.random.uniform(size=10) * au.m
    )

    # Create this utility generator to drive your data generation
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

    # Can create IterativeCalibrator yourself, or else use the create_simple_calibrator function.
    calibrator = IterativeCalibrator.create_simple_calibrator(
        plot_folder='demo_plots',
        run_name='demo',
        full_stokes=True,
        antennas=antennas,
        verbose=True,
        devices=devices
    )

    # Run the calibration
    calibrator.run(data_generator(input_gen, num_ant=len(antennas)), Ts=None, Cs=None)


if __name__ == '__main__':
    main()
