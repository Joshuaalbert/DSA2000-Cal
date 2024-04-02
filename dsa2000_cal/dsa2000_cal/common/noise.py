from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from pyrap import tables as pt

from dsa2000_cal.assets.arrays.array import AbstractArray
from dsa2000_cal.common.quantity_utils import quantity_to_jnp


def calc_baseline_noise(system_equivalent_flux_density: float | jax.Array, chan_width_hz: float | jax.Array,
                        t_int_s: float | jax.Array) -> float:
    """Calculate the per visibility rms for identical antennas.

    Args:
        system_equivalent_flux_density (float): System Equivalent Flux Density (SEFD) per antennas in Jy
            (already includes efficiency)
        chan_width_hz (float): Channel width in Hz.
        t_int_s (float): Accumulation time in seconds.

    Returns:
        float: noise standard devation per part visibility.
    """
    # The 2 is for number of polarizations.
    return system_equivalent_flux_density / jnp.sqrt(2 * chan_width_hz * t_int_s)


def calc_image_noise(system_equivalent_flux_density: float, bandwidth_hz: float, t_int_s: float, num_antennas: int,
                     flag_frac: float) -> float:
    """
    Calculate the image noise for the central pixel.
    
    Args:
        system_equivalent_flux_density: the system equivalent flux density in Jy (already includes efficiency)
        bandwidth_hz: the bandwidth in Hz
        t_int_s: the integration time in seconds
        num_antennas: the number of antennas
        flag_frac: the fraction of flagged visibilities

    Returns:
        the image noise in Jy
    """
    num_baselines = (1. - flag_frac) * num_antennas * (num_antennas - 1) / 2.
    return calc_baseline_noise(system_equivalent_flux_density=system_equivalent_flux_density,
                               chan_width_hz=bandwidth_hz,
                               t_int_s=t_int_s) / jnp.sqrt(num_baselines)


def sum_and_add_noise(output_ms_file: str, input_ms_files: List[str], array: AbstractArray,
                      channel_width_hz: float, integration_time_s: float):
    """
    Sum the visibilities in the input measurement sets and add noise to the output measurement set.

    Args:
        output_ms_file: the output measurement set file
        input_ms_files: the input measurement set files
        array: the array object
    """
    noise_sigma = calc_baseline_noise(
        system_equivalent_flux_density=quantity_to_jnp(array.get_system_equivalent_flux_density(), 'Jy'),
        chan_width_hz=channel_width_hz,
        t_int_s=integration_time_s
    )
    print(f"Adding noise with sigma {noise_sigma} to {output_ms_file}")

    with pt.table(output_ms_file, readonly=False) as output_ms:
        shape = output_ms.getcol('DATA').shape
        dtype = output_ms.getcol('DATA').dtype
        # Divide noise scale by sqrt(2) to account for real and imaginary parts.
        output_with_noise = (
                np.random.normal(loc=0, scale=noise_sigma / np.sqrt(2.), size=shape).astype(dtype)
                + 1j * np.random.normal(loc=0, scale=noise_sigma / np.sqrt(2.), size=shape).astype(dtype)
        )
        for input_ms_file in input_ms_files:
            with pt.table(input_ms_file) as input_ms:
                output_with_noise += input_ms.col('DATA')[:]
        output_ms.putcol('DATA', output_with_noise)
