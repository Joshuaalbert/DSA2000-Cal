import logging
from typing import List

import numpy as np
from pyrap import tables as pt

from dsa2000_cal.assets.arrays.array import AbstractArray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calc_noise(system_equivalent_flux_density: float, chan_width_hz: float, t_int_s: float,
               system_efficiency: float) -> float:
    """Calculate the per visibility rms for identical antennas.
    Args:
        system_equivalent_flux_density (float): System Equivalent Flux Density (SEFD) per antennas in Jy.
        chan_width_khz (float): Channel width in Hz.
        t_int_s (float): Accumulation time in seconds.
        system_efficiency (float): System efficiency.

    Returns:
        float: noise standard devation per part visibility.
    """
    return (system_equivalent_flux_density / system_efficiency) / np.sqrt(2 * chan_width_hz * t_int_s)


def sum_and_add_noise(output_ms_file: str, input_ms_files: List[str], array: AbstractArray,
                      channel_width_hz: float, integration_time_s: float):
    """
    Sum the visibilities in the input measurement sets and add noise to the output measurement set.

    Args:
        output_ms_file: the output measurement set file
        input_ms_files: the input measurement set files
        array: the array object
    """
    noise_sigma = calc_noise(
        system_equivalent_flux_density=array.system_equivalent_flux_density(),
        chan_width_hz=channel_width_hz,
        t_int_s=integration_time_s,
        system_efficiency=array.system_efficency()
    )
    logger.info(f"Adding noise with sigma {noise_sigma} to {output_ms_file}")

    with pt.table(output_ms_file, readonly=False) as output_ms:
        shape = output_ms.getcol('DATA').shape
        dtype = output_ms.getcol('DATA').dtype
        output_with_noise = np.random.normal(loc=0, scale=noise_sigma, size=shape).astype(dtype)
        for input_ms_file in input_ms_files:
            with pt.table(input_ms_file) as input_ms:
                output_with_noise += input_ms.col('DATA')[...]
        output_ms.putcol('DATA', output_with_noise)
