import os
import shutil

import numpy as np
from h5parm import DataPack
from tomographic_kernel.frames import ENU
from tqdm import tqdm

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.astropy_utils import mean_itrs
from dsa2000_cal.faint_sky_model import prepare_gain_fits

fill_registries()
from dsa2000_cal.run_config import RunConfig
from dsa2000_cal.assets.registries import array_registry


def main(run_config: RunConfig):
    if not os.path.exists(run_config.ionosphere_h5parm):
        raise ValueError(f"Ionosphere H5parm file {run_config.ionosphere_h5parm} does not exist.")
    shutil.copy(run_config.ionosphere_h5parm, run_config.beam_h5parm)
    array = array_registry.get_instance(array_registry.get_match(run_config.array_name))
    antenna_beam = array.antenna_beam()
    with DataPack(run_config.beam_h5parm, readonly=False) as dp:
        dp.current_solset = 'sol000'
        if dp.axes_order != ['pol', 'ant', 'dir', 'freq', 'time']:
            raise ValueError("Expects axes order must be ['pol', 'ant', 'dir', 'freq', 'time']")
        phase, axes = dp.phase
        _, antennas = dp.get_antennas(axes['ant'])
        _, times = dp.get_times(axes['time'])
        _, freqs = dp.get_freqs(axes['freq'])
        _, directions = dp.get_directions(axes['dir'])  # [num_sources]
        dp.add_soltab(
            soltab='ampltude000',
            values=np.zeros_like(phase),
            axes=axes
        )
        amplitude, axes = dp.amplitude  # [1, num_antennas, num_sources, num_freqs, num_times]
        dp.delete_soltab('phase000')
    array_centre = mean_itrs(antennas)
    pbar = tqdm(times)
    for i, time in enumerate(pbar):
        enu_frame = ENU(location=array_centre.earth_location, obstime=time)
        for j, direction in enumerate(directions):
            for k, freq in enumerate(freqs):
                pbar.set_description(f"Computing beam at {time} for {direction} and {freq}")
                beam_amplitude = antenna_beam.get_amplitude(
                    pointing=run_config.pointing_centre,
                    source=direction,
                    freq_hz=freq.value,
                    enu_frame=enu_frame
                )  # [1]
                amplitude[0, :, j, k, i] = beam_amplitude[0]
    with DataPack(run_config.beam_h5parm, readonly=False) as dp:
        dp.current_solset = 'sol000'
        dp.amplitude = amplitude

    # get gains in  [num_time, num_ant, num_dir, num_freq, 2, 2]
    amplitude = np.transpose(amplitude, (4, 1, 2, 3, 0))  # [num_time, num_ant, num_dir, num_freq, pol]
    gains = np.zeros(amplitude.shape[:-1] + (2, 2), dtype=np.complex64)
    gains[..., 0, 0] = amplitude[..., 0]
    gains[..., 1, 1] = amplitude[..., 0]
    prepare_gain_fits(
        output_file=run_config.beam_fits,
        pointing_centre=run_config.pointing_centre,
        gains=gains,
        directions=directions,
        freq_hz=freqs.to('Hz').value,
        times=times,
        num_pix=32
    )


if __name__ == '__main__':
    if not os.path.exists('run_config.json'):
        raise ValueError('run_config.json must be present in the current directory.')
    run_config = RunConfig.parse_file('run_config.json')
    main(run_config)
