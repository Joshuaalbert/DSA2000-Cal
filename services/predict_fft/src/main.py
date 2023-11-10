import logging
import os
import subprocess

from h5parm import DataPack
from pyrap import tables as pt

from dsa2000_cal.faint_sky_model import write_diagonal_a_term_correction_file, prepare_gain_fits
from dsa2000_cal.gains import extract_scalar_gains
from dsa2000_cal.run_config import RunConfig

logger = logging.getLogger(__name__)

def make_diagonal_a_term_correction_files(run_config:RunConfig):

    with DataPack(run_config.beam_h5parm, readonly=True) as dp:
        dp.current_solset = 'sol000'
        if dp.axes_order != ['pol', 'dir', 'ant', 'freq', 'time']:
            raise ValueError(f"Expects axes order must be ['pol', 'dir', 'ant', 'freq', 'time'], got {dp.axes_order}")
        axes = dp.axes_phase
        _, antennas = dp.get_antennas(axes['ant'])
        _, times = dp.get_times(axes['time'])
        _, freqs = dp.get_freqs(axes['freq'])
        _, directions = dp.get_directions(axes['dir'])  # [num_sources]

    # get gains in  [num_time, num_ant, num_dir, num_freq, 2, 2]
    gains = extract_scalar_gains(h5parm=run_config.beam_h5parm)

    prepare_gain_fits(
        output_file=run_config.beam_fits,
        pointing_centre=run_config.pointing_centre,
        gains=gains,
        directions=directions,
        freq_hz=freqs.to('Hz').value,
        times=times,
        num_pix=32
    )


    with DataPack(run_config.ionosphere_h5parm, readonly=True) as dp:
        # get phase
        dp.current_solset = 'sol000'
        if dp.axes_order != ['pol', 'dir', 'ant', 'freq', 'time']:
            raise ValueError(f"Expects axes order must be ['pol', 'dir', 'ant', 'freq', 'time'], got {dp.axes_order}")
        axes = dp.axes_phase
        _, antennas = dp.get_antennas(axes['ant'])
        _, times = dp.get_times(axes['time'])
        _, freqs = dp.get_freqs(axes['freq'])
        _, directions = dp.get_directions(axes['dir'])  # [num_sources]

    # get gains in  [num_time, num_ant, num_dir, num_freq, 2, 2]
    gains = extract_scalar_gains(h5parm=run_config.ionosphere_h5parm)
    prepare_gain_fits(
        output_file=run_config.ionosphere_fits,
        pointing_centre=run_config.pointing_centre,
        gains=gains,
        directions=directions,
        freq_hz=freqs.to('Hz').value,
        times=times,
        num_pix=32
    )


def main(run_config: RunConfig):
    if run_config.faint_sky_model_fits is None:
        raise ValueError("Faint sky model must be specified to run FFT Predict.")

    # Create aterms.fits
    a_term_file = os.path.abspath('predict_fft_a_corr.parset')
    write_diagonal_a_term_correction_file(
        a_term_file=a_term_file,
        diagonal_gain_fits_files=[run_config.beam_fits, run_config.ionosphere_fits]
    )

    # Take the .fits off the end.
    image_name = os.path.join(
        os.path.dirname(run_config.faint_sky_model_fits),
        os.path.basename(run_config.faint_sky_model_fits).rsplit('-model.fits', 1)[0]
    )
    num_cpus = os.cpu_count()
    completed_process = subprocess.run(
        [
            'wsclean',
            '-predict',
            '-gridder', 'idg',
            '-idg-mode', 'cpu',  # Try hybrid
            # '-aterm-config', a_term_file,
            '-wgridder-accuracy', '1e-4',
            '-nwlayers-factor', '1',
            '-channels-out', '1',
            '-j', f'{num_cpus}',
            '-name', image_name,
            run_config.fft_visibilities_path
        ]
    )
    if completed_process.returncode != 0:
        raise ValueError("Failed to run WSClean.")

    with pt.table(run_config.fft_visibilities_path, readonly=False) as vis_table:
        data = vis_table.getcol('MODEL_DATA')
        vis_table.putcol('DATA', data)
    logger.info(f"Predicted visibilities written to {run_config.fft_visibilities_path}")


if __name__ == '__main__':
    if not os.path.exists('run_config.json'):
        raise ValueError('run_config.json must be present in the current directory.')
    run_config = RunConfig.parse_file('run_config.json')
    main(run_config)
