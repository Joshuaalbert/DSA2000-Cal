import logging
import os
import subprocess

from pyrap import tables as pt

from dsa2000_cal.faint_sky_model import write_diagonal_a_term_correction_file
from dsa2000_cal.run_config import RunConfig

logger = logging.getLogger(__name__)


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

    # symlink run_config.faint_sky_model_fits to ...-model-pb.fits
    os.symlink(
        run_config.faint_sky_model_fits,
        os.path.join(
            os.path.dirname(run_config.faint_sky_model_fits),
            os.path.basename(run_config.faint_sky_model_fits).replace('-model.fits', '-model-pb.fits')
        )
    )

    num_cpus = os.cpu_count()
    completed_process = subprocess.run(
        [
            'wsclean',
            '-predict',
            '-gridder', 'idg',
            '-idg-mode', 'cpu',  # Try hybrid
            '-pol','i',
            '-aterm-config', a_term_file,
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
