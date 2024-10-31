import logging
import os
import subprocess

from dsa2000_cal.common.fits_utils import write_diagonal_a_term_correction_file
from dsa2000_cal.models.run_config import RunConfig

logger = logging.getLogger(__name__)


def main(run_config: RunConfig):
    if run_config.faint_sky_model_fits is None:
        raise ValueError("Faint sky model must be specified to run FFT Predict.")

    # Create aterms.fits

    a_term_file = os.path.abspath('dirty_a_corr.parset')
    write_diagonal_a_term_correction_file(
        a_term_file=a_term_file,
        diagonal_gain_fits_files=[run_config.beam_fits]
    )

    completed_process = subprocess.run(
        [
            'wsclean',
            '--help'
        ]
    )

    num_cpus = os.cpu_count()
    image_name = 'dirty_dft'

    completed_process = subprocess.run(
        [
            'wsclean',
            '-gridder', 'idg',
            '-idg-mode', 'cpu',  # Try hybrid
            '-wgridder-accuracy', '1e-4',
            '-pol', 'i',
            # '-aterm-config', a_term_file,
            '-name', image_name,
            '-size', f"{run_config.image_size}", f"{run_config.image_size}",
            '-scale', f"{run_config.image_pixel_arcsec}asec",
            '-channels-out', '1',
            '-nwlayers-factor', '1',
            '-weight', 'natural',
            '-j', f'{num_cpus}',
            run_config.dft_visibilities_path
        ]
    )
    if completed_process.returncode != 0:
        raise ValueError("Failed to run WSClean.")


if __name__ == '__main__':
    if not os.path.exists('run_config.json'):
        raise ValueError('run_config.json must be present in the current directory.')
    run_config = RunConfig.parse_file('run_config.json')
    main(run_config)
