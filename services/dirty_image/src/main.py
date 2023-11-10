import logging
import os
import subprocess

from dsa2000_cal.faint_sky_model import write_diagonal_a_term_correction_file
from dsa2000_cal.run_config import RunConfig

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

    # Take the .fits off the end.
    image_name = os.path.join(
        os.path.dirname(run_config.faint_sky_model_fits),
        os.path.basename(run_config.faint_sky_model_fits).rsplit('-model.fits', 1)[0]
    )

    completed_process = subprocess.run(
        [
            'wsclean',
            '--help'
        ]
    )
    # # if completed_process.returncode != 0:
    # #     raise ValueError("Failed to run WSClean.")
    #
    # raise ValueError()
    num_cpus = os.cpu_count()

    completed_process = subprocess.run(
        [
            'wsclean',
            '-gridder', 'idg',
            '-idg-mode', 'cpu',  # Try hybrid
            '-wgridder-accuracy', '1e-4',
            # '-aterm-config', a_term_file,
            '-name', image_name,
            '-size', f"{run_config.image_size}", f"{run_config.image_size}",
            '-scale', f"{run_config.image_pixel_arcsec}asec",
            '-channels-out', '1',
            '-nwlayers-factor', '1',
            '-make-psf',
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
