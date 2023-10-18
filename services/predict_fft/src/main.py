import logging
import os

from dsa2000_cal.assets.content_registry import fill_registries

fill_registries()

from dsa2000_cal.run_config import RunConfig
from pyrap import tables as pt
import subprocess

logger = logging.getLogger(__name__)


def main(run_config: RunConfig):
    if run_config.faint_sky_model_fits is None:
        raise ValueError("Faint sky model must be specified to run FFT Predict.")

    # Create aterms.fits

    # Take the .fits off the end.
    image_name = os.path.join(
        os.path.dirname(run_config.faint_sky_model_fits),
        os.path.basename(run_config.faint_sky_model_fits).split('.')[0]
    )
    completed_process = subprocess.run(
        [
            'wsclean',
            '-predict',
            '-gridder', 'wgridder',
            '-wgridder-accuracy', '1e-4',
            # '-aterm-config', run_config.a_term_parset,
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
