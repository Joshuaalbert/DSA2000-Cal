import os

from dsa2000_cal.rfi.rfi_simulation import run_rfi_simulation
from dsa2000_cal.run_config import RunConfig


def main(run_config: RunConfig):
    run_rfi_simulation(
        ms_file=run_config.rfi_visibilities_path,
        rfi_sim_config=run_config.rfi_sim_config,
        overwrite=True
    )


if __name__ == '__main__':
    if not os.path.exists('run_config.json'):
        raise ValueError('run_config.json must be present in the current directory.')
    run_config = RunConfig.parse_file('run_config.json')
    main(run_config)
