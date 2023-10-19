import os
import subprocess

from dsa2000_cal.run_config import RunConfig


def main(run_config: RunConfig):
    if not os.path.exists(run_config.bright_sky_model_bbs):
        raise ValueError(f"Bright sky model {run_config.bright_sky_model_bbs} does not exist.")
    if not os.path.exists(run_config.calibration_parset):
        raise ValueError(f"Calibration parset {run_config.calibration_parset} does not exist.")

    print("Converting sky model to Tigger.")
    completed_process = subprocess.run(
        [
            "tigger-convert",
            "-t", "BBS",
            "-o", "Tigger",
            run_config.bright_sky_model_bbs,
            "bright_sky_model.lsm.html"
        ]
    )
    if completed_process.returncode != 0:
        raise RuntimeError(f"tigger-convert failed with return code {completed_process.returncode}")

    print("Running goquartical.")
    completed_process = subprocess.run(
        [
            "goquartical",
            run_config.calibration_parset
        ]
    )
    if completed_process.returncode != 0:
        raise RuntimeError(f"goquartical failed with return code {completed_process.returncode}")


if __name__ == '__main__':
    if not os.path.exists('run_config.json'):
        raise ValueError('run_config.json must be present in the current directory.')
    run_config = RunConfig.parse_file('run_config.json')
    main(run_config)
