import os
import subprocess

from dsa2000_cal.run_config import RunConfig


def main(run_config: RunConfig):
    if not os.path.exists(run_config.bright_sky_model_bbs):
        raise ValueError(f"Bright sky model {run_config.bright_sky_model_bbs} does not exist.")
    if not os.path.exists(run_config.calibration_parset):
        raise ValueError(f"Calibration parset {run_config.calibration_parset} does not exist.")
    if not os.path.isdir('calibration'):
        raise ValueError("Calibration output directory does not exist.")

    num_cpus = os.cpu_count()

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
            f"input_ms.path={run_config.visibilities_path}",
            "input_ms.data_column=DATA",
            "input_ms.weight_column=WEIGHT",
            "input_ms.time_chunk=2",  # make sure multiple of time_interval
            "input_ms.freq_chunk=32",  # make sure multiple of freq_interval
            "input_model.recipe=bright_sky_model.lsm.html",
            "output.gain_directory=calibration/gains",
            "output.log_directory=calibration/logs",
            "output.log_to_terminal=True",
            "output.overwrite=True",
            "output.flags=True",
            "dask.threads=0",
            "dask.scheduler=threads",
            "solver.terms=G",
            "solver.iter_recipe=1000",
            "solver.propagate_flags=True",
            "solver.robust=False",
            f"solver.threads={num_cpus}",
            "solver.reference_antenna=0",
            "G.type=phase",
            "G.time_interval=2",
            "G.freq_interval=32"
        ]
    )

    if completed_process.returncode != 0:
        raise RuntimeError(f"command failed with return code {completed_process.returncode}")

    # Extract gains and analyse
    # gains = xds_from_zarr('calibration/gains::G')
    # print(gains)


if __name__ == '__main__':
    if not os.path.exists('run_config.json'):
        raise ValueError('run_config.json must be present in the current directory.')
    run_config = RunConfig.parse_file('run_config.json')
    main(run_config)
