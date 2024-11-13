import logging
import sys
import warnings
from uuid import uuid4

import ray

from dsa2000_cal.forward_models.streaming.process_actor import SFMProcessParams, SFMProcess

sys.tracebacklimit = None  # Increase as needed; -1 to suppress tracebacks


def main(num_processes: int, process_id: int, plot_folder: str):
    warnings.warn("To run on CPU-only set 'JAX_PLATFORMS=cpu'.")

    ray.init(
        address="auto",
        logging_level=logging.DEBUG,
        log_to_driver=False,
        _enable_object_reconstruction=False
    )

    process = SFMProcess(
        worker_id=str(uuid4()),
        params=SFMProcessParams(num_processes=num_processes, process_id=process_id, plot_folder=plot_folder)
    )
    process.run()


if __name__ == '__main__':
    # # port = get_free_port()
    # main(1, 0, 'plots')
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, required=True, help="Number of processes")
    parser.add_argument("--process_id", type=int, required=True, help="Process ID")
    parser.add_argument("--plot_folder", type=str, required=True, help="Plot folder")
    args = parser.parse_args()
    main(args.num_processes, args.process_id, args.plot_folder)
