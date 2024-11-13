import sys
import warnings

import jax

from dsa2000_cal.forward_models.streaming.process import process_start

sys.tracebacklimit = None  # Increase as needed; -1 to suppress tracebacks


def main(process_id: int, plot_folder: str):
    warnings.warn("To run on CPU-only set 'JAX_PLATFORMS=cpu'.")

    process_start(
        process_id=process_id,
        key=jax.random.PRNGKey(0),
        array_name="dsa2000_31b",
        full_stokes=True,
        plot_folder=plot_folder
    )


if __name__ == '__main__':
    # # port = get_free_port()
    # main(1, 0, 'plots')
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--process_id", type=int, required=True, help="Process ID")
    parser.add_argument("--plot_folder", type=str, required=True, help="Plot folder")
    args = parser.parse_args()
    main(args.process_id, args.plot_folder)
