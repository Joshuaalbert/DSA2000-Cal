import datetime
import os
import warnings

import jax

from dsa2000_cal.common.ray_utils import get_free_port

# Set num jax devices to number of CPUs
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"
jax.config.update('jax_threefry_partitionable', True)


# jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

def main(num_processes: int, process_id: int, coordinator_address: str, plot_folder: str):
    warnings.warn("To run on CPU-only set 'JAX_PLATFORMS=cpu'.")
    print(f"Beginning multi-host initialisation at {datetime.datetime.now()}")
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id
    )
    print(f"Initialised at {datetime.datetime.now()}")

    # Must import only after jax.distributed.initialize to avoid issues with jax devices
    from dsa2000_cal.forward_models.streaming.process import process_start

    process_start(
        process_id=process_id,
        key=jax.random.PRNGKey(0),
        array_name="dsa2000_31b",
        full_stokes=True,
        plot_folder=plot_folder
    )


if __name__ == '__main__':
    port = get_free_port()
    main(1, 0, f"localhost:{port}", 'plots')
    # # Parse arguments
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num_processes", type=int, required=True, help="Number of processes")
    # parser.add_argument("--process_id", type=int, required=True, help="Process ID")
    # parser.add_argument("--coordinator_address", type=str, required=True,
    #                     help="Coordinator address, e.g. '10.0.0.1:1234")
    # parser.add_argument("--plot_folder", type=str, required=True, help="Plot folder")
    # args = parser.parse_args()
    # main(args.num_processes, args.process_id, args.coordinator_address, args.plot_folder)
