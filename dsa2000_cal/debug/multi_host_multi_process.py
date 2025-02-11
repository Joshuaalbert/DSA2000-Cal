import time

import jax


def main(num_processes: int, process_id: int, coordinator_address: str):
    print(f"initializing at {time.time()}")
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id
    )
    print(f"initialized at {time.time()}")
    print(jax.local_devices())
    print(jax.devices())


if __name__ == '__main__':
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, required=True, help="Number of processes")
    parser.add_argument("--process_id", type=int, required=True, help="Process ID")
    parser.add_argument("--coordinator_address", type=str, required=True,
                        help="Coordinator address, e.g. '10.0.0.1:1234")
    args = parser.parse_args()
    main(args.num_processes, args.process_id, args.coordinator_address)
