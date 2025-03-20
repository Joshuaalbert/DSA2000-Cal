import os

os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import queue
from dsa2000_fm.bright_souces.evaluate_rms import simulate_rms
import astropy.coordinates as ac
import astropy.units as au
import jax.random


def run(result_idx, cpu_idx, gpu_idx, pointing_offset_stddev, axial_focus_error_stddev,
        horizon_peak_astigmatism_stddev, with_smearing):
    cpus = jax.devices("cpu")
    gpus = jax.devices("cuda")
    cpu = cpus[cpu_idx]
    gpu = gpus[0]
    simulate_rms(
        cpu=cpu,
        gpu=gpu,
        result_num=result_idx,
        seed=0,
        save_folder='sky_loss_varying_systematics',
        array_name='dsa2000_optimal_v1',
        pointing=ac.ICRS(0 * au.deg, 0 * au.deg),
        num_measure_points=256,
        image_batch_size=64,
        source_batch_size=64,
        angular_radius=1.75 * au.deg,
        prior_psf_sidelobe_peak=1e-3,
        bright_source_id='nvss_calibrators',
        pointing_offset_stddev=pointing_offset_stddev,
        axial_focus_error_stddev=axial_focus_error_stddev,
        horizon_peak_astigmatism_stddev=horizon_peak_astigmatism_stddev,
        turbulent=True,
        dawn=True,
        high_sun_spot=True,
        with_ionosphere=True,
        with_dish_effects=True,
        with_smearing=with_smearing
    )


def main(gpu_idx: int, node_idx: int, num_nodes: int):
    cpus = jax.devices("cpu")
    gpus = jax.devices("cuda")

    # fill queues with input args
    args = []
    for pointing_offset_stddev in [0, 1, 2, 4] * au.arcmin:
        for axial_focus_error_stddev in [0, 3, 5] * au.mm:
            for horizon_peak_astigmatism_stddev in [0, 1, 2, 4] * au.mm:
                for with_smearing in [True, False]:
                    args.append((pointing_offset_stddev,
                                 axial_focus_error_stddev, horizon_peak_astigmatism_stddev, with_smearing))

    for result_idx in range(gpu_idx + len(gpus) * node_idx, len(args), len(gpus) * num_nodes):
        cpu_idx = result_idx % len(cpus)
        run(result_idx, cpu_idx, gpu_idx, *args[result_idx])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Log resource usage to a file at a given cadence.")
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--node_idx", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)

    args = parser.parse_args()

    main(args.gpu_idx, args.node_idx, args.num_nodes)
