import os

os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import jax.random
import astropy.units as au
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import misc_registry
from dsa2000_common.common.logging import dsa_logger
from dsa2000_fm.bright_souces.evaluate_rms import simulate_rms


def run(result_idx, cpu_idx, gpu_idx, pointing):
    cpus = jax.devices("cpu")
    gpus = jax.devices("cuda")
    cpu = cpus[cpu_idx]
    gpu = gpus[0]
    simulate_rms(
        cpu=cpu,
        gpu=gpu,
        result_num=result_idx,
        seed=0,
        save_folder='sky_loss_nominal_survey',
        array_name='dsa2000_optimal_v1',
        pointing=pointing,
        num_measure_points=256,
        image_batch_size=256,
        source_batch_size=256,
        angular_radius=1.75 * au.deg,
        prior_psf_sidelobe_peak=1e-3,
        bright_source_id='nvss_calibrators',
        pointing_offset_stddev=1 * au.arcmin,
        axial_focus_error_stddev=1 * au.mm,
        horizon_peak_astigmatism_stddev=1 * au.mm,
        turbulent=True,
        dawn=True,
        high_sun_spot=True,
        with_ionosphere=True,
        with_dish_effects=True,
        with_smearing=True
    )


def main(gpu_idx: int, node_idx: int, num_nodes: int):
    dsa_logger.info(f"Launching {gpu_idx} gpus")
    cpus = jax.devices("cpu")
    gpus = jax.devices("cuda")
    dsa_logger.info(f"Launching on GPU {gpu_idx}")

    fill_registries()
    survey_pointings = misc_registry.get_instance(misc_registry.get_match('survey_pointings'))
    pointings = survey_pointings.survey_pointings_v1()

    for result_idx in range(gpu_idx + len(gpus) * node_idx, len(pointings), len(gpus) * num_nodes):
        cpu_idx = result_idx % len(cpus)
        run(result_idx, cpu_idx, gpu_idx, pointings[result_idx])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Log resource usage to a file at a given cadence.")
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--node_idx", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)

    args = parser.parse_args()

    main(args.gpu_idx, args.node_idx, args.num_nodes)
