import os

os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import queue

from tqdm import tqdm
import astropy.units as au
import jax.random
from dsa2000_fm.bright_souces.evaluate_rms import simulate_rms
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import misc_registry


def run(result_idx, cpu, gpu, pointing):
    simulate_rms(
        cpu=cpu,
        gpu=gpu,
        result_num=result_idx,
        seed=0,
        save_folder='sky_loss_11Mar2025_full_survey_more_stats',
        array_name='dsa2000_optimal_v1',
        pointing=pointing,
        num_measure_points=256,
        sim_batch_size=128,
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


def main(node_idx: int, num_nodes: int):
    cpus = jax.devices("cpu")
    gpus = jax.devices("cuda")
    queues = [queue.Queue() for _ in gpus]

    fill_registries()
    survey_pointings = misc_registry.get_instance(misc_registry.get_match('survey_pointings'))
    pointings = survey_pointings.survey_pointings_v1()

    # fill queues with input args
    result_idx = 0
    node_id = 0
    for pointing in tqdm(pointings):
        if (node_id % num_nodes) == node_idx:
            q = queues[result_idx % len(gpus)]
            gpu = gpus[result_idx % len(gpus)]
            cpu = cpus[result_idx % len(cpus)]
            q.put((run, result_idx, cpu, gpu, pointing))
        result_idx += 1
        node_id += 1

    # now run the jobs in thread pool
    def worker(queue):
        while True:
            args = queue.get()
            if args is None:
                break
            f = args[0]
            args = args[1:]
            f(*args)

    # now run the jobs in thread pool, each job processes a queue
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for q in queues:
            executor.submit(worker, q)


if __name__ == '__main__':
    # add argparser
    import argparse

    parser = argparse.ArgumentParser(description='Run varying systematics')
    parser.add_argument('node_idx', type=int, help='Node index')
    parser.add_argument('num_nodes', type=int, help='Number of nodes')
    args = parser.parse_args()
    main(args.node_idx, args.num_nodes)
