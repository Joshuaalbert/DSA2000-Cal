import logging
import logging
import os
from uuid import uuid4

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from tomographic_kernel.frames import ENU

from dsa2000_cal.actors.namespace import NAMESPACE
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.delay_models.base_far_field_delay_engine import build_far_field_delay_engine
from dsa2000_cal.delay_models.base_near_field_delay_engine import build_near_field_delay_engine
from dsa2000_cal.forward_models.streaming.distributed.aggregator import Aggregator, AggregatorParams
from dsa2000_cal.forward_models.streaming.distributed.calibrator import CalibrationSolutionCache, Calibrator, \
    CalibrationSolutionCacheParams
from dsa2000_cal.forward_models.streaming.distributed.common import ChunkParams, ForwardModellingRunParams, ImageParams
from dsa2000_cal.forward_models.streaming.distributed.data_streamer import DataStreamerParams, DataStreamer
from dsa2000_cal.forward_models.streaming.distributed.gridder import Gridder
from dsa2000_cal.forward_models.streaming.distributed.model_predictor import ModelPredictor, ModelPredictorParams
from dsa2000_cal.forward_models.streaming.distributed.system_gain_simulator import SystemGainSimulator, \
    SystemGainSimulatorParams
from dsa2000_cal.geodesics.base_geodesic_model import build_geodesic_model
from dsa2000_cal.imaging.utils import get_image_parameters
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMeta

logger = logging.getLogger('ray')


def build_run_params(array_name: str, with_autocorr: bool, field_of_view: au.Quantity | None,
                     oversample_factor: float, full_stokes: bool, num_cal_facets: int,
                     root_folder: str, run_name: str) -> ForwardModellingRunParams:
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))

    antennas = array.get_antennas()
    array_location = array.get_array_location()

    num_antennas = len(antennas)
    if with_autocorr:
        num_baselines = (num_antennas * (num_antennas + 1)) // 2
    else:
        num_baselines = (num_antennas * (num_antennas - 1)) // 2

    freqs = array.get_channels()[:40]

    num_channels = len(freqs)
    num_sub_bands_per_image = 1
    num_freqs_per_sol_int = 40

    channel_width = array.get_channel_width()
    solution_interval_freq = num_freqs_per_sol_int * channel_width
    sub_band_interval = channel_width * num_channels / num_sub_bands_per_image
    num_sol_ints_per_sub_band = int(sub_band_interval / solution_interval_freq)

    integration_interval = array.integration_time()
    solution_interval = 6 * au.s
    validity_interval = 12 * au.s
    observation_duration = 624 * au.s

    num_times_per_sol_int = int(solution_interval / integration_interval)
    num_sol_ints_per_accumlate = int(validity_interval / solution_interval)
    num_accumulates_per_image = int(observation_duration / validity_interval)
    num_integrations = int(observation_duration / integration_interval)

    ref_time = at.Time("2021-01-01T00:00:00", scale="utc")
    num_timesteps = int(observation_duration / integration_interval)
    obstimes = ref_time + np.arange(num_timesteps) * integration_interval

    phase_center = pointing = ENU(0, 0, 1, obstime=ref_time, location=array_location).transform_to(ac.ICRS())

    dish_effects_params = array.get_dish_effect_params()

    system_equivalent_flux_density = array.get_system_equivalent_flux_density()

    chunk_params = ChunkParams(
        num_channels=num_channels,
        num_integrations=num_integrations,
        num_baselines=num_baselines,
        num_freqs_per_sol_int=num_freqs_per_sol_int,
        num_sol_ints_per_sub_band=num_sol_ints_per_sub_band,
        num_sub_bands_per_image=num_sub_bands_per_image,
        num_times_per_sol_int=num_times_per_sol_int,
        num_sol_ints_per_accumlate=num_sol_ints_per_accumlate,
        num_accumulates_per_image=num_accumulates_per_image,
        num_model_times_per_solution_interval=1,
        num_model_freqs_per_solution_interval=1
    )

    meta = MeasurementSetMeta(
        array_name=array_name,
        array_location=array_location,
        phase_center=phase_center,
        pointings=pointing,
        channel_width=array.get_channel_width(),
        integration_time=integration_interval,
        coherencies=('XX', 'XY', 'YX', 'YY') if full_stokes else ('I',),
        times=obstimes,
        ref_time=ref_time,
        freqs=freqs,
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=with_autocorr,
        mount_types='ALT-AZ',
        system_equivalent_flux_density=system_equivalent_flux_density,
        convention='physical',
        static_beam=True
    )

    num_pixel, dl, dm, center_l, center_m = get_image_parameters(
        meta=meta,
        field_of_view=field_of_view,
        oversample_factor=oversample_factor
    )

    image_params = ImageParams(
        l0=center_l,
        m0=center_m,
        dl=dl,
        dm=dm,
        num_l=num_pixel,
        num_m=num_pixel
    )

    plot_folder = os.path.join(root_folder, run_name)
    os.makedirs(plot_folder, exist_ok=True)

    return ForwardModellingRunParams(
        ms_meta=meta,
        dish_effects_params=dish_effects_params,
        chunk_params=chunk_params,
        image_params=image_params,
        full_stokes=full_stokes,
        num_cal_facets=num_cal_facets,
        plot_folder=plot_folder,
        run_name=run_name
    )


def main(array_name: str, with_autocorr: bool, field_of_view: float | None,
         oversample_factor: float, full_stokes: bool, num_cal_facets: int,
         root_folder: str, run_name: str):
    # Connect to Ray.
    ray.init(address="auto", namespace=NAMESPACE)  # ray:{os.environ['RAY_REDIS_PORT']}

    field_of_view = field_of_view * au.deg if field_of_view is not None else None

    run_params = build_run_params(array_name, with_autocorr, field_of_view, oversample_factor,
                                  full_stokes, num_cal_facets, root_folder, run_name)

    geodesic_model = build_geodesic_model(
        antennas=run_params.ms_meta.antennas,
        array_location=run_params.ms_meta.array_location,
        phase_center=run_params.ms_meta.phase_center,
        obstimes=run_params.ms_meta.times,
        ref_time=run_params.ms_meta.ref_time,
        pointings=run_params.ms_meta.pointings
    )

    far_field_delay_engine = build_far_field_delay_engine(
        antennas=run_params.ms_meta.antennas,
        start_time=run_params.ms_meta.times[0],
        end_time=run_params.ms_meta.times[-1],
        ref_time=run_params.ms_meta.ref_time,
        phase_center=run_params.ms_meta.phase_center
    )

    near_field_delay_engine = build_near_field_delay_engine(
        antennas=run_params.ms_meta.antennas,
        start_time=run_params.ms_meta.times[0],
        end_time=run_params.ms_meta.times[-1],
        ref_time=run_params.ms_meta.ref_time
    )

    system_gain_simulator_params = SystemGainSimulatorParams(
        geodesic_model=geodesic_model,
        init_key=jax.random.PRNGKey(0),
    )

    data_streamer_params = DataStreamerParams(
        sky_model_id='cas_a',
        bright_sky_model_id='cas_a',
        num_facets_per_side=2,
        crop_box_size=None,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        geodesic_model=geodesic_model
    )

    predict_params = ModelPredictorParams(
        sky_model_id='cas_a',
        num_facets_per_side=2,
        crop_box_size=None,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        geodesic_model=geodesic_model
    )

    def compute_system_gain_simulator_options(run_params: ForwardModellingRunParams):
        # memory is 2 * n * n * A * num_coh * itemsize(gains)
        num_coh = 4 if run_params.full_stokes else 1
        n = 257
        A = len(run_params.ms_meta.antennas)
        itemsize_gains = np.dtype(np.complex64).itemsize
        memory = 2 * n * n * A * num_coh * itemsize_gains
        return {
            "num_cpus": 1,
            "num_gpus": 0,
            'memory': 1.1 * memory
        }

    system_gain_simulator = SystemGainSimulator.options(
        max_queued_requests=-1,  # no queue limit
        max_ongoing_requests=1,  # only one request at a time
        num_replicas=1,
        ray_actor_options=compute_system_gain_simulator_options(run_params)
    ).bind(run_params, system_gain_simulator_params)

    def compute_data_streamer_options(run_params: ForwardModellingRunParams):
        # memory is 2 * B * num_coh * (itemsize(vis) + itemsize(weights) + itemsize(flags))
        num_coh = 4 if run_params.full_stokes else 1
        B = run_params.chunk_params.num_baselines
        itemsize_vis = np.dtype(np.complex64).itemsize
        itemsize_weights = np.dtype(np.float16).itemsize
        itemsize_flags = np.dtype(np.bool_).itemsize
        memory = 2 * B * num_coh * (itemsize_vis + itemsize_weights + itemsize_flags)
        return {
            "num_cpus": 1,
            "num_gpus": 0,
            'memory': 1.1 * memory
        }

    data_streamer = DataStreamer.options(
        max_queued_requests=-1,  # no queue limit
        max_ongoing_requests=1,  # only one request at a time
        num_replicas=1,
        ray_actor_options=compute_data_streamer_options(run_params)
    ).bind(run_params, data_streamer_params, system_gain_simulator)

    def compute_model_predictor_options(run_params: ForwardModellingRunParams):
        # memory is 2 * D * B * num_coh * itemsize(vis)
        # 2 is for buffer in predict
        num_coh = 4 if run_params.full_stokes else 1
        D = run_params.num_cal_facets
        B = run_params.chunk_params.num_baselines
        itemsize_vis = np.dtype(np.complex64).itemsize
        memory = 2 * D * B * num_coh * itemsize_vis
        return {
            "num_cpus": 1,
            "num_gpus": 0,
            'memory': 1.1 * memory
        }

    model_predictor = ModelPredictor.options(
        max_queued_requests=-1,  # no queue limit
        max_ongoing_requests=1,  # only one request at a time
        num_replicas=1,
        ray_actor_options=compute_model_predictor_options(run_params)
    ).bind(run_params, predict_params)

    def compuate_calibration_solution_cache_options(run_params: ForwardModellingRunParams):
        # memory os Tm * A * Cm * num_coh * itemsize(gains)
        num_coh = 4 if run_params.full_stokes else 1
        Tm = run_params.chunk_params.num_model_times_per_solution_interval
        Cm = run_params.chunk_params.num_model_freqs_per_solution_interval
        A = len(run_params.ms_meta.antennas)
        itemsize_gains = np.dtype(np.complex64).itemsize
        memory = Tm * A * Cm * num_coh * itemsize_gains
        return {
            'memory': 1.1 * memory
        }

    calibration_soluation_cache = CalibrationSolutionCache(params=CalibrationSolutionCacheParams(),
                                                           **compuate_calibration_solution_cache_options(run_params))

    def compute_calibrator_options(run_params: ForwardModellingRunParams):
        # memory for inputs from stream:
        # Ts * B * Cs * num_coh * (itemsize(vis) + itemsize(weights) + itemsize(flags))

        # memory for averaged data:
        # Tm * B * Cm * num_coh * (itemsize(vis) + itemsize(weights) + itemsize(flags))

        # memory for model:
        # D * Tm * B * Cm * num_coh * itemsize(vis)

        # memory for solution:
        # D * Tm * A * Cm * num_coh * itemsize(gains)

        num_coh = 4 if run_params.full_stokes else 1
        Ts = run_params.chunk_params.num_times_per_sol_int
        B = run_params.chunk_params.num_baselines
        Cs = run_params.chunk_params.num_freqs_per_sol_int
        D = run_params.num_cal_facets
        Tm = 1
        Cm = 1
        A = len(run_params.ms_meta.antennas)
        itemsize_vis = np.dtype(np.complex64).itemsize
        itemsize_weights = np.dtype(np.float16).itemsize
        itemsize_flags = np.dtype(np.bool_).itemsize
        itemsize_gains = np.dtype(np.complex64).itemsize
        memory = Ts * B * Cs * num_coh * (itemsize_vis + itemsize_weights + itemsize_flags) + \
                 Tm * B * Cm * num_coh * (itemsize_vis + itemsize_weights + itemsize_flags) + \
                 D * Tm * B * Cm * num_coh * itemsize_vis + \
                 D * Tm * A * Cm * num_coh * itemsize_gains
        # TODO: flip to GPU once we standardise the GPU's per device. This would need different resources for varying GPU's
        return {
            "num_cpus": 1,
            "num_gpus": 0,
            'memory': 1.1 * memory
        }

    calibrator = Calibrator.options(
        max_queued_requests=-1,  # no queue limit
        max_ongoing_requests=1,  # only one request at a time
        num_replicas=1,
        ray_actor_options=compute_calibrator_options(run_params)
    ).bind(run_params, data_streamer, model_predictor, calibration_soluation_cache)

    def compute_gridder_options(run_params: ForwardModellingRunParams):
        # Memory is 2 * num_pix^2 * num_coh * itemsize(image)
        num_coh = 4 if run_params.full_stokes else 1
        num_pix_l = run_params.image_params.num_l
        num_pix_m = run_params.image_params.num_m
        # image is f64
        itemsize_image = np.dtype(np.float64).itemsize
        memory = 2 * num_pix_l * num_pix_m * num_coh * itemsize_image
        return {
            "num_cpus": run_params.chunk_params.num_freqs_per_sol_int,  # 1 thread per channel * (num coh)
            "num_gpus": 0,  # Doesn't use GPU
            'memory': 1.1 * memory
        }

    gridder = Gridder.options(
        max_queued_requests=-1,  # no queue limit
        max_ongoing_requests=1,  # only one request at a time
        num_replicas=1,
        ray_actor_options=compute_gridder_options(run_params)
    ).bind(run_params, calibrator)

    gridder_handle: DeploymentHandle = serve.run(gridder)

    # TODO: merge into Caller and run simultaneously over all subbands
    # TODO: remove beam before saving image

    def compute_aggregator_options(run_params: ForwardModellingRunParams):
        # memory is 2 * num_pix^2 * num_coh * itemsize(image)
        num_coh = 4 if run_params.full_stokes else 1
        num_pix_l = run_params.image_params.num_l
        num_pix_m = run_params.image_params.num_m
        # image is f64
        itemsize_image = np.dtype(np.float64).itemsize
        memory = 2 * num_pix_l * num_pix_m * num_coh * itemsize_image
        return {
            "num_cpus": 0,  # Doesn't use CPU
            "num_gpus": 0,  # Doesn't use GPU
            'memory': 1.1 * memory
        }

    aggregator = Aggregator(
        worker_id=str(uuid4()),
        params=AggregatorParams(
            sol_int_freq_idxs=[0],
            fm_run_params=run_params,
            gridder=gridder_handle,
            image_suffix='sb0',
        ),
        **compute_aggregator_options(run_params)
    )

    key = jax.random.PRNGKey(0)
    for sol_int_time_idx in range(run_params.chunk_params.num_times_per_sol_int):
        run_key, key = jax.random.split(key, 2)
        save_to_disk = True
        aggregator_response = aggregator(key, sol_int_time_idx, save_to_disk)
        logger.info(
            f"Image {sol_int_time_idx} done. Saved to:\n"
            f"Image: {aggregator_response.image_path}\n"
            f"PSF: {aggregator_response.psf_path}"
        )


if __name__ == '__main__':
    # parse args
    import argparse


    # bool parser
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    def none_or_float(value):
        if value.lower() == 'none':
            return None
        return float(value)


    parser = argparse.ArgumentParser(description='Run the forward modelling pipeline.')
    parser.add_argument('--array_name', type=str, help='Name of the array to use.')
    parser.add_argument('--field_of_view', default=None, type=none_or_float, help='Field of view in degrees.')
    parser.add_argument('--oversample_factor', default=5., type=float, help='Oversample factor for the image.')
    parser.add_argument('--full_stokes', default=True, type=str2bool, help='Use full stokes.')
    parser.add_argument('--num_cal_facets', default=1, type=int, help='Number of calibration facets.')
    parser.add_argument('--root_folder', type=str, help='Root folder to save output plots.')
    parser.add_argument('--run_name', type=str, help='Name of the run.')
    args = parser.parse_args()

    main(
        array_name=args.array_name, with_autocorr=False, field_of_view=args.field_of_view,
        oversample_factor=args.oversample_factor, full_stokes=args.full_stokes, num_cal_facets=args.num_cal_facets,
        root_folder=args.root_folder, run_name=args.run_name
    )
