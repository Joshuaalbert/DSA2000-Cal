import asyncio
import datetime
import logging
import os
from uuid import uuid4

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import ray
from tomographic_kernel.frames import ENU

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.alert_utils import post_completed_forward_modelling_run
from dsa2000_cal.delay_models.base_far_field_delay_engine import build_far_field_delay_engine
from dsa2000_cal.delay_models.base_near_field_delay_engine import build_near_field_delay_engine
from dsa2000_cal.geodesics.base_geodesic_model import build_geodesic_model
from dsa2000_cal.imaging.utils import get_image_parameters
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMeta
from dsa2000_fm.forward_models.streaming.distributed.aggregator import Aggregator, AggregatorParams, \
    compute_aggregator_options
from dsa2000_fm.forward_models.streaming.distributed.calibration_solution_cache import CalibrationSolutionCacheParams, \
    CalibrationSolutionCache, compute_calibration_solution_cache_options
from dsa2000_fm.forward_models.streaming.distributed.calibrator import Calibrator, \
    compute_calibrator_options, CalibratorParams
from dsa2000_fm.forward_models.streaming.distributed.common import ChunkParams, ForwardModellingRunParams, ImageParams
from dsa2000_fm.forward_models.streaming.distributed.data_streamer import DataStreamerParams, DataStreamer, \
    compute_data_streamer_options
from dsa2000_fm.forward_models.streaming.distributed.degridding_predictor import DegriddingPredictor, \
    compute_degridding_predictor_options
from dsa2000_fm.forward_models.streaming.distributed.dft_predictor import DFTPredictor, compute_dft_predictor_options
from dsa2000_fm.forward_models.streaming.distributed.gridder import Gridder, compute_gridder_options
from dsa2000_fm.forward_models.streaming.distributed.model_predictor import ModelPredictor, ModelPredictorParams, \
    compute_model_predictor_options
from dsa2000_fm.forward_models.streaming.distributed.supervisor import create_supervisor
from dsa2000_fm.forward_models.streaming.distributed.system_gain_simulator import SystemGainSimulator, \
    SystemGainSimulatorParams, compute_system_gain_simulator_options
from dsa2000_rcp.actors.namespace import NAMESPACE

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

    # 10000/10 = 1000, 1000/40 = 25
    num_sub_bands = 1
    num_freqs_per_sol_int = 1

    central_freq_idx = len(array.get_channels()) // 2

    freqs = array.get_channels()[central_freq_idx:central_freq_idx + 1]

    num_channels = len(freqs)

    channel_width = 40 * array.get_channel_width()
    solution_interval_freq = num_freqs_per_sol_int * channel_width
    sub_band_interval = channel_width * num_channels / num_sub_bands
    num_sol_ints_per_sub_band = int(sub_band_interval / solution_interval_freq)

    integration_interval = 4 * array.get_integration_time()
    solution_interval = 6 * au.s
    observation_duration = 624 * au.s

    num_times_per_sol_int = int(solution_interval / integration_interval)
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
        num_sub_bands=num_sub_bands,
        num_times_per_sol_int=num_times_per_sol_int,
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
    ray.init(address="auto", namespace=NAMESPACE)

    field_of_view = field_of_view * au.deg if field_of_view is not None else None

    run_params = build_run_params(array_name, with_autocorr, field_of_view, oversample_factor,
                                  full_stokes, num_cal_facets, root_folder, run_name)
    with open("run_params.json", "w") as f:
        f.write(run_params.json(indent=2))
    print(f"Stored run params in run_params.json")

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
        apply_effects=True,  # Skip for now
        simulate_ionosphere=False  # Skip for now
    )

    # TODO: add list of sky model for each sub-type
    data_streamer_params = DataStreamerParams(
        sky_model_id='trecs',
        bright_sky_model_id='mock_calibrators',
        num_facets_per_side=2,
        crop_box_size=3 * au.deg,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        geodesic_model=geodesic_model
    )

    predict_params = ModelPredictorParams(
        sky_model_id='mock_calibrators',
        background_sky_model_id='trecs',
        num_facets_per_side=2,
        crop_box_size=3 * au.deg,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        geodesic_model=geodesic_model
    )

    calibrator_params = CalibratorParams(
        do_calibration=True
    )

    asyncio.run(run_forward_model(run_params, data_streamer_params, predict_params, system_gain_simulator_params,
                                  calibrator_params))


async def run_forward_model(run_params, data_streamer_params, predict_params, system_gain_simulator_params,
                            calibrator_params):
    dft_predictor_remote = DFTPredictor.options(
        **compute_dft_predictor_options(run_params)
    )
    dft_predictor = create_supervisor(
        dft_predictor_remote, 'dft_predictor', 1,
        run_params
    )

    degridding_predictor_remote = DegriddingPredictor.options(
        **compute_degridding_predictor_options(run_params)
    )
    degridding_predictor = create_supervisor(
        degridding_predictor_remote, 'degridding_predictor', 1,
        run_params
    )

    system_gain_simulator_remote = SystemGainSimulator.options(
        **compute_system_gain_simulator_options(run_params)
    )
    system_gain_simulator = create_supervisor(
        system_gain_simulator_remote, 'system_gain_simulator', 1,
        run_params, system_gain_simulator_params
    )

    model_predictor_remote = ModelPredictor.options(
        **compute_model_predictor_options(run_params)
    )
    model_predictor = create_supervisor(
        model_predictor_remote, 'model_predictor', 1,
        run_params, predict_params, dft_predictor, degridding_predictor
    )

    data_streamer_remote = DataStreamer.options(
        **compute_data_streamer_options(run_params)
    )
    data_streamer = create_supervisor(
        data_streamer_remote, 'data_streamer', 1,
        run_params, data_streamer_params, system_gain_simulator, dft_predictor, degridding_predictor
    )

    calibration_soluation_cache = CalibrationSolutionCache(
        params=CalibrationSolutionCacheParams(),
        **compute_calibration_solution_cache_options(run_params)
    )
    calibrator_remote = Calibrator.options(
        **compute_calibrator_options(run_params)
    )
    calibrator = create_supervisor(
        calibrator_remote, 'calibrator', 1,
        run_params, calibrator_params, data_streamer, model_predictor, calibration_soluation_cache
    )
    gridder_remote = Gridder.options(
        **compute_gridder_options(run_params)
    )
    gridder = create_supervisor(
        gridder_remote, 'gridder', 1,
        run_params, calibrator
    )

    # This is the Caller
    async def run_aggregator(key, sub_band_idx: int):
        sol_int_freq_idxs = (
                np.arange(run_params.chunk_params.num_sol_ints_per_sub_band)
                + sub_band_idx * run_params.chunk_params.num_sol_ints_per_sub_band
        )
        aggregator = Aggregator(
            worker_id=str(uuid4()),
            params=AggregatorParams(
                sol_int_freq_idxs=sol_int_freq_idxs.tolist(),
                fm_run_params=run_params,
                gridder=gridder,
                image_suffix=f'SB{sub_band_idx:01d}',
            ),
            **compute_aggregator_options(run_params)
        )
        sol_int_time_idxs = list(range(run_params.chunk_params.num_sol_ints_time))
        gen = aggregator.stream(key, sol_int_time_idxs, save_to_disk=True)
        async for aggregator_response_ref in gen:
            aggregator_response = await aggregator_response_ref
            logger.info(
                f"Image saved to:\n"
                f"Image: {aggregator_response.image_path}\n"
                f"PSF: {aggregator_response.psf_path}"
            )

    async def run_all(key):
        start_time = datetime.datetime.now()
        tasks = []
        for sub_band_idx in range(run_params.chunk_params.num_sub_bands):
            aggregator_key, key = jax.random.split(key, 2)
            tasks.append(asyncio.create_task(run_aggregator(aggregator_key, sub_band_idx)))
        await asyncio.gather(*tasks)
        end_time = datetime.datetime.now()
        post_completed_forward_modelling_run(
            run_dir=os.getcwd(),
            start_time=start_time,
            duration=end_time - start_time
        )

    # Submit with PRNG key
    await run_all(jax.random.PRNGKey(0))


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
