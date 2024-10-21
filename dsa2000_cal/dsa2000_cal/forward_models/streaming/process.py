import datetime
import os
from functools import partial
from typing import Dict, Tuple, NamedTuple, Any

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
from jax import numpy as jnp
from tomographic_kernel.frames import ENU

import dsa2000_cal.common.context as ctx
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.alert_utils import post_completed_forward_modelling_run
from dsa2000_cal.common.datetime_utils import current_utc
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.ray_utils import get_free_port
from dsa2000_cal.common.types import IntArray
from dsa2000_cal.forward_models.streaming.abc import AbstractCoreStep
from dsa2000_cal.forward_models.streaming.core.average import AverageStep
from dsa2000_cal.forward_models.streaming.core.calibrate import CalibrateStep
from dsa2000_cal.forward_models.streaming.core.create_calibration_model_data import CreateCalibrationModelDataStep
from dsa2000_cal.forward_models.streaming.core.create_model_data import CreateModelDataStep
from dsa2000_cal.forward_models.streaming.core.dd_average import DDAverageStep
from dsa2000_cal.forward_models.streaming.core.dd_predict import DDPredictStep
from dsa2000_cal.forward_models.streaming.core.flag import FlagStep
from dsa2000_cal.forward_models.streaming.core.image import ImageStep
from dsa2000_cal.forward_models.streaming.core.predict_and_sample import PredictAndSampleStep
from dsa2000_cal.forward_models.streaming.core.setup_observation import SetupObservationStep
from dsa2000_cal.forward_models.streaming.core.simulate_beam import SimulateBeamStep
from dsa2000_cal.forward_models.streaming.core.simulate_dish import SimulateDishStep
from dsa2000_cal.forward_models.streaming.core.simulate_ionosphere import SimulateIonosphereStep
from dsa2000_cal.forward_models.streaming.core.subtract import SubtractStep

# Set num jax devices to number of CPUs
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"
jax.config.update('jax_threefry_partitionable', True)


# jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

def get_process_local_params(process_id: int, array_name: str):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))
    chan_per_process = 40
    chan_slice = slice(process_id * chan_per_process, (process_id + 1) * chan_per_process)
    channel_width = array.channel_width()
    freqs = array.get_channels()[chan_slice]

    ref_time = at.Time("2021-01-01T00:00:00", scale="utc")
    solution_interval = 6 * au.s
    validity_interval = 12 * au.s
    observation_duration = 624 * au.s
    integration_interval = array.integration_time()
    num_timesteps = int(observation_duration / integration_interval)
    obstimes = ref_time + np.arange(num_timesteps) * integration_interval
    antennas = array.get_antennas()
    array_location = array.get_array_location()
    phase_center = pointing = ENU(0, 0, 1, obstime=ref_time, location=array_location).transform_to(ac.ICRS)

    dish_effects_params = array.get_dish_effect_params()
    return freqs, channel_width, antennas, array_location, pointing, phase_center, ref_time, obstimes, solution_interval, validity_interval, integration_interval, dish_effects_params


def process_start(process_id: int,
                  key,
                  array_name: str,
                  full_stokes: bool
                  ):
    (
        freqs, channel_width, antennas, array_location, pointings, phase_center, ref_time, obstimes, solution_interval,
        validity_interval, integration_interval, dish_effects_params
    ) = get_process_local_params(
        process_id=process_id,
        array_name=array_name
    )

    # Check the units of each process parameter
    if not freqs.unit.is_equivalent("Hz"):
        raise ValueError("freqs must be in Hz")
    if not integration_interval.unit.is_equivalent("s"):
        raise ValueError("integration_interval must be in seconds")
    if not channel_width.unit.is_equivalent("Hz"):
        raise ValueError("channel_width must be in Hz")
    if len(obstimes) % solution_interval != 0:
        raise ValueError("obstimes must be a multiple of solution_interval")

    total_duration = len(obstimes) * integration_interval
    total_bandwidth = len(freqs) * channel_width
    num_steps = len(obstimes) / solution_interval
    print(
        f"Streaming Forward Model:\n"
        f"- {num_steps} steps\n"
        f"- observation duration: {total_duration}\n"
        f"- bandwidth: {total_bandwidth}"
    )

    setup_observation_step = SetupObservationStep(
        freqs=freqs,
        antennas=antennas,
        array_location=array_location,
        phase_center=phase_center,
        obstimes=obstimes,
        ref_time=ref_time,
        pointings=pointings
    )

    simulate_beam_step = SimulateBeamStep(
        array_name=array_name,
        full_stokes=full_stokes,
        model_times=at.Time([obstimes[0], obstimes[-1]]),
        freqs=freqs
    )

    # TODO: complete this step
    simulate_dish_step = SimulateDishStep(
        static_beam=True,
        dish_effects_params=dish_effects_params
    )

    simulate_ionosphere_step = SimulateIonosphereStep()

    create_model_data_step = CreateModelDataStep()

    predict_and_sample_step = PredictAndSampleStep()

    flag_step = FlagStep()

    average_step = AverageStep()

    create_calibration_model_data_step = CreateCalibrationModelDataStep()

    dd_predict_step = DDPredictStep()

    dd_average_step = DDAverageStep()

    calibrate_step = CalibrateStep()

    subtract_step = SubtractStep()

    image_step = ImageStep()

    # DAG
    dag: Dict[AbstractCoreStep, Tuple[AbstractCoreStep, ...]] = dict()  # step -> primals
    dag[setup_observation_step] = ()
    dag[simulate_beam_step] = ()

    # dag[simulate_dish_step] = (simulate_beam_step,)
    # dag[simulate_ionosphere_step] = (simulate_dish_step,)
    # dag[create_model_data_step] = (simulate_dish_step, simulate_ionosphere_step)
    # dag[predict_and_sample_step] = (create_model_data_step,)
    # dag[flag_step] = (predict_and_sample_step,)
    # dag[average_step] = (flag_step,)
    # dag[create_calibration_model_data_step] = (simulate_beam_step,)
    # dag[dd_predict_step] = (create_calibration_model_data_step,)
    # dag[dd_average_step] = (dd_predict_step,)
    # dag[calibrate_step] = (average_step, dd_average_step)  # <--- return solutions
    # dag[subtract_step] = (calibrate_step, dd_predict_step)
    # dag[image_step] = (subtract_step, flag_step, simulate_beam_step)  # <-- return state

    def exectute_dag() -> Dict[AbstractCoreStep, Any]:
        # Traverse DAG instead of hardcoding
        step_outputs = dict()
        step_keeps = dict()
        done = set()
        step_idx = 0
        while len(done) < len(dag):
            step_key = ctx.next_rng_key()
            step_idx += 1
            # Get the first step that has all its primals done
            for step, primals in dag.items():
                if step in done:
                    continue
                if all(primal in done for primal in primals):
                    break
            else:
                raise RuntimeError("Cyclic dependency in DAG")
            # Run the step
            print(
                f"{step_idx}. {step.name} :: ({', '.join(primal.output_name for primal in primals)}) "
                f"-> ({step.state_name}, {step.output_name})"
            )
            step_primals = tuple(step_outputs[primal] for primal in primals)
            step_output, step_keep = step.step(primals=step_primals)
            step_outputs[step] = step_output
            step_keeps[step] = step_keep
            done.add(step)
        return step_keeps

    exectute_dag_transformed = ctx.transform_with_state(exectute_dag)

    @partial(jax.jit, static_argnames=['num_steps'], donate_argnums=(0, 1, 2))
    def run_process(key, params: ctx.ImmutableParams, init_states: ctx.MutableParams, num_steps: int):

        class Carry(NamedTuple):
            states: ctx.MutableParams

        class XType(NamedTuple):
            step_idx: IntArray
            key: jax.Array

        def body(carry: Carry, x: XType):
            apply_return = exectute_dag_transformed.apply(params, carry.states, x.key)
            return Carry(states=apply_return.states), apply_return.fn_val

        keys = jax.random.split(key, num_steps)
        carry, keep = jax.lax.scan(
            body,
            Carry(states=init_states),
            XType(step_idx=jnp.arange(num_steps), key=keys)
        )
        return keep

    # Run the process
    hostname = os.uname().nodename
    print(f"Running on {hostname}")
    start_time = current_utc()
    run_key, init_key = jax.random.split(key, 2)
    print("Initialising...")
    init = block_until_ready(exectute_dag_transformed.init(init_key))
    print("Running...")
    final_keep = block_until_ready(
        run_process(init.params, init.states, run_key)
    )
    end_time = current_utc()
    run_time = (end_time - start_time).total_seconds()
    print(f"Run time: {run_time:.2f}s")

    # Tell Slack we're done
    post_completed_forward_modelling_run(
        run_dir=os.getcwd(),
        start_time=start_time,
        duration=end_time - start_time
    )


def main(num_processes: int, process_id: int, coordinator_address: str):
    print(f"Beginning multi-host initialisation at {datetime.datetime.now()}")
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id
    )
    print(f"Initialised at {datetime.datetime.now()}")
    process_start(
        process_id=process_id,
        key=jax.random.PRNGKey(0),
        array_name="DSA2000W",
        full_stokes=True,

    )


if __name__ == '__main__':
    port = get_free_port()
    main(1, 0, f"localhost:{port}")
    # # Parse arguments
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num_processes", type=int, required=True, help="Number of processes")
    # parser.add_argument("--process_id", type=int, required=True, help="Process ID")
    # parser.add_argument("--coordinator_address", type=str, required=True,
    #                     help="Coordinator address, e.g. '10.0.0.1:1234")
    # args = parser.parse_args()
    # main(args.num_processes, args.process_id, args.coordinator_address)
