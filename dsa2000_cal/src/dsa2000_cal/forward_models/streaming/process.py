import os
from functools import partial
from typing import Dict, Tuple, NamedTuple, Any

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
from tomographic_kernel.frames import ENU

import dsa2000_cal.common.context as ctx
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.alert_utils import post_completed_forward_modelling_run
from dsa2000_cal.common.datetime_utils import current_utc
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.ray_utils import MemoryLogger
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.forward_models.streaming.abc import AbstractCoreStep
from dsa2000_cal.forward_models.streaming.core.setup_observation import SetupObservationStep
from dsa2000_cal.forward_models.streaming.core.simulate_beam import SimulateBeamStep
from dsa2000_cal.forward_models.streaming.core.simulate_dish import SimulateDishStep
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsParams


class ProcessLocalParams(SerialisableBaseModel):
    freqs: au.Quantity
    channel_width: au.Quantity
    antennas: ac.EarthLocation
    array_location: ac.EarthLocation
    pointings: ac.ICRS
    phase_center: ac.ICRS
    ref_time: at.Time
    obstimes: at.Time
    solution_interval: au.Quantity
    validity_interval: au.Quantity
    integration_interval: au.Quantity
    dish_effects_params: DishEffectsParams


def get_process_local_params(process_id: int, array_name: str) -> ProcessLocalParams:
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))
    chan_per_process = 40
    chan_slice = slice(process_id * chan_per_process, (process_id + 1) * chan_per_process)
    channel_width = array.get_channel_width()
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
    phase_center = pointing = ENU(0, 0, 1, obstime=ref_time, location=array_location).transform_to(ac.ICRS())

    dish_effects_params = array.get_dish_effect_params()
    return ProcessLocalParams(
        freqs=freqs,
        channel_width=channel_width,
        antennas=antennas,
        array_location=array_location,
        pointings=pointing,
        phase_center=phase_center,
        ref_time=ref_time,
        obstimes=obstimes,
        solution_interval=solution_interval,
        validity_interval=validity_interval,
        integration_interval=integration_interval,
        dish_effects_params=dish_effects_params
    )


def build_process_scan(execute_dag_transformed: ctx.TransformedWithStateFn):
    @partial(jax.jit, static_argnames=['num_steps'], donate_argnums=(0, 1, 2))
    def run_process(key, params: ctx.ImmutableParams, init_states: ctx.MutableParams, num_steps: int):
        class Carry(NamedTuple):
            states: ctx.MutableParams

        class XType(NamedTuple):
            key: jax.Array

        def body(carry: Carry, x: XType):
            apply_return = execute_dag_transformed.apply(params, carry.states, x.key)
            print(apply_return)
            return Carry(states=apply_return.states), apply_return.fn_val

        carry, keep = jax.lax.scan(
            body,
            Carry(states=init_states),
            XType(key=jax.random.split(key, num_steps))
        )
        return keep

    return run_process


def build_process_for(execute_dag_transformed: ctx.TransformedWithStateFn):
    def run_process(key, params: ctx.ImmutableParams, init_states: ctx.MutableParams, num_steps: int):
        class Carry(NamedTuple):
            states: ctx.MutableParams

        class XType(NamedTuple):
            key: jax.Array

        def body(carry: Carry, x: XType):
            apply_return = execute_dag_transformed.apply(params, carry.states, x.key)
            return Carry(states=apply_return.states), apply_return.fn_val

        carry = Carry(states=init_states)
        xs = XType(key=jax.random.split(key, num_steps))
        keep = []
        for i in range(num_steps):
            x = jax.tree.map(lambda _x: _x[i], xs)
            carry, _keep = body(carry, x)
            keep.append(_keep)
        # stack keeps
        keep = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *keep)
        return keep

    return run_process


def build_process_core_dag(process_id, array_name, full_stokes, plot_folder):
    """
    Build the core DAG for the streaming forward model process.

    Args:
        process_id: the process ID
        array_name: the name of the array
        full_stokes: whether to use full stokes
        plot_folder: the folder to save plots

    Returns:
        a function that executes the DAG and returns the per-step keeps.
    """
    process_local_params = get_process_local_params(
        process_id=process_id,
        array_name=array_name
    )
    plot_folder = os.path.join(plot_folder, f"process_{process_id}")
    os.makedirs(plot_folder, exist_ok=True)
    with open(os.path.join(plot_folder, "process_local_params.json"), "w") as f:
        f.write(process_local_params.json(indent=2))
    # Check the units of each process parameter
    if not process_local_params.freqs.unit.is_equivalent("Hz"):
        raise ValueError("freqs must be in Hz")
    if not process_local_params.integration_interval.unit.is_equivalent("s"):
        raise ValueError("integration_interval must be in seconds")
    if not process_local_params.channel_width.unit.is_equivalent("Hz"):
        raise ValueError("channel_width must be in Hz")
    if process_local_params.solution_interval % process_local_params.integration_interval != 0:
        raise ValueError("solution_interval must be a multiple of integration_interval")
    if process_local_params.validity_interval % process_local_params.solution_interval != 0:
        raise ValueError("validity_interval must be a multiple of solution_interval")
    total_duration = len(process_local_params.obstimes) * process_local_params.integration_interval
    if total_duration % process_local_params.solution_interval != 0:
        raise ValueError("obstimes must be a multiple of solution_interval")
    total_bandwidth = len(process_local_params.freqs) * process_local_params.channel_width
    num_steps = int(total_duration / process_local_params.solution_interval)
    num_antennas = len(process_local_params.antennas)
    print(
        f"Streaming Forward Model:\n"
        f"- array: {array_name}\n"
        f"- num steps: {num_steps}\n"
        f"- observation duration: {total_duration}\n"
        f"- bandwidth: {total_bandwidth}\n"
        f"- spectral window: {process_local_params.freqs.min()} - {process_local_params.freqs.max()}\n"
        f"- num channels: {len(process_local_params.freqs)}\n"
        f"- integration interval: {process_local_params.integration_interval}\n"
        f"- solution interval: {process_local_params.solution_interval}\n"
        f"- validity interval: {process_local_params.validity_interval}\n"
        f"- channel width: {process_local_params.channel_width}\n"
        f"- array location: {process_local_params.array_location.geodetic}\n"
        f"- phase center: {process_local_params.phase_center}\n"
        f"- ref time: {process_local_params.ref_time}\n"
        f"- antennas: {num_antennas}\n"
        f"- plot folder: {plot_folder}"
    )
    setup_observation_step = SetupObservationStep(
        freqs=process_local_params.freqs,
        antennas=process_local_params.antennas,
        array_location=process_local_params.array_location,
        phase_center=process_local_params.phase_center,
        obstimes=process_local_params.obstimes,
        ref_time=process_local_params.ref_time,
        pointings=process_local_params.pointings,
        plot_folder=os.path.join(plot_folder, "setup_observation"),
        solution_interval=process_local_params.solution_interval,
        validity_interval=process_local_params.validity_interval,
        integration_interval=process_local_params.integration_interval
    )
    simulate_beam_step = SimulateBeamStep(
        array_name=array_name,
        full_stokes=full_stokes,
        times=process_local_params.obstimes,
        ref_time=process_local_params.ref_time,
        freqs=process_local_params.freqs,
        plot_folder=os.path.join(plot_folder, "simulate_beam")
    )
    simulate_dish_step = SimulateDishStep(
        static_beam=True,
        dish_effects_params=process_local_params.dish_effects_params,
        convention="physical",
        num_antennas=num_antennas,
        plot_folder=os.path.join(plot_folder, "simulate_dish")
    )
    # simulate_ionosphere_step = SimulateIonosphereStep()
    #
    # create_model_data_step = CreateModelDataStep()
    #
    # predict_and_sample_step = PredictAndSampleStep()
    #
    # flag_step = FlagStep()
    #
    # average_step = AverageStep()
    #
    # create_calibration_model_data_step = CreateCalibrationModelDataStep()
    #
    # dd_predict_step = DDPredictStep()
    #
    # dd_average_step = DDAverageStep()
    #
    # calibrate_step = CalibrateStep()
    #
    # subtract_step = SubtractStep()
    #
    # image_step = ImageStep()
    # DAG
    dag: Dict[AbstractCoreStep, Tuple[AbstractCoreStep, ...]] = dict()  # step -> primals
    dag[setup_observation_step] = ()
    dag[simulate_beam_step] = (setup_observation_step,)
    dag[simulate_dish_step] = (setup_observation_step, simulate_beam_step)

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
    def execute_dag() -> Dict[AbstractCoreStep, Any]:
        # Traverse DAG instead of hardcoding
        step_outputs = dict()
        step_keeps = dict()
        done = set()
        step_idx = 0
        while len(done) < len(dag):
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
                f"-> ({step.output_name}, {step.keep_name})"
            )
            step_primals = tuple(step_outputs[primal] for primal in primals)
            with ctx.scope(step.name):
                step_output, step_keep = step.step(primals=step_primals)
            step_outputs[step] = step_output
            step_keeps[step] = step_keep
            done.add(step)
        return step_keeps

    return execute_dag


def process_start(
        process_id: int,
        key,
        array_name: str,
        full_stokes: bool,
        plot_folder: str,
        max_memory_MB: int | None = None
):
    execute_dag = build_process_core_dag(process_id, array_name, full_stokes, plot_folder)

    execute_dag_transformed = ctx.transform_with_state(execute_dag)

    run_process = build_process_for(execute_dag_transformed)

    # Run the process
    hostname = os.uname().nodename
    print(f"Running on {hostname}")
    start_time = current_utc()
    run_key, init_key = jax.random.split(key, 2)
    print("Initialising...")
    with MemoryLogger(log_file=os.path.join(plot_folder, 'memory_usage.log'), interval=0.5,
                      kill_threshold=max_memory_MB):
        init = block_until_ready(execute_dag_transformed.init(init_key))

        print("Running...")
        final_keep = block_until_ready(
            run_process(run_key, init.params, init.states, num_steps=1)
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

    print(final_keep)
