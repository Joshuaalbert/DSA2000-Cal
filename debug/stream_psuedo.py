from typing import NamedTuple, List, Any, Callable

import jax
import jax.numpy as jnp
from jax import lax

from dsa2000_cal.calibration.probabilistic_models.probabilistic_model import AbstractProbabilisticModel
from dsa2000_cal.delay_models.far_field import VisibilityCoords, FarFieldDelayEngine
from dsa2000_cal.imaging.dirty_imaging import DirtyImaging
from dsa2000_cal.measurement_sets.measurement_set import VisibilityData
from dsa2000_cal.visibility_model.rime_model import RIMEModel


def stream():
    num_time: int = ...
    solution_interval: int = ...
    validity_interval: int = ...
    # Solve once per validity interval
    solve_cadence = validity_interval // solution_interval
    # Define callbacks after each step
    aggegrate_image: Callable = ...
    callbacks = [
        aggegrate_image
    ]

    # Holders for state that will be passed forward
    init_params = None
    solver_state = None
    for cadence_idx in range(0, num_time // solution_interval):
        # Determine if we solve or not
        if cadence_idx % solve_cadence == 0:
            num_cal_iters = 15
        else:
            num_cal_iters = 0
        # Run step
        times = jnp.arange(solution_interval) + cadence_idx * solution_interval
        key = jax.random.PRNGKey(cadence_idx)
        step_return = step(
            key=key,
            num_calibration_iters=num_cal_iters,
            times=times,
            init_params=init_params,
            solver_state=solver_state
        )
        # Update state that will be passed forward
        init_params = step_return.cal_params
        solver_state = step_return.calibration_solver_state
        # Run callbacks
        for callback in callbacks:
            callback(step_return)


class StepReturn(NamedTuple):
    cal_params: Any
    calibration_solver_state: Any
    solver_aux: Any
    vis_residual: jax.Array
    image_pb_cor: jax.Array
    image_psf: jax.Array


def step(key: jax.Array, times: jax.Array, init_params: List[jax.Array] | None,
         solver_state: Any | None, num_calibration_iters: int) -> StepReturn:
    """
    Kernel that defines a single step of forward model.

    Args:
        key: PRNGkey for reproducibility
        times: the times simulated at this step, must be a solution interval worth
        init_params: last cal params
        solver_state: last solver state
        num_calibration_iters: number of calibration iterations to run, 0 means no calibration on this step

    Returns:
        StepReturn: the return values of the step
    """
    # Define core components of the forward model
    rime_model: RIMEModel = ...
    probabilistic_models: List[AbstractProbabilisticModel] = ...
    far_field_delay_engine: FarFieldDelayEngine = ...
    noise_scale: jax.Array = ...
    solver = ...
    imagor: DirtyImaging = ...
    freqs: jax.Array = ...

    # Run sequence of steps, essentially aggregated across loops of individual steps. Normally we'd do each step for
    # the whole forward model, but now we do it streaming.
    visibility_coords = step_compute_visibility_coords(far_field_delay_engine, times)
    vis_data = step_compute_simulated_visibilities(key, noise_scale, rime_model, times, visibility_coords)
    stream_flags, vis_data = step_compute_stream_flag(vis_data)
    final_state, params, gains, solver_aux = step_compute_calibration(freqs, init_params,
                                                                      num_calibration_iters,
                                                                      probabilistic_models,
                                                                      solver, solver_state,
                                                                      times, vis_data,
                                                                      visibility_coords)
    vis_residual = step_compute_subtraction(gains, rime_model, vis_data, visibility_coords)
    image_pb_cor, image_psf = step_compute_image(imagor, stream_flags, vis_residual, visibility_coords)
    # Send returns of step
    return StepReturn(
        cal_params=params,
        calibration_solver_state=final_state,
        solver_aux=solver_aux,
        vis_residual=vis_residual,
        image_pb_cor=image_pb_cor,
        image_psf=image_psf
    )


def step_compute_image(imagor, stream_flags, vis_residual, visibility_coords):
    # Image the subtracted visibilities
    image_weights: jax.Array = ...
    image = imagor.image_visibilities(
        uvw=visibility_coords.uvw,
        vis=vis_residual,
        weights=image_weights,
        flags=stream_flags
    )
    # PB correction
    image_pb_cor = ...
    image_psf = imagor.image_visibilities(
        uvw=visibility_coords.uvw,
        vis=jnp.ones_like(vis_residual),
        weights=image_weights,
        flags=stream_flags
    )
    return image_pb_cor, image_psf


def step_compute_subtraction(gains: jax.Array, rime_model: RIMEModel, vis_data: VisibilityData,
                             visibility_coords: VisibilityCoords):
    # Predict at full resolution
    vis_model = rime_model.apply_gains(gains, vis_data.vis, visibility_coords)
    vis_residual = vis_data.vis - vis_model
    return vis_residual


def step_compute_calibration(freqs, init_params, num_calibration_iters, probabilistic_models, solver, solver_state,
                             times, vis_data, visibility_coords):
    # Create calibration probabilistic model instances
    probabilistic_model_instances = [
        probabilistic_model.create_model_instance(
            freqs=freqs,
            times=times,
            vis_data=vis_data,
            vis_coords=visibility_coords
        ) for probabilistic_model in probabilistic_models
    ]
    # Add together the probabilistic model instances into one
    probabilistic_model_instance = probabilistic_model_instances[0]
    for other_model in probabilistic_model_instances[1:]:
        probabilistic_model_instance = probabilistic_model_instance + other_model
    # Prepare initial guess and solver state, using last state if available
    if init_params is None:
        init_params = probabilistic_model_instance.get_init_params()
    if solver_state is None:
        solver_state = solver.init_state(init_params=init_params)

    def body_fn(carry, x):
        params, solver_state = carry
        (params, solver_state), solver_aux = solver.update(params=params, state=solver_state)
        return (params, solver_state), solver_aux

    carry = (init_params, solver_state)
    if num_calibration_iters > 0:
        (params, final_state), solver_aux = lax.scan(body_fn, carry, xs=jnp.arange(num_calibration_iters))

    else:
        (params, final_state) = carry
        solver_aux = jnp.asarray([])
    _, gains = probabilistic_model_instance.forward(params)
    return final_state, params, gains, solver_aux


def step_compute_stream_flag(vis_data):
    # Flag data and update data
    stream_flags = ...
    vis_data = vis_data._replace(flags=stream_flags)
    return stream_flags, vis_data


def step_compute_simulated_visibilities(key, noise_scale, rime_model, times, visibility_coords):
    # Get model data
    model_data = rime_model.get_model_data(times)
    # Compute visibilities
    visibilities: jax.Array = rime_model.predict_visibilities(
        model_data=model_data,
        visibility_coords=visibility_coords
    )
    # Simulate measurement noise
    key1, key2 = jax.random.split(key)
    noise = noise_scale * (
            jax.random.normal(key1, visibilities.shape) + 1j * jax.random.normal(key2, visibilities.shape)
    )
    visibilities += noise
    # Create visibility data
    weights = jnp.full(visibilities.shape, 1. / noise_scale ** 2)
    flags = jnp.zeros(visibilities.shape, dtype=bool)
    vis_data = VisibilityData(
        vis=visibilities,
        weights=weights,
        flags=flags
    )
    return vis_data


def step_compute_visibility_coords(far_field_delay_engine, times):
    # Compute visibility coords
    visibility_coords: VisibilityCoords = far_field_delay_engine.compute_visibility_coords(
        times=times
    )
    return visibility_coords
