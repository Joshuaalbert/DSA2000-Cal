import asyncio
import logging
import os
from datetime import timedelta
from typing import NamedTuple

import jax
import numpy as np
import ray
from ray.runtime_env import RuntimeEnv

from dsa2000_common.common.array_types import FloatArray, IntArray
from dsa2000_common.common.ray_utils import resource_logger
from dsa2000_common.common.serialise_utils import SerialisableBaseModel
from dsa2000_common.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_common.gain_models.gain_model import GainModel
from dsa2000_common.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_fm.forward_models.streaming.common import ForwardModellingRunParams
from dsa2000_fm.systematics.dish_aperture_effects import DishApertureEffects

logger = logging.getLogger('ray')


class SimulationParams(NamedTuple):
    dish_diameter: jax.Array
    focal_length: jax.Array
    elevation_pointing_error_stddev: jax.Array
    cross_elevation_pointing_error_stddev: jax.Array
    axial_focus_error_stddev: jax.Array
    elevation_feed_offset_stddev: jax.Array
    cross_elevation_feed_offset_stddev: jax.Array
    horizon_peak_astigmatism_stddev: jax.Array
    surface_error_mean: jax.Array
    surface_error_stddev: jax.Array


class StaticDishRealisationParams(NamedTuple):
    elevation_feed_offset: jax.Array
    cross_elevation_feed_offset: jax.Array
    horizon_peak_astigmatism: jax.Array
    surface_error: jax.Array


class DynamicDishRealisationParams(NamedTuple):
    elevation_point_error: jax.Array  # [num_time, num_ant]
    cross_elevation_point_error: jax.Array  # [num_time, num_ant]
    axial_focus_error: jax.Array  # [num_time, num_ant]


class SimulateDishState(NamedTuple):
    beam_aperture: jax.Array

    # Transition parameters
    dish_effect_params: SimulationParams
    static_system_params: StaticDishRealisationParams

    # Static parameters
    L: jax.Array
    M: jax.Array
    dl: jax.Array
    dm: jax.Array
    X: jax.Array
    Y: jax.Array
    dx: jax.Array
    dy: jax.Array
    model_freqs: jax.Array
    model_times: FloatArray
    lvec: FloatArray
    mvec: FloatArray
    lmn_image: jax.Array  # [Nl, Nm, 3]


class SystemGainSimulatorParams(SerialisableBaseModel):
    geodesic_model: BaseGeodesicModel
    init_key: IntArray
    apply_effects: bool
    simulate_ionosphere: bool


class SystemGainSimulatorResponse(NamedTuple):
    gain_model: GainModel


def compute_system_gain_simulator_options(run_params: ForwardModellingRunParams):
    # memory is 600MB
    memory = 600 * 2 ** 20
    return {
        "num_cpus": 1,
        "num_gpus": 0,
        'memory': 1.1 * memory,
        "runtime_env": RuntimeEnv(
            env_vars={
                # "XLA_PYTHON_CLIENT_MEM_FRACTION": ".75",
                # "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                # "XLA_PYTHON_CLIENT_ALLOCATOR":"platform",
                "JAX_PLATFORMS": "cpu"
            }
        )
    }


@ray.remote
class SystemGainSimulator:

    def __init__(self, params: ForwardModellingRunParams, system_gain_simulator_params: SystemGainSimulatorParams):
        self.params = params
        self.system_gain_simulator_params = system_gain_simulator_params
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'system_gain_simulator')
        os.makedirs(self.params.plot_folder, exist_ok=True)
        self._initialised = False
        self._memory_logger_task: asyncio.Task | None = None

    async def init(self):
        if self._initialised:
            return
        self._initialised = True
        self._memory_logger_task = asyncio.create_task(
            resource_logger(task='system_gain_simulator', cadence=timedelta(seconds=5)))

        self.beam_model = build_beam_gain_model(
            array_name=self.params.ms_meta.array_name,
            times=self.params.ms_meta.times,
            ref_time=self.params.ms_meta.ref_time,
            freqs=self.params.ms_meta.freqs,
            full_stokes=self.params.full_stokes
        )

        if self.system_gain_simulator_params.apply_effects:
            dish_aperture_effects = DishApertureEffects(
                dish_diameter=self.params.dish_effects_params.dish_diameter,
                focal_length=self.params.dish_effects_params.focal_length,
                elevation_pointing_error_stddev=self.params.dish_effects_params.elevation_pointing_error_stddev,
                cross_elevation_pointing_error_stddev=self.params.dish_effects_params.cross_elevation_pointing_error_stddev
            )
            self.beam_model = dish_aperture_effects.apply_dish_aperture_effects(
                key=jax.random.PRNGKey(0),
                beam_model=self.beam_model,
                geodesic_model=self.system_gain_simulator_params.geodesic_model
            )

        if self.system_gain_simulator_params.simulate_ionosphere:
            pass

    async def __call__(self, key, time_idx: int, freq_idx: int) -> SystemGainSimulatorResponse:
        logger.info(f"Simulating dish gains for time {time_idx} and freq {freq_idx}")
        await self.init()

        # Jones = Beam x Ionosphere (order matters)

        if self.system_gain_simulator_params.simulate_ionosphere:
            raise NotImplementedError("Ionosphere simulation not implemented yet.")

        gain_model = self.beam_model

        # check for nans
        if np.any(np.isnan(gain_model.model_gains)):
            raise ValueError("NaNs in the gain model.")
        return SystemGainSimulatorResponse(gain_model=gain_model)
