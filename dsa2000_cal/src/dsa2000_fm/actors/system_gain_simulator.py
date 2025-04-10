import asyncio
import logging
import os
from datetime import timedelta
from typing import NamedTuple

import jax
import numpy as np
import ray
from ray.runtime_env import RuntimeEnv

from dsa2000_assets.registries import array_registry
from dsa2000_common.common.array_types import FloatArray, IntArray
from dsa2000_common.common.astropy_utils import create_spherical_spiral_grid
from dsa2000_common.common.ray_utils import resource_logger, TimerLog
from dsa2000_common.common.serialise_utils import SerialisableBaseModel
from dsa2000_common.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_common.gain_models.gain_model import GainModel
from dsa2000_common.geodesics.base_geodesic_model import BaseGeodesicModel, build_geodesic_model
from dsa2000_fm.actors.common import ForwardModellingRunParams
from dsa2000_fm.systematics.dish_aperture_effects import build_dish_aperture_effects
from dsa2000_fm.systematics.ionosphere import compute_x0_radius, build_ionosphere_gain_model
from dsa2000_fm.systematics.ionosphere_models import construct_canonical_ionosphere

from dsa2000_common.common.logging import dsa_logger as logger


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


@jax.jit
def apply_dish_effects(sample_key, dish_aperture_effects, beam_model, geodesic_model):
    return dish_aperture_effects.apply_dish_aperture_effects(
        sample_key,
        beam_model,
        geodesic_model=geodesic_model
    )


@ray.remote
class SystemGainSimulator:

    def __init__(self, params: ForwardModellingRunParams, system_gain_simulator_params: SystemGainSimulatorParams):
        self.params = params
        self.system_gain_simulator_params = system_gain_simulator_params
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'system_gain_simulator')
        os.makedirs(self.params.plot_folder, exist_ok=True)
        self._initialised = False
        self._memory_logger_task: asyncio.Task | None = None

    async def init(self, key):
        if self._initialised:
            return
        self._initialised = True
        self._memory_logger_task = asyncio.create_task(
            resource_logger(task='system_gain_simulator', cadence=timedelta(seconds=5)))

        self.uncorrupted_beam_model = build_beam_gain_model(
            array_name=self.params.ms_meta.array_name,
            times=self.params.ms_meta.times,
            ref_time=self.params.ms_meta.ref_time,
            freqs=self.params.ms_meta.freqs,
            full_stokes=self.params.full_stokes
        )
        cpu = jax.devices("cpu")[0]
        array = array_registry.get_instance(array_registry.get_match(self.params.ms_meta.array_name))
        if self.system_gain_simulator_params.apply_effects:
            geodesic_model = build_geodesic_model(
                antennas=array.get_antennas(),
                array_location=array.get_array_location(),
                phase_center=self.params.ms_meta.phase_center,
                obstimes=self.params.ms_meta.times,
                ref_time=self.params.ms_meta.ref_time,
                pointings=self.params.ms_meta.phase_center
            )
            with TimerLog('Simulating dish aperture effects'):
                with jax.default_device(cpu):
                    dish_aperture_effects = build_dish_aperture_effects(
                        dish_diameter=array.get_antenna_diameter(),
                        focal_length=array.get_focal_length(),
                        elevation_pointing_error_stddev=self.params.dish_effects_params.elevation_pointing_error_stddev,
                        cross_elevation_pointing_error_stddev=self.params.dish_effects_params.cross_elevation_pointing_error_stddev,
                        axial_focus_error_stddev=self.params.dish_effects_params.axial_focus_error_stddev,
                        elevation_feed_offset_stddev=self.params.dish_effects_params.elevation_feed_offset_stddev,
                        cross_elevation_feed_offset_stddev=self.params.dish_effects_params,
                        horizon_peak_astigmatism_stddev=self.params.dish_effects_params.horizon_peak_astigmatism_stddev,
                        # surface_error_mean=0 * au.mm, # TODO: update to use a GP model for RMS surface error
                        # surface_error_stddev=1 * au.mm
                    )
                    key, sample_key = jax.random.split(key)
                    self.beam_model = apply_dish_effects(
                        sample_key,
                        dish_aperture_effects,
                        self.uncorrupted_beam_model,
                        geodesic_model
                    )
                    self.beam_model.plot_regridded_beam(
                        save_fig=os.path.join(self.params.plot_folder, f'beam_model_with_aperture_effects.png'),
                        ant_idx=-1,
                        show=False
                    )
        else:
            self.beam_model = self.uncorrupted_beam_model

        if self.system_gain_simulator_params.simulate_ionosphere:
            with TimerLog("Simulating ionosphere..."):

                with jax.default_device(cpu):
                    x0_radius = compute_x0_radius(array.get_array_location(), self.params.ms_meta.ref_time)
                    ionosphere = construct_canonical_ionosphere(
                        x0_radius=x0_radius,
                        turbulent=self.params.ionosphere_params.turbulent,
                        dawn=self.params.ionosphere_params.dawn,
                        high_sun_spot=self.params.ionosphere_params.high_sun_spot
                    )

                    ionosphere_model_directions = create_spherical_spiral_grid(
                        pointing=self.params.ms_meta.phase_center,
                        num_points=20,
                        angular_radius=0.5 * self.params.field_of_view
                    )
                    print(f"Number of ionosphere sample directions: {len(ionosphere_model_directions)}")

                    key, sim_key = jax.random.split(key)
                    ionosphere_gain_model = build_ionosphere_gain_model(
                        key=sim_key,
                        ionosphere=ionosphere,
                        model_freqs=self.params.ms_meta.freqs[[0, len(self.params.ms_meta.freqs) // 2, -1]],
                        antennas=array.get_antennas(),
                        ref_location=array.get_array_location(),
                        times=self.params.ms_meta.times,
                        ref_time=self.params.ms_meta.ref_time,
                        directions=ionosphere_model_directions,
                        phase_centre=self.params.ms_meta.phase_center,
                        full_stokes=False,
                        predict_batch_size=512,
                        save_file=os.path.join(self.params.plot_folder, f'simulated_dtec.json')
                    )
                    ionosphere_gain_model.plot_regridded_beam(
                        save_fig=os.path.join(self.params.plot_folder, f'ionosphere_model.png'),
                        ant_idx=-1,
                        show=False
                    )

            self.beam_model = self.beam_model @ ionosphere_gain_model
        else:
            self.beam_model = self.beam_model

        # check for nans
        if np.any(np.isnan(self.beam_model.model_gains)):
            raise ValueError("NaNs in the gain model.")

    async def __call__(self, key, time_idx: int, freq_idx: int) -> SystemGainSimulatorResponse:
        logger.info(f"Simulating dish gains for time {time_idx} and freq {freq_idx}")
        key, init_key = jax.random.split(key)
        await self.init(key)
        # Jones = Beam x Ionosphere (order matters)
        # TODO: interpolation onto a certain time/freq
        return SystemGainSimulatorResponse(gain_model=self.beam_model)
