import dataclasses
import os
from typing import NamedTuple, Tuple

import astropy.units as au
import jax
import numpy as np

import dsa2000_cal.common.context as ctx
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.common.array_types import ComplexArray
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.noise import calc_baseline_noise
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.forward_models.streaming.abc import AbstractCoreStep
from dsa2000_cal.forward_models.streaming.core.setup_observation import SetupObservationOutput
from dsa2000_cal.forward_models.streaming.core.simulate_dish import SimulateDishOutput
from dsa2000_cal.visibility_model.source_models.celestial.base_fits_source_model import BaseFITSSourceModel, \
    build_fits_source_model_from_wsclean_components
from dsa2000_cal.visibility_model.source_models.celestial.base_gaussian_source_model import \
    build_gaussian_source_model_from_wsclean_components, BaseGaussianSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.base_point_source_model import BasePointSourceModel, \
    build_point_source_model_from_wsclean_components


class PredictAndSampleState(NamedTuple):
    faint_source_model: BaseFITSSourceModel
    bright_source_model_points: BasePointSourceModel
    bright_source_model_gaussians: BaseGaussianSourceModel


class PredictAndSampleOutput(NamedTuple):
    visibilities: ComplexArray


class PredictAndSampleReturn(NamedTuple):
    visibilities: ComplexArray


@dataclasses.dataclass(eq=False)
class PredictAndSampleStep(AbstractCoreStep[PredictAndSampleOutput, PredictAndSampleReturn]):
    plot_folder: str
    freqs: au.Quantity
    faint_sky_model_id: str  # Likely TRECS
    bright_sky_model_id: str  # Likely FIRST > 5Jy resampled onto sky with about 100 over >-30deg sky
    num_facets_per_side: int
    crop_box_size: au.Quantity | None
    full_stokes: bool
    system_equivalent_flux_density: au.Quantity
    channel_width: au.Quantity
    integration_time: au.Quantity
    convention: str = 'physical'

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        fill_registries()

        model_freqs = au.Quantity([self.freqs[0], self.freqs[-1]])
        wsclean_fits_files = source_model_registry.get_instance(
            source_model_registry.get_match(self.faint_sky_model_id)).get_wsclean_fits_files()
        # -04:00:28.608,40.43.33.595

        self.faint_sky_model = build_fits_source_model_from_wsclean_components(
            wsclean_fits_files=wsclean_fits_files,
            model_freqs=model_freqs,
            full_stokes=self.full_stokes,
            crop_box_size=self.crop_box_size,
            num_facets_per_side=self.num_facets_per_side
        )

        wsclean_clean_component_file = source_model_registry.get_instance(
            source_model_registry.get_match(self.bright_sky_model_id)).get_wsclean_clean_component_file()

        self.bright_sky_model_points = build_point_source_model_from_wsclean_components(
            wsclean_clean_component_file=wsclean_clean_component_file,
            model_freqs=model_freqs,
            full_stokes=self.full_stokes
        )

        self.bright_sky_model_gaussians = build_gaussian_source_model_from_wsclean_components(
            wsclean_clean_component_file=wsclean_clean_component_file,
            model_freqs=model_freqs,
            full_stokes=self.full_stokes
        )

        # TODO: RFI emiiter add here

    def get_state(self) -> PredictAndSampleState:
        return PredictAndSampleState(
            faint_source_model=self.faint_sky_model,
            bright_source_model_points=self.bright_sky_model_points,
            bright_source_model_gaussians=self.bright_sky_model_gaussians
        )

    def step(self, primals: Tuple[SetupObservationOutput, SimulateDishOutput]) -> Tuple[
        PredictAndSampleOutput, PredictAndSampleReturn]:
        state: PredictAndSampleState = ctx.get_state('state', init=lambda: self.get_state())
        (setup_observation_output, simulate_dish_output) = primals

        visibility_coords = setup_observation_output.far_field_delay_engine.compute_visibility_coords(
            freqs=setup_observation_output.freqs,
            times=setup_observation_output.times,
            with_autocorr=True,
            convention=self.convention
        )
        vis_faint = state.faint_source_model.predict(
            visibility_coords=visibility_coords,
            gain_model=simulate_dish_output.gain_model,
            near_field_delay_engine=setup_observation_output.near_field_delay_engine,
            far_field_delay_engine=setup_observation_output.far_field_delay_engine,
            geodesic_model=setup_observation_output.geodesic_model
        )
        vis_bright_points = state.bright_source_model_points.predict(
            visibility_coords=visibility_coords,
            gain_model=simulate_dish_output.gain_model,
            near_field_delay_engine=setup_observation_output.near_field_delay_engine,
            far_field_delay_engine=setup_observation_output.far_field_delay_engine,
            geodesic_model=setup_observation_output.geodesic_model
        )
        vis_bright_gaussians = state.bright_source_model_gaussians.predict(
            visibility_coords=visibility_coords,
            gain_model=simulate_dish_output.gain_model,
            near_field_delay_engine=setup_observation_output.near_field_delay_engine,
            far_field_delay_engine=setup_observation_output.far_field_delay_engine,
            geodesic_model=setup_observation_output.geodesic_model
        )
        visibilities = vis_faint + vis_bright_points + vis_bright_gaussians

        noise_scale = calc_baseline_noise(
            system_equivalent_flux_density=quantity_to_jnp(self.system_equivalent_flux_density, 'Jy'),
            chan_width_hz=quantity_to_jnp(self.channel_width, 'Hz'),
            t_int_s=quantity_to_jnp(self.integration_time, 's')
        )

        # Simulate measurement noise
        key1, key2 = jax.random.split(ctx.next_rng_key())
        # Divide by sqrt(2) to account for 2 polarizations
        num_pol = 2 if self.full_stokes else 1
        noise = mp_policy.cast_to_vis((noise_scale / np.sqrt(num_pol)) * jax.lax.complex(
            jax.random.normal(key1, np.shape(visibilities)), jax.random.normal(key2, np.shape(visibilities))
        ))
        visibilities += noise
        output = PredictAndSampleOutput(visibilities=visibilities)
        step_return = PredictAndSampleReturn(visibilities=visibilities)
        return output, step_return
