import dataclasses
import os
from typing import Tuple

import astropy.units as au
import jax
import jax.numpy as jnp
from jax._src.typing import SupportsDType
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import DiagonalUnconstrainedGain
from dsa2000_cal.calibration.probabilistic_models.gains_per_facet import GainsPerFacet
from dsa2000_cal.common.alert_utils import post_completed_forward_modelling_run
from dsa2000_cal.common.datetime_utils import current_utc
from dsa2000_cal.common.fits_utils import ImageModel
from dsa2000_cal.forward_model.simulation.simulate_systematics import SimulateSystematics
from dsa2000_cal.forward_model.simulation.simulate_visibilties import SimulateVisibilities
from dsa2000_cal.forward_model.synthetic_sky_model.synthetic_sky_model_producer import SyntheticSkyModelProducer
from dsa2000_cal.forward_model.systematics.dish_effects_simulation import DishEffectsParams
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.imaging.dirty_imaging import DirtyImaging
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet
from dsa2000_cal.visibility_model.facet_model import FacetModel
from dsa2000_cal.visibility_model.rime_model import RIMEModel


@dataclasses.dataclass(eq=False)
class ForwardModel:
    """
    Runs forward modelling using a sharded data structure over devices.

    Args:
        synthetic_sky_model_producer: the synthetic sky model producer
        dish_effect_params: the dish effect model parameters
        ionosphere_specification: the ionosphere model specification, see tomographic_kernel.models.cannonical_models
        plot_folder: the folder to store plots
        cache_folder: the folder to store cache
        ionosphere_seed: the seed for the ionosphere model
        dish_effects_seed: the seed for the dish effects model
        simulation_seed: the seed for the simulation
        calibration_seed: the seed for the calibration
        imaging_seed: the seed for the imaging
        field_of_view: the field of view for imaging, default computes from dish model
        oversample_factor: the oversample factor for imaging, default 2.5
        epsilon: the epsilon for wgridder, default 1e-4
        convention: the convention for imaging, default 'casa' which negates uvw coordinates,
            i.e. FT with e^{2\pi i} unity root
        dtype: the dtype for imaging, default jnp.complex64
        verbose: the verbosity for imaging, default False
    """

    # Synthetic sky model producer
    synthetic_sky_model_producer: SyntheticSkyModelProducer

    # Dish effect model parameters
    dish_effect_params: DishEffectsParams

    # Ionosphere model parameters
    ionosphere_specification: SPECIFICATION

    # Plot and cache folders
    plot_folder: str
    solution_folder: str
    cache_folder: str

    # Seeds
    ionosphere_seed: int = 42
    dish_effects_seed: int = 42
    simulation_seed: int = 424242
    calibration_seed: int = 42424242
    imaging_seed: int = 4242424242

    # Imaging parameters
    field_of_view: au.Quantity | None = None
    oversample_factor: float = 5.

    # Wgridder parameters
    epsilon: float = 1e-4

    # Common parameters
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64
    verbose: bool = False
    num_shards: int = 1

    def forward(self, ms: MeasurementSet):
        start_time = current_utc()

        # Simulate systematics
        system_gain_model, dish_effects_gain_model = self._simulate_systematics(
            ms=ms
        )

        # Simulate visibilities
        self._simulate_visibilities(
            ms=ms,
            system_gain_model=system_gain_model,
            dish_effects_gain_model=dish_effects_gain_model
        )

        # # Calibrate visibilities
        # subtracted_ms = self._calibrate_visibilities(
        #     ms=ms
        # )

        # Image visibilities
        self._image_visibilities(image_name='dirty_image', ms=ms)

        # # Image subtracted visibilities
        # self._image_visibilities(image_name='subtracted_dirty_image', ms=subtracted_ms)

        # Tell Slack we're done
        post_completed_forward_modelling_run(
            run_dir=os.getcwd(),
            start_time=start_time,
            duration=current_utc() - start_time
        )

    def _simulate_systematics(self, ms: MeasurementSet) -> Tuple[GainModel, GainModel]:
        """
        Simulate systematics such as ionosphere and dish effects.

        Returns:
            system_gain_model: the system gain model
            dish_gain_model: the beam gain model
        """

        simulator = SimulateSystematics(
            dish_effect_params=self.dish_effect_params,
            ionosphere_specification=self.ionosphere_specification,
            plot_folder=self.plot_folder,
            cache_folder=self.cache_folder,
            ionosphere_seed=self.ionosphere_seed,
            dish_effects_seed=self.dish_effects_seed,
            verbose=self.verbose,
            full_stokes=ms.is_full_stokes()
        )

        # Order is by right multiplication of systematics encountered by radiation from source to observer
        return simulator.simulate(ms=ms)

    def _simulate_visibilities(self, ms: MeasurementSet, system_gain_model: GainModel,
                               dish_effects_gain_model: GainModel):
        """
        Simulate visibilities using the system gain model, and store the results in the MS.

        Args:
            ms: the measurement set to store the results
            system_gain_model: the system gain model
            dish_effects_gain_model: the dish effects gain model
        """
        # Construct single RIME model with systematics gain models
        bright_point_sources = self.synthetic_sky_model_producer.create_sources_outside_fov(
            full_stokes=ms.is_full_stokes()
        )
        bright_point_sources.plot(save_file=os.path.join(self.plot_folder, 'bright_point_sources.png'))
        inner_point_sources = self.synthetic_sky_model_producer.create_sources_inside_fov(
            full_stokes=ms.is_full_stokes()
        )
        inner_point_sources.plot(save_file=os.path.join(self.plot_folder, 'inner_point_sources.png'))
        inner_diffuse_sources = self.synthetic_sky_model_producer.create_diffuse_sources_inside_fov(
            full_stokes=ms.is_full_stokes()
        )
        inner_diffuse_sources.plot(save_file=os.path.join(self.plot_folder, 'inner_diffuse_sources.png'))
        rfi_emitter_sources = self.synthetic_sky_model_producer.create_rfi_emitter_sources(
            full_stokes=ms.is_full_stokes()
        )
        rfi_emitter_sources[0].plot(save_file=os.path.join(self.plot_folder, 'rfi_emitter_sources.png'))

        # Note: if a-team added then make sure to create separate facet for each to give each its own gain evaluation.
        celestial_facet_model = FacetModel(
            point_source_model=inner_point_sources,
            gaussian_source_model=inner_diffuse_sources,
            rfi_emitter_source_model=None,
            fits_source_model=None,
            gain_model=None,  # system_gain_model,
            near_field_delay_engine=ms.near_field_delay_engine,
            far_field_delay_engine=ms.far_field_delay_engine,
            geodesic_model=ms.geodesic_model,
            convention=self.convention,
            dtype=self.dtype
        )
        # Give RFI just dish effects, not ionosphere
        rfi_facet_models = [
            FacetModel(
                point_source_model=None,
                gaussian_source_model=None,
                rfi_emitter_source_model=rfi_emitter_source,
                fits_source_model=None,
                gain_model=None,  # dish_effects_gain_model,
                near_field_delay_engine=ms.near_field_delay_engine,
                far_field_delay_engine=ms.far_field_delay_engine,
                geodesic_model=ms.geodesic_model,
                convention=self.convention,
                dtype=self.dtype
            )
            for rfi_emitter_source in rfi_emitter_sources
        ]
        rime_model = RIMEModel(
            facet_models=[celestial_facet_model] + rfi_facet_models
        )

        simulator = SimulateVisibilities(
            rime_model=rime_model,
            verbose=self.verbose,
            seed=self.simulation_seed,
            num_shards=self.num_shards,
            plot_folder=os.path.join(self.plot_folder, 'simulate')
        )
        simulator.simulate(
            ms=ms
        )

    def _calibrate_visibilities(self, ms: MeasurementSet) -> MeasurementSet:
        """
        Calibrate visibilities using the RIME model.

        Args:
            ms: the measurement set to store the results

        Returns:
            subtracted_ms: the calibrated visibilities
        """
        beam_gain_model = beam_gain_model_factory(ms.meta.array_name)

        gain_prior_model = DiagonalUnconstrainedGain()
        # These are the same
        inner_point_sources = self.synthetic_sky_model_producer.create_sources_inside_fov()
        calibrator_facets = [
            FacetModel(
                point_source_model=inner_point_sources[i:i + 1],
                gaussian_source_model=None,
                rfi_emitter_source_model=None,
                fits_source_model=None,
                gain_model=beam_gain_model,
                near_field_delay_engine=ms.near_field_delay_engine,
                far_field_delay_engine=ms.far_field_delay_engine,
                geodesic_model=ms.geodesic_model,
                convention=self.convention,
                dtype=self.dtype
            )
            for i in range(inner_point_sources.num_sources)
        ]
        rime_model = RIMEModel(
            facet_models=calibrator_facets

        )
        gains_per_facet = GainsPerFacet(
            gain_prior_model=gain_prior_model,
            rime_model=rime_model
        )

        probabilistic_models = [gains_per_facet]

        calibration = Calibration(
            probabilistic_models=probabilistic_models,
            num_iterations=15,
            inplace_subtract=False,
            residual_ms_folder='residual_ms',
            solution_interval=None,
            validity_interval=None,
            verbose=self.verbose,
            seed=self.calibration_seed,
            num_shards=self.num_shards,
            plot_folder=os.path.join(self.plot_folder, 'calibration'),
            solution_folder=self.solution_folder
        )

        return calibration.calibrate(ms=ms)

    def _image_visibilities(self, image_name: str, ms: MeasurementSet) -> ImageModel:
        imagor = DirtyImaging(
            plot_folder=os.path.join(self.plot_folder, 'imaging'),
            field_of_view=self.field_of_view,
            seed=self.imaging_seed,
            oversample_factor=self.oversample_factor,
            nthreads=len(jax.devices()),
            convention=self.convention
        )
        return imagor.image(image_name=image_name, ms=ms)
