import dataclasses
import os

import jax
import jax.numpy as jnp
from astropy import units as au
from jax._src.typing import SupportsDType

from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import DiagonalUnconstrainedGain
from dsa2000_cal.calibration.probabilistic_models.gains_per_facet import GainsPerFacet
from dsa2000_cal.common.alert_utils import post_completed_forward_modelling_run
from dsa2000_cal.common.datetime_utils import current_utc
from dsa2000_cal.common.fits_utils import ImageModel
from dsa2000_cal.forward_models.forward_model import AbstractForwardModel
from dsa2000_cal.forward_models.simulation.simulate_visibilties import SimulateVisibilities
from dsa2000_cal.forward_models.synthetic_sky_model.synthetic_sky_model_producer import SyntheticSkyModelProducer
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.imaging.dirty_imaging import DirtyImaging
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet
from dsa2000_cal.visibility_model.facet_model import FacetModel
from dsa2000_cal.visibility_model.rime_model import RIMEModel


@dataclasses.dataclass(eq=False)
class LWAForwardModel(AbstractForwardModel):
    """
    Runs forward modelling using a sharded data structure over devices.

    Args:
        synthetic_sky_model_producer: the synthetic sky model producer
        plot_folder: the folder to store plots
        cache_folder: the folder to store cache
        simulation_seed: the seed for the simulation
        calibration_seed: the seed for the calibration
        imaging_seed: the seed for the imaging
        field_of_view: the field of view for imaging, default computes from dish model
        oversample_factor: the oversample factor for imaging, default 2.5
        epsilon: the epsilon for wgridder, default 1e-4
        dtype: the dtype for imaging, default jnp.complex64
        verbose: the verbosity for imaging, default False
    """

    # Synthetic sky model producer
    synthetic_sky_model_producer: SyntheticSkyModelProducer

    # Plot and cache folders
    run_folder: str

    # Seeds
    simulation_seed: int = 424242
    calibration_seed: int = 42424242
    imaging_seed: int = 4242424242

    # Simulation parameters
    add_noise: bool = True

    # Imaging parameters
    field_of_view: au.Quantity | None = None
    oversample_factor: float = 5.
    weighting: str = 'natural'
    epsilon: float = 1e-4

    # Common parameters
    verbose: bool = False
    num_shards: int = 1
    dtype: SupportsDType = jnp.complex64

    def __post_init__(self):
        self.plot_folder = os.path.join(self.run_folder, 'plots')
        self.solution_folder = os.path.join(self.run_folder, 'solution')
        self.cache_folder = os.path.join(self.run_folder, 'cache')
        os.makedirs(self.plot_folder, exist_ok=True)
        os.makedirs(self.solution_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)

    def forward(self, ms: MeasurementSet):
        start_time = current_utc()

        beam_gain_model = beam_gain_model_factory(ms)

        beam_gain_model.plot_beam(os.path.join(self.plot_folder, 'beam.png'))

        # Simulate visibilities
        self._simulate_visibilities(
            ms=ms,
            beam_gain_model=beam_gain_model
        )

        # # Calibrate visibilities
        # subtracted_ms = self._calibrate_visibilities(
        #     ms=ms,
        #     beam_gain_model=beam_gain_model
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

    def _simulate_visibilities(self, ms: MeasurementSet, beam_gain_model: GainModel):
        """
        Simulate visibilities using the system gain model, and store the results in the MS.

        Args:
            ms: the measurement set to store the results
            beam_gain_model: the system gain model
        """
        # Construct single RIME model with systematics gain models
        bright_point_sources = self.synthetic_sky_model_producer.create_sources_outside_fov(
            full_stokes=ms.is_full_stokes()
        )
        bright_point_sources.plot(save_file=os.path.join(self.plot_folder, 'bright_point_sources.png'))

        rfi_emitter_sources = self.synthetic_sky_model_producer.create_rfi_emitter_sources(
            rfi_sources=['lwa_cell_tower'],
            full_stokes=ms.is_full_stokes()
        )
        rfi_emitter_sources[0].plot(save_file=os.path.join(self.plot_folder, 'rfi_emitter_sources.png'))
        a_team_sources = self.synthetic_sky_model_producer.create_a_team_sources(
            full_stokes=ms.is_full_stokes()
        )
        for i, a_team_source in enumerate(a_team_sources):
            a_team_source.plot(save_file=os.path.join(self.plot_folder, f'ateam{i}.png'))

        celestial_facet_models = []
        celestial_facet_models.append(
            FacetModel(
                point_source_model=bright_point_sources,
                gaussian_source_model=None,
                rfi_emitter_source_model=None,
                fits_source_model=None,
                gain_model=beam_gain_model,
                near_field_delay_engine=ms.near_field_delay_engine,
                far_field_delay_engine=ms.far_field_delay_engine,
                geodesic_model=ms.geodesic_model,
                convention=ms.meta.convention,
                dtype=self.dtype
            )
        )
        # for a_team_source in ["cas_a", "cyg_a", "vir_a", "tau_a"]:
        for a_team_source in a_team_sources:
            # source_model_asset = source_model_registry.get_instance(source_model_registry.get_match(a_team_source))
            celestial_facet_models.append(
                FacetModel(
                    point_source_model=None,
                    gaussian_source_model=None,
                    rfi_emitter_source_model=None,
                    fits_source_model=a_team_source,
                    gain_model=beam_gain_model,
                    near_field_delay_engine=ms.near_field_delay_engine,
                    far_field_delay_engine=ms.far_field_delay_engine,
                    geodesic_model=ms.geodesic_model,
                    convention=ms.meta.convention,
                    dtype=self.dtype
                )
            )

        # Give RFI just dish effects, not ionosphere
        rfi_facet_models = []
        for rfi_emitter_source in rfi_emitter_sources:
            rfi_facet_models.append(
                FacetModel(
                    point_source_model=None,
                    gaussian_source_model=None,
                    rfi_emitter_source_model=rfi_emitter_source,
                    fits_source_model=None,
                    gain_model=None,  # TODO: check beam is correct, because on horizon the beam is zero
                    near_field_delay_engine=ms.near_field_delay_engine,
                    far_field_delay_engine=ms.far_field_delay_engine,
                    geodesic_model=ms.geodesic_model,
                    convention=ms.meta.convention,
                    dtype=self.dtype
                )
            )

        rime_model = RIMEModel(
            facet_models=rfi_facet_models# + celestial_facet_models
        )

        simulator = SimulateVisibilities(
            rime_model=rime_model,
            verbose=self.verbose,
            seed=self.simulation_seed,
            num_shards=self.num_shards,
            plot_folder=os.path.join(self.plot_folder, 'simulate'),
            add_noise=self.add_noise
        )
        simulator.simulate(
            ms=ms
        )

    def _calibrate_visibilities(self, ms: MeasurementSet, beam_gain_model: GainModel) -> MeasurementSet:
        """
        Calibrate visibilities using the RIME model.

        Args:
            ms: the measurement set to store the results

        Returns:
            subtracted_ms: the calibrated visibilities
        """

        gain_prior_model = DiagonalUnconstrainedGain()
        rfi_emitter_sources = self.synthetic_sky_model_producer.create_rfi_emitter_sources(
            full_stokes=ms.is_full_stokes()
        )
        rfi_emitter_sources[0].plot(save_file=os.path.join(self.plot_folder, 'rfi_emitter_sources.png'))
        a_team_sources = self.synthetic_sky_model_producer.create_a_team_sources(
            full_stokes=ms.is_full_stokes()
        )
        for i, a_team_source in enumerate(a_team_sources):
            a_team_source.plot(save_file=os.path.join(self.plot_folder, f'ateam{i}.png'))

        celestial_facet_models = []
        for a_team_source in a_team_sources:
            celestial_facet_models.append(
                FacetModel(
                    point_source_model=None,
                    gaussian_source_model=None,
                    rfi_emitter_source_model=None,
                    fits_source_model=a_team_source,
                    gain_model=beam_gain_model,
                    near_field_delay_engine=ms.near_field_delay_engine,
                    far_field_delay_engine=ms.far_field_delay_engine,
                    geodesic_model=ms.geodesic_model,
                    convention=ms.meta.convention,
                    dtype=self.dtype
                )
            )

        # Give RFI just dish effects, not ionosphere
        rfi_facet_models = []
        for rfi_emitter_source in rfi_emitter_sources:
            rfi_facet_models.append(
                FacetModel(
                    point_source_model=None,
                    gaussian_source_model=None,
                    rfi_emitter_source_model=rfi_emitter_source,
                    fits_source_model=None,
                    gain_model=None,  # beam_gain_model,
                    near_field_delay_engine=ms.near_field_delay_engine,
                    far_field_delay_engine=ms.far_field_delay_engine,
                    geodesic_model=ms.geodesic_model,
                    convention=ms.meta.convention,
                    dtype=self.dtype
                )
            )

        rime_model = RIMEModel(
            facet_models=rfi_facet_models  # + celestial_facet_models
        )

        gains_per_facet = GainsPerFacet(
            gain_prior_model=gain_prior_model,
            rime_model=rime_model
        )

        # TODO: add rfi model

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
            convention=ms.meta.convention,
            weighting=self.weighting
        )
        return imagor.image(image_name=image_name, ms=ms)
