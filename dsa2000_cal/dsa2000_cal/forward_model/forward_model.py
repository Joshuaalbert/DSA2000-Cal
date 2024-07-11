import dataclasses
import os

import astropy.units as au
import jax
import jax.numpy as jnp
from jax._src.typing import SupportsDType
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.common.alert_utils import post_completed_forward_modelling_run
from dsa2000_cal.common.datetime_utils import current_utc
from dsa2000_cal.common.fits_utils import ImageModel
from dsa2000_cal.forward_model.simulation.simulate_systematics import SimulateSystematics
from dsa2000_cal.forward_model.simulation.simulate_visibilties import SimulateVisibilities
from dsa2000_cal.forward_model.systematics.dish_effects_simulation import DishEffectsParams
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.gain_models.gain_model import ProductGainModel
from dsa2000_cal.imaging.dirty_imaging import DirtyImaging
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet
from dsa2000_cal.uvw.near_field import NearFieldDelayEngine
from dsa2000_cal.visibility_model.rime_model import FacetSourceModel, RIMEModel
from dsa2000_cal.visibility_model.source_models.celestial.celestial_source_model import CelestialPredict
from dsa2000_cal.visibility_model.source_models.rfi.lte_source.lte_source_model import LTEPredict
from dsa2000_cal.visibility_model.source_models.rfi.rfi_source_model import RFIPredict


@dataclasses.dataclass(eq=False)
class ForwardModel:
    """
    Runs forward modelling using a sharded data structure over devices.

    Args:
        sky_model: the sky model
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

    # Simulation parameters
    sky_model: FacetSourceModel

    # Dish effect model parameters
    dish_effect_params: DishEffectsParams

    # Ionosphere model parameters
    ionosphere_specification: SPECIFICATION

    # Calibration parameters
    calibration_sky_model: FacetSourceModel

    # Plot and cache folders
    plot_folder: str
    cache_folder: str

    # Seeds
    ionosphere_seed: int = 42
    dish_effects_seed: int = 42
    simulation_seed: int = 424242
    calibration_seed: int = 42424242
    imaging_seed: int = 4242424242

    # Imaging parameters
    field_of_view: au.Quantity | None = None
    oversample_factor: float = 1.5

    # Wgridder parameters
    epsilon: float = 1e-4

    # Common parameters
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64
    verbose: bool = False
    num_shards: int = 1

    def forward(self, ms: MeasurementSet):
        start_time = current_utc()

        celestial_predict = CelestialPredict(
            convention=self.convention,
            dtype=self.dtype
        )
        rfi_predict = RFIPredict(
            lte_predict=LTEPredict(
                delay_engine=NearFieldDelayEngine(
                    antennas=ms.meta.antennas,
                    start_time=ms.meta.times[0],
                    end_time=ms.meta.times[-1]
                )
            )
        )
        rime_model = RIMEModel(
            celestial_predict=celestial_predict,
            rfi_predict=rfi_predict
        )

        # Simulate systematics
        system_gain_model = self._simulate_systematics(
            ms=ms
        )
        # Simulate visibilities
        self._simulate_visibilities(
            ms=ms,
            system_gain_model=system_gain_model,
            rime_model=rime_model
        )

        # Calibrate visibilities
        subtracted_ms = self._calibrate_visibilities(
            ms=ms,
            rime_model=rime_model
        )

        # Image visibilities
        self._image_visibilities(image_name='dirty_image', ms=ms)

        # Image subtracted visibilities
        self._image_visibilities(image_name='subtracted_dirty_image', ms=subtracted_ms)

        # Tell Slack we're done
        post_completed_forward_modelling_run(
            run_dir=os.getcwd(),
            start_time=start_time,
            duration=current_utc() - start_time
        )

    def _simulate_systematics(self, ms: MeasurementSet) -> ProductGainModel:
        """
        Simulate systematics such as ionosphere and dish effects.

        Returns:
            system_gain_model: the system gain model
        """

        simulator = SimulateSystematics(
            dish_effect_params=self.dish_effect_params,
            ionosphere_specification=self.ionosphere_specification,
            plot_folder=self.plot_folder,
            cache_folder=self.cache_folder,
            ionosphere_seed=self.ionosphere_seed,
            dish_effects_seed=self.dish_effects_seed,
            verbose=self.verbose
        )

        # Order is by right multiplication of systematics encountered by radiation from source to observer
        system_gain_model = simulator.simulate(ms=ms)

        return system_gain_model

    def _simulate_visibilities(self, ms: MeasurementSet, system_gain_model: ProductGainModel, rime_model: RIMEModel):
        """
        Simulate visibilities using the system gain model, and store the results in the MS.

        Args:
            ms: the measurement set to store the results
            system_gain_model: the system gain model
            rime_model: the RIME model
        """
        simulator = SimulateVisibilities(
            sky_model=self.sky_model,
            rime_model=rime_model,
            verbose=self.verbose,
            seed=self.simulation_seed,
            num_shards=self.num_shards,
            plot_folder=os.path.join(self.plot_folder, 'simulate')
        )
        simulator.simulate(
            ms=ms,
            system_gain_model=system_gain_model
        )

    def _calibrate_visibilities(self, ms: MeasurementSet, rime_model: RIMEModel) -> MeasurementSet:
        """
        Calibrate visibilities using the RIME model.

        Args:
            ms: the measurement set to store the results
            rime_model: the RIME model

        Returns:
            subtracted_ms: the calibrated visibilities
        """
        beam_gain_model = beam_gain_model_factory(ms.meta.array_name)
        calibration = Calibration(
            num_iterations=15,
            sky_model=self.calibration_sky_model,
            rime_model=rime_model,
            gain_prior_model=gain_prior_model,
            preapply_gain_model=beam_gain_model,
            inplace_subtract=False,
            residual_ms_folder='residual_ms',
            solution_interval=None,
            validity_interval=None,
            verbose=self.verbose,
            seed=self.calibration_seed,
            num_shards=self.num_shards,
            plot_folder=os.path.join(self.plot_folder, 'calibration')
        )

        return calibration.calibrate(ms=ms)

    def _image_visibilities(self, image_name: str, ms: MeasurementSet) -> ImageModel:
        imagor = DirtyImaging(
            plot_folder=os.path.join(self.plot_folder, 'imaging'),
            field_of_view=self.field_of_view,
            seed=self.imaging_seed,
            oversample_factor=self.oversample_factor,
            nthreads=len(jax.devices())
        )
        return imagor.image(image_name=image_name, ms=ms)
