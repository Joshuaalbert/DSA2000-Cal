import dataclasses
from typing import List

import astropy.units as au
import jax.numpy as jnp
from jax._src.typing import SupportsDType
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.common.fits_utils import ImageModel
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.gain_models.dish_effects_gain_model import DishEffectsGainModelParams
from dsa2000_cal.gain_models.gain_model import ProductGainModel
from dsa2000_cal.imaging.dirty_imaging import DirtyImaging
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet
from dsa2000_cal.simulation.simulate_systematics import SimulateSystematics
from dsa2000_cal.simulation.simulate_visibilties import SimulateVisibilities
from dsa2000_cal.source_models.fits_stokes_I_source_model import FitsStokesISourceModel
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel


@dataclasses.dataclass(eq=False)
class ForwardModel:
    """
    Runs forward modelling using a sharded data structure over devices.

    Args:
        simulation_wsclean_source_models: the source models to simulate visibilities
        simulation_fits_source_models: the source models to simulate visibilities
        dish_effect_params: the dish effect model parameters
        ionosphere_specification: the ionosphere model specification, see tomographic_kernel.models.cannonical_models
        calibration_wsclean_source_models: the source models to calibrate visibilities
        calibration_fits_source_models: the source models to calibrate visibilities
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
    simulation_wsclean_source_models: List[WSCleanSourceModel]
    simulation_fits_source_models: List[FitsStokesISourceModel]

    # Dish effect model parameters
    dish_effect_params: DishEffectsGainModelParams

    # Ionosphere model parameters
    ionosphere_specification: SPECIFICATION

    # Calibration parameters
    calibration_wsclean_source_models: List[WSCleanSourceModel]
    calibration_fits_source_models: List[FitsStokesISourceModel]

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
    oversample_factor: float = 2.5

    # Wgridder parameters
    epsilon: float = 1e-4

    # Common parameters
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64
    verbose: bool = False
    num_shards: int = 1

    def forward(self, ms: MeasurementSet):
        # Simulate systematics
        system_gain_model = self._simulate_systematics(
            ms=ms
        )
        # Simulate visibilities
        self._simulate_visibilities(
            ms=ms,
            system_gain_model=system_gain_model
        )
        # Calibrate visibilities
        subtracted_ms = self._calibrate_visibilities(
            ms=ms
        )
        # Image visibilities
        self._image_visibilities(image_name='dirty_image', ms=subtracted_ms)

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

    def _simulate_visibilities(self, ms: MeasurementSet, system_gain_model: ProductGainModel):
        """
        Simulate visibilities using the system gain model, and store the results in the MS.

        Args:
            ms: the measurement set to store the results
            system_gain_model: the system gain model
        """
        simulator = SimulateVisibilities(
            wsclean_source_models=self.simulation_wsclean_source_models,
            fits_source_models=self.simulation_fits_source_models,
            convention=self.convention,
            dtype=self.dtype,
            verbose=self.verbose,
            seed=self.simulation_seed,
            num_shards=self.num_shards
        )
        simulator.simulate(
            ms=ms,
            system_gain_model=system_gain_model
        )

    def _calibrate_visibilities(self, ms: MeasurementSet) -> MeasurementSet:
        beam_gain_model = beam_gain_model_factory(ms.meta.array_name)
        calibration = Calibration(
            num_iterations=15,
            wsclean_source_models=self.calibration_wsclean_source_models,
            fits_source_models=self.calibration_fits_source_models,
            preapply_gain_model=beam_gain_model,
            inplace_subtract=True,
            average_interval=None,
            solution_cadence=None,
            verbose=self.verbose,
            seed=self.calibration_seed,
            num_shards=self.num_shards
        )

        return calibration.calibrate(ms=ms)

    def _image_visibilities(self, image_name: str, ms: MeasurementSet) -> ImageModel:
        imagor = DirtyImaging(
            plot_folder=self.plot_folder,
            cache_folder=self.cache_folder,
            field_of_view=self.field_of_view,
            seed=self.imaging_seed
        )
        return imagor.image(image_name=image_name, ms=ms)
