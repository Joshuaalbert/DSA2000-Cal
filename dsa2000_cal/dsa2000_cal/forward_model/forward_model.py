import dataclasses
from typing import List

import astropy.units as au
import jax
import jax.numpy as jnp
from jax._src.typing import SupportsDType
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.common.fits_utils import ImageModel
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.gain_models.dish_effects_gain_model import DishEffectsGainModelParams
from dsa2000_cal.gain_models.gain_model import ProductGainModel
from dsa2000_cal.imaging.dirty_imaging import DirtyImaging
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet, VisibilityData
from dsa2000_cal.simulation.simulate_systematics import SimulateSystematics
from dsa2000_cal.simulation.simulate_visibilties import SimulateVisibilities
from dsa2000_cal.source_models.fits_stokes_I_source_model import FitsStokesISourceModel
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel


@dataclasses.dataclass(eq=False)
class ForwardModel:
    """
    Runs forward modelling using a sharded data structure over devices.
    """
    ms: MeasurementSet
    simulation_wsclean_source_models: List[WSCleanSourceModel]
    simulation_fits_source_models: List[FitsStokesISourceModel]

    # Dish effect model parameters
    dish_effect_params: DishEffectsGainModelParams

    # Ionosphere model parameters
    ionosphere_specification: SPECIFICATION

    # Calibration parameters
    calibration_wsclean_source_models: List[WSCleanSourceModel]
    calibration_fits_source_models: List[FitsStokesISourceModel]

    # Imaging parameters
    field_of_view: au.Quantity

    plot_folder: str
    cache_folder: str
    ionosphere_seed: int
    dish_effects_seed: int
    seed: int

    oversample_factor: float = 2.5
    epsilon: float = 1e-4
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64
    verbose: bool = False

    def __post_init__(self):
        if not self.field_of_view.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected field_of_view to be in degrees, got {self.field_of_view.unit}")
        self.beam_gain_model = beam_gain_model_factory(self.ms.meta.array_name)
        self.key = jax.random.PRNGKey(self.seed)

    def forward(self):
        # Simulate systematics
        system_gain_model = self._simulate_systematics(
            ms=self.ms
        )
        # Simulate visibilities
        self._simulate_visibilities(
            ms=self.ms,
            system_gain_model=system_gain_model
        )
        # Calibrate visibilities
        self._calibrate_visibilities(
            ms=self.ms
        )
        # Subtract visibilities
        subtracted_ms = self._subtract_visibilities()
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
            verbose=self.verbose
        )
        simulator.simulate(
            ms=ms,
            system_gain_model=system_gain_model
        )

    def _calibrate_visibilities(self, ms: MeasurementSet):
        calibration = Calibration(
            num_iterations=15,
            wsclean_source_models=self.calibration_wsclean_source_models,
            fits_source_models=self.calibration_fits_source_models,
            preapply_gain_model=self.beam_gain_model,
            inplace_subtract=True,
            average_interval=None,
            solution_cadence=None,
            verbose=self.verbose
        )

        calibration.calibrate(ms=ms)

    def _subtract_visibilities(self) -> MeasurementSet:
        subtracted_ms = self.ms.clone(ms_folder='subtracted_ms')
        # Predict model with calibrated gains
        gen = subtracted_ms.create_block_generator(vis=True, weights=True, flags=False)
        gen_response = None
        while True:
            try:
                time, visibility_coords, data = gen.send(gen_response)
            except StopIteration:
                break

            # Predict model with calibrated gains
            vis_model = ...
            residual = data.vis - vis_model
            residual_variance = ...

            # Assuming weights represent 1/variance, we should add to the variance to account for uncertainty in the calibration
            # weights = 1 / (1 / weights + 1 / calibration_variance)
            weights = data.weights * residual_variance / (data.weights + residual_variance)

            # Subtract model from data
            gen_response = VisibilityData(
                vis=residual,
                weights=weights
            )

    def _image_visibilities(self, image_name: str, ms: MeasurementSet) -> ImageModel:
        imagor = DirtyImaging(
            plot_folder=self.plot_folder,
            cache_folder=self.cache_folder,
            field_of_view=self.field_of_view
        )
        return imagor.image(image_name=image_name, ms=ms)
