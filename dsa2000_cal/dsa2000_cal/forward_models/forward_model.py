import dataclasses
import os
from abc import abstractmethod, ABC
from typing import List

import jax
from astropy import units as au
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.calibration.probabilistic_models.probabilistic_model import AbstractProbabilisticModel
from dsa2000_cal.common.alert_utils import post_completed_forward_modelling_run
from dsa2000_cal.common.datetime_utils import current_utc
from dsa2000_cal.forward_models.simulation.simulate_systematics import SimulateSystematics
from dsa2000_cal.forward_models.simulation.simulate_visibilties import SimulateVisibilities
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsParams
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.imaging.imagor import Imagor
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet, beam_gain_model_factory
from dsa2000_cal.visibility_model.rime_model import RIMEModel


@dataclasses.dataclass(eq=False)
class AbstractForwardModel(ABC):
    @abstractmethod
    def forward(self, ms: MeasurementSet):
        """
        Run forward modelling. Will perform the steps of simulating systematics, simulating visibilities, calibration,
        subtraction and imaging.

        Args:
            ms: the measurement set to fill and use.
        """
        ...


class BaseForwardModel(AbstractForwardModel):
    """
    Runs forward modelling using a sharded data structure over devices.

    Args:
        dish_effect_params: the dish effect model parameters
        ionosphere_specification: the ionosphere model specification, see tomographic_kernel.models.cannonical_models
        run_folder: the folder to store plots and caches
        ionosphere_seed: the seed for the ionosphere model
        dish_effects_seed: the seed for the dish effects model
        simulation_seed: the seed for the simulation
        calibration_seed: the seed for the calibration
        imaging_seed: the seed for the imaging
        field_of_view: the field of view for imaging, default computes from dish model
        oversample_factor: the oversample factor for imaging, default 2.5
        epsilon: the epsilon for wgridder, default 1e-4
        verbose: the verbosity for imaging, default False
    """

    def __init__(self,
                 run_folder: str,
                 add_noise: bool,
                 include_ionosphere: bool,
                 include_dish_effects: bool,
                 include_simulation: bool,
                 include_calibration: bool,
                 dish_effect_params: DishEffectsParams | None,
                 ionosphere_specification: SPECIFICATION | None,
                 num_cal_iters: int,
                 solution_interval: au.Quantity | None,
                 validity_interval: au.Quantity | None,
                 field_of_view: au.Quantity | None,
                 oversample_factor: float,
                 weighting: str,
                 epsilon: float,
                 verbose: bool,
                 num_shards: int,
                 ionosphere_seed: int,
                 dish_effects_seed: int,
                 simulation_seed: int,
                 calibration_seed: int,
                 imaging_seed: int
                 ):
        self.run_folder = run_folder
        self.add_noise = add_noise
        self.include_ionosphere = include_ionosphere
        self.include_dish_effects = include_dish_effects
        self.include_calibration = include_calibration
        self.dish_effect_params = dish_effect_params
        self.ionosphere_specification = ionosphere_specification
        self.num_cal_iters = num_cal_iters
        self.solution_interval = solution_interval
        self.validity_interval = validity_interval
        self.field_of_view = field_of_view
        self.oversample_factor = oversample_factor
        self.weighting = weighting
        self.epsilon = epsilon
        self.verbose = verbose
        self.num_shards = num_shards
        self.ionosphere_seed = ionosphere_seed
        self.dish_effects_seed = dish_effects_seed
        self.simulation_seed = simulation_seed
        self.include_simulation = include_simulation
        self.calibration_seed = calibration_seed
        self.imaging_seed = imaging_seed

        self.plot_folder = os.path.join(self.run_folder, 'plots')
        self.solution_folder = os.path.join(self.run_folder, 'solution')
        self.cache_folder = os.path.join(self.run_folder, 'cache')
        os.makedirs(self.plot_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)
        os.makedirs(self.solution_folder, exist_ok=True)

    def forward(self, ms: MeasurementSet):
        """
        Run forward modelling. Will perform the steps of simulating systematics, simulating visibilities, calibration,
        subtraction and imaging.

        Args:
            ms: the measurement set to fill and use.
        """
        start_time = current_utc()

        beam_gain_model = beam_gain_model_factory(ms)
        beam_gain_model.plot_regridded_beam(os.path.join(self.plot_folder, 'beam.png'))

        # Simulate systematics
        systematics_simulator = SimulateSystematics(
            dish_effect_params=self.dish_effect_params,
            ionosphere_specification=self.ionosphere_specification,
            plot_folder=self.plot_folder,
            cache_folder=self.cache_folder,
            ionosphere_seed=self.ionosphere_seed,
            dish_effects_seed=self.dish_effects_seed,
            verbose=self.verbose,
            full_stokes=ms.is_full_stokes()
        )

        imagor = Imagor(
            plot_folder=os.path.join(self.plot_folder, 'imaging'),
            field_of_view=self.field_of_view,
            seed=self.imaging_seed,
            oversample_factor=self.oversample_factor,
            convention=ms.meta.convention,
            weighting=self.weighting
        )

        if self.include_dish_effects:
            systematics_gain_model = systematics_simulator.simulate_dish_effects(ms=ms)
            horizon_gain_model = None
        else:
            systematics_gain_model = beam_gain_model
            horizon_gain_model = None

        if self.include_ionosphere:
            ionosphere_gain_model = systematics_simulator.simulate_ionosphere(ms=ms)
            systematics_gain_model = systematics_gain_model @ ionosphere_gain_model

        if self.include_simulation:
            # Simulate visibilities
            simulator = SimulateVisibilities(
                rime_model=self._build_simulation_rime_model(
                    ms=ms,
                    system_gain_model=systematics_gain_model,
                    horizon_gain_model=horizon_gain_model
                ),
                verbose=self.verbose,
                seed=self.simulation_seed,
                num_shards=self.num_shards,
                plot_folder=os.path.join(self.plot_folder, 'simulate'),
                add_noise=self.add_noise
            )
            simulator.simulate(
                ms=ms
            )

            # Image visibilities
            imagor.image(image_name='dirty_image', ms=ms)

            # Create psf
            imagor.image(image_name='psf', ms=ms, psf=True)

        if self.include_calibration:
            # Calibrate visibilities
            calibration = Calibration(
                probabilistic_models=self._build_calibration_probabilistic_models(
                    ms=ms,
                    a_priori_system_gain_model=systematics_gain_model,
                    a_priori_horizon_gain_model=horizon_gain_model
                ),
                num_iterations=self.num_cal_iters,
                num_approx_steps=0,
                inplace_subtract=False,
                residual_ms_folder='residual_ms',
                solution_interval=self.solution_interval,
                validity_interval=self.validity_interval,
                verbose=self.verbose,
                seed=self.calibration_seed,
                plot_folder=os.path.join(self.plot_folder, 'calibration'),
                solution_folder=self.solution_folder
            )
            subtracted_ms = calibration.calibrate(ms=ms)

            # Image subtracted visibilities
            imagor.image(image_name='subtracted_dirty_image', ms=subtracted_ms)

        # Tell Slack we're done
        post_completed_forward_modelling_run(
            run_dir=os.getcwd(),
            start_time=start_time,
            duration=current_utc() - start_time
        )

    @abstractmethod
    def _build_simulation_rime_model(self,
                                     ms: MeasurementSet,
                                     system_gain_model: GainModel | None,
                                     horizon_gain_model: GainModel | None
                                     ) -> RIMEModel:
        """
        Build the RIME model for simulation.

        Args:
            ms: the measurement set to use.
            system_gain_model: the system gain model to use.
            horizon_gain_model: the horizon gain model to use, i.e. excluding the sky.

        Returns:
            the RIME model to use for simulation.
        """
        ...

    @abstractmethod
    def _build_calibration_probabilistic_models(
            self,
            ms: MeasurementSet,
            a_priori_system_gain_model: GainModel | None,
            a_priori_horizon_gain_model: GainModel | None
    ) -> List[AbstractProbabilisticModel]:
        """
        Build the probabilistic models for calibration.

        Args:
            ms: the measurement set to use.
            a_priori_system_gain_model: the a priori system gain model to use.
            a_priori_horizon_gain_model: the a priori horizon gain model to use.

        Returns:
            the probabilistic models for calibration.
        """
        ...
