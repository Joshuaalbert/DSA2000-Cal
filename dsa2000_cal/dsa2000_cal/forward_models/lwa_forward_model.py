import dataclasses
import os
from typing import List

import astropy.units as au
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import DiagonalUnconstrainedGain, \
    ScalarUnconstrainedGain
from dsa2000_cal.calibration.probabilistic_models.gains_per_facet_model import GainsPerFacet
from dsa2000_cal.calibration.probabilistic_models.probabilistic_model import AbstractProbabilisticModel
from dsa2000_cal.forward_models.forward_model import BaseForwardModel
from dsa2000_cal.forward_models.synthetic_sky_model.synthetic_sky_model_producer import SyntheticSkyModelProducer
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsParams
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet
from dsa2000_cal.visibility_model.facet_model import FacetModel
from dsa2000_cal.visibility_model.rime_model import RIMEModel


@dataclasses.dataclass(eq=False)
class LWAForwardModel(BaseForwardModel):
    """
    Runs forward modelling using a sharded data structure over devices.

    Args:
        synthetic_sky_model_producer: the synthetic sky model producer
    """

    # Synthetic sky model producer
    synthetic_sky_model_producer: SyntheticSkyModelProducer

    # Overridden params
    run_folder: str
    add_noise: bool = True
    include_ionosphere: bool = False
    include_dish_effects: bool = False
    include_simulation: bool = True
    include_calibration: bool = False
    dish_effect_params: DishEffectsParams | None = None
    ionosphere_specification: SPECIFICATION | None = None
    num_cal_iters: int = 15
    solution_interval: au.Quantity | None = None
    validity_interval: au.Quantity | None = None
    field_of_view: au.Quantity | None = None
    oversample_factor: float = 5.
    weighting: str = 'natural'
    epsilon: float = 1e-4
    verbose: bool = False
    num_shards: int = 1
    ionosphere_seed: int = 42
    dish_effects_seed: int = 4242
    simulation_seed: int = 42424242
    calibration_seed: int = 4242424242
    imaging_seed: int = 424242424242

    def __post_init__(self):
        BaseForwardModel.__init__(
            self,
            run_folder=self.run_folder,
            add_noise=self.add_noise,
            include_ionosphere=self.include_ionosphere,
            include_dish_effects=self.include_dish_effects,
            include_simulation=self.include_simulation,
            include_calibration=self.include_calibration,
            dish_effect_params=self.dish_effect_params,
            ionosphere_specification=self.ionosphere_specification,
            num_cal_iters=self.num_cal_iters,
            solution_interval=self.solution_interval,
            validity_interval=self.validity_interval,
            field_of_view=self.field_of_view, oversample_factor=self.oversample_factor,
            weighting=self.weighting, epsilon=self.epsilon,
            verbose=self.verbose, num_shards=self.num_shards,
            ionosphere_seed=self.ionosphere_seed, dish_effects_seed=self.dish_effects_seed,
            simulation_seed=self.simulation_seed, calibration_seed=self.calibration_seed,
            imaging_seed=self.imaging_seed
        )

    def _build_simulation_rime_model(self,
                                     ms: MeasurementSet,
                                     system_gain_model: GainModel | None,
                                     horizon_gain_model: GainModel | None
                                     ) -> RIMEModel:
        # Construct single RIME model with:
        # 1. Bright sources over whole sky (filtered to those above horizon) with system gain model
        # 2. A-team sources (those above horizon) with system gain model
        # 3. RFI sources (near horizon) with horizon gain model (excluding sky)
        bright_point_sources = self.synthetic_sky_model_producer.create_sources_outside_fov(
            full_stokes=ms.is_full_stokes()
        )
        bright_point_sources.plot(save_file=os.path.join(self.plot_folder, 'bright_point_sources.png'))
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
                gain_model=system_gain_model,
                near_field_delay_engine=ms.near_field_delay_engine,
                far_field_delay_engine=ms.far_field_delay_engine,
                geodesic_model=ms.geodesic_model,
                convention=ms.meta.convention
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
                    gain_model=system_gain_model,
                    near_field_delay_engine=ms.near_field_delay_engine,
                    far_field_delay_engine=ms.far_field_delay_engine,
                    geodesic_model=ms.geodesic_model,
                    convention=ms.meta.convention
                )
            )

        # Give RFI just dish effects, not ionosphere
        rfi_emitter_sources = self.synthetic_sky_model_producer.create_rfi_emitter_sources(
            rfi_sources=['lwa_cell_tower'],
            full_stokes=ms.is_full_stokes()
        )
        rfi_emitter_sources[0].plot(save_file=os.path.join(self.plot_folder, 'rfi_emitter_sources.png'))
        rfi_facet_models = []
        for rfi_emitter_source in rfi_emitter_sources:
            rfi_facet_models.append(
                FacetModel(
                    point_source_model=None,
                    gaussian_source_model=None,
                    rfi_emitter_source_model=rfi_emitter_source,
                    fits_source_model=None,
                    gain_model=horizon_gain_model,
                    near_field_delay_engine=ms.near_field_delay_engine,
                    far_field_delay_engine=ms.far_field_delay_engine,
                    geodesic_model=ms.geodesic_model,
                    convention=ms.meta.convention
                )
            )

        rime_model = RIMEModel(
            facet_models=rfi_facet_models + celestial_facet_models
        )
        return rime_model

    def _build_calibration_probabilistic_models(
            self,
            ms: MeasurementSet,
            a_priori_system_gain_model: GainModel | None,
            a_priori_horizon_gain_model: GainModel | None
    ) -> List[AbstractProbabilisticModel]:
        # Construct single gain model per:
        # 1. a-team source
        a_team_sources = self.synthetic_sky_model_producer.create_a_team_sources(
            full_stokes=ms.is_full_stokes()
        )

        # Note: if a-team added then make sure to create separate facet for each to give each its own gain evaluation.
        celestial_facet_models = [
            FacetModel(
                point_source_model=None,
                gaussian_source_model=None,
                rfi_emitter_source_model=None,
                fits_source_model=a_team_source,
                gain_model=a_priori_system_gain_model,
                near_field_delay_engine=ms.near_field_delay_engine,
                far_field_delay_engine=ms.far_field_delay_engine,
                geodesic_model=ms.geodesic_model,
                convention=ms.meta.convention
            ) for a_team_source in a_team_sources
        ]

        rime_model = RIMEModel(
            facet_models=celestial_facet_models
        )

        if ms.is_full_stokes():
            gain_prior_model = DiagonalUnconstrainedGain()
        else:
            gain_prior_model = ScalarUnconstrainedGain()

        gains_per_facet = GainsPerFacet(
            gain_prior_model=gain_prior_model,
            rime_model=rime_model
        )
        # TODO: Construct RFI parameterisation per RFI source.

        probabilistic_models = [gains_per_facet]

        return probabilistic_models
