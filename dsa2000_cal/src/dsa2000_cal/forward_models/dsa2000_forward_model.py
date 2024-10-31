import dataclasses
import os
from typing import List

import astropy.units as au
from jax._src.typing import SupportsDType
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import DiagonalUnconstrainedGain
from dsa2000_cal.calibration.probabilistic_models.gains_per_facet_model import GainsPerFacet
from dsa2000_cal.calibration.probabilistic_models.probabilistic_model import AbstractProbabilisticModel
from dsa2000_cal.common.mixed_precision_utils import complex_type
from dsa2000_cal.forward_models.forward_model import BaseForwardModel
from dsa2000_cal.forward_models.synthetic_sky_model import SyntheticSkyModelProducer
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsParams
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.measurement_sets import MeasurementSet
from dsa2000_cal.visibility_model.facet_model import FacetModel
from dsa2000_cal.visibility_model.rime_model import RIMEModel


@dataclasses.dataclass(eq=False)
class DSA2000ForwardModel(BaseForwardModel):
    """
    Runs forward modelling using a sharded data structure over devices.

    Args:
        synthetic_sky_model_producer: the synthetic sky model producer
    """

    # Synthetic sky model producer
    synthetic_sky_model_producer: SyntheticSkyModelProducer | None = None

    # Overridden params
    run_folder: str
    add_noise: bool = True
    include_ionosphere: bool = False
    include_dish_effects: bool = False
    include_simulation: bool = True
    include_calibration: bool = False
    dish_effect_params: DishEffectsParams | None = None
    ionosphere_specification: SPECIFICATION = 'light_dawn'
    num_cal_iters: int = 15
    solution_interval: au.Quantity | None = None
    validity_interval: au.Quantity | None = None
    field_of_view: au.Quantity | None = None
    oversample_factor: float = 5.
    weighting: str = 'natural'
    epsilon: float = 1e-4
    dtype: SupportsDType = complex_type
    verbose: bool = False
    num_shards: int = 1
    ionosphere_seed: int = 42
    dish_effects_seed: int = 4242
    simulation_seed: int = 42424242
    calibration_seed: int = 4242424242
    imaging_seed: int = 424242424242
    overwrite: bool = False

    def __post_init__(self):
        if self.synthetic_sky_model_producer is None:
            raise ValueError("synthetic_sky_model_producer must be provided")

        super().__post_init__()

    def _build_simulation_rime_model(self,
                                     ms: MeasurementSet,
                                     system_gain_model: GainModel | None,
                                     horizon_gain_model: GainModel | None
                                     ) -> RIMEModel:
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

        # Note: if a-team added then make sure to create separate facet for each to give each its own gain evaluation.
        celestial_facet_models = [
            FacetModel(
                point_source_model=bright_point_sources,
                gaussian_source_model=None,
                rfi_emitter_source_model=None,
                fits_source_model=None,
                gain_model=system_gain_model,
                near_field_delay_engine=ms.near_field_delay_engine,
                far_field_delay_engine=ms.far_field_delay_engine,
                geodesic_model=ms.geodesic_model,
                convention=ms.meta.convention,
                dtype=self.dtype
            ),
            FacetModel(
                point_source_model=inner_point_sources,
                gaussian_source_model=None,
                rfi_emitter_source_model=None,
                fits_source_model=None,
                gain_model=system_gain_model,
                near_field_delay_engine=ms.near_field_delay_engine,
                far_field_delay_engine=ms.far_field_delay_engine,
                geodesic_model=ms.geodesic_model,
                convention=ms.meta.convention,
                dtype=self.dtype
            ),
            FacetModel(
                point_source_model=None,
                gaussian_source_model=inner_diffuse_sources,
                rfi_emitter_source_model=None,
                fits_source_model=None,
                gain_model=system_gain_model,
                near_field_delay_engine=ms.near_field_delay_engine,
                far_field_delay_engine=ms.far_field_delay_engine,
                geodesic_model=ms.geodesic_model,
                convention=ms.meta.convention,
                dtype=self.dtype
            )
        ]

        # # Give RFI just dish effects, not ionosphere
        # rfi_emitter_sources = self.synthetic_sky_model_producer.create_rfi_emitter_sources(
        #     full_stokes=ms.is_full_stokes()
        # )
        # rfi_emitter_sources[0].plot(save_file=os.path.join(self.plot_folder, 'rfi_emitter_sources.png'))
        # rfi_facet_models = [
        #     FacetModel(
        #         point_source_model=None,
        #         gaussian_source_model=None,
        #         rfi_emitter_source_model=rfi_emitter_source,
        #         fits_source_model=None,
        #         gain_model=horizon_gain_model,
        #         near_field_delay_engine=ms.near_field_delay_engine,
        #         far_field_delay_engine=ms.far_field_delay_engine,
        #         geodesic_model=ms.geodesic_model,
        #         convention=ms.meta.convention,
        #         dtype=self.dtype
        #     )
        #     for rfi_emitter_source in rfi_emitter_sources
        # ]
        rime_model = RIMEModel(
            facet_models=celestial_facet_models  # + rfi_facet_models
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
                convention=ms.meta.convention,
                dtype=self.dtype
            ) for a_team_source in a_team_sources
        ]

        rime_model = RIMEModel(
            facet_models=celestial_facet_models
        )

        gain_prior_model = DiagonalUnconstrainedGain()

        gains_per_facet = GainsPerFacet(
            gain_prior_model=gain_prior_model,
            rime_model=rime_model
        )

        probabilistic_models = [gains_per_facet]

        return probabilistic_models
