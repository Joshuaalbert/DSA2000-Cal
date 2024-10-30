import dataclasses
from typing import NamedTuple, Tuple, Any

import jax
from tomographic_kernel.tomographic_kernel import TomographicKernel

from dsa2000_cal.common.types import FloatArray
from dsa2000_cal.forward_models.streaming.abc import AbstractCoreStep
from dsa2000_cal.forward_models.streaming.types import StepData
from dsa2000_cal.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet


class PredictAndSampleState(NamedTuple):
    window_dtec: FloatArray  # [window_size, num_model_ant, num_model_dir]
    window_geodesics: FloatArray  # [window_size, num_model_ant, num_model_dir, 3]
    tomographic_kernel: TomographicKernel


class PredictAndSampleOutput(NamedTuple):
    gain_model: BaseSphericalInterpolatorGainModel


@dataclasses.dataclass(eq=False)
class PredictAndSampleStep(AbstractCoreStep[PredictAndSampleState, PredictAndSampleOutput, None]):
    def get_state(self) -> PredictAndSampleState:
        pass

    def step(self, primals: Tuple[Any, ...]) -> \
            Tuple[PredictAndSampleState, PredictAndSampleOutput, None]:
        pass
