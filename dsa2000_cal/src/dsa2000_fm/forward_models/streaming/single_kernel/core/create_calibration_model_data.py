import dataclasses
from typing import NamedTuple, Tuple, Any

from tomographic_kernel.tomographic_kernel import TomographicKernel

from dsa2000_cal.common.array_types import FloatArray
from dsa2000_fm.forward_models.streaming.single_kernel.abc import AbstractCoreStep
from dsa2000_common.gain_models import BaseSphericalInterpolatorGainModel


class CreateCalibrationModelDataState(NamedTuple):
    window_dtec: FloatArray  # [window_size, num_model_ant, num_model_dir]
    window_geodesics: FloatArray  # [window_size, num_model_ant, num_model_dir, 3]
    tomographic_kernel: TomographicKernel


class CreateCalibrationModelDataOutput(NamedTuple):
    gain_model: BaseSphericalInterpolatorGainModel


@dataclasses.dataclass(eq=False)
class CreateCalibrationModelDataStep(AbstractCoreStep[CreateCalibrationModelDataState, CreateCalibrationModelDataOutput, None]):
    def get_state(self) -> CreateCalibrationModelDataState:
        pass

    def step(self, primals: Tuple[Any, ...]) -> \
            Tuple[CreateCalibrationModelDataState, CreateCalibrationModelDataOutput, None]:
        pass
