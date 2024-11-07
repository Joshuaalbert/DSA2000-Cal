import dataclasses
from typing import NamedTuple, Tuple, Any

from tomographic_kernel.tomographic_kernel import TomographicKernel

from dsa2000_cal.common.array_types import FloatArray
from dsa2000_cal.forward_models.streaming.abc import AbstractCoreStep
from dsa2000_cal.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel


class CalibrateState(NamedTuple):
    window_dtec: FloatArray  # [window_size, num_model_ant, num_model_dir]
    window_geodesics: FloatArray  # [window_size, num_model_ant, num_model_dir, 3]
    tomographic_kernel: TomographicKernel


class CalibrateOutput(NamedTuple):
    gain_model: BaseSphericalInterpolatorGainModel

class CalibrateKeep(NamedTuple):
    gain_model: BaseSphericalInterpolatorGainModel

@dataclasses.dataclass(eq=False)
class CalibrateStep(AbstractCoreStep[CalibrateState, CalibrateOutput, CalibrateKeep]):
    def get_state(self) -> CalibrateState:
        pass

    def step(self, primals: Tuple[Any, ...]) -> Tuple[CalibrateState, CalibrateOutput, CalibrateKeep]:
        state = self.get_state()
        pass
