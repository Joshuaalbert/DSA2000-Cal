import dataclasses
from typing import NamedTuple, Tuple, Any

from tomographic_kernel.tomographic_kernel import TomographicKernel

from dsa2000_cal.common.array_types import FloatArray
from dsa2000_fm.forward_models.streaming.single_kernel.abc import AbstractCoreStep
from dsa2000_common.gain_models import BaseSphericalInterpolatorGainModel


class ImageState(NamedTuple):
    window_dtec: FloatArray  # [window_size, num_model_ant, num_model_dir]
    window_geodesics: FloatArray  # [window_size, num_model_ant, num_model_dir, 3]
    tomographic_kernel: TomographicKernel


class ImageOutput(NamedTuple):
    gain_model: BaseSphericalInterpolatorGainModel

class ImageKeep(NamedTuple):
    image: FloatArray
    psf: FloatArray

@dataclasses.dataclass(eq=False)
class ImageStep(AbstractCoreStep[ImageState, ImageOutput, ImageKeep]):
    def get_state(self) -> ImageState:
        pass

    def step(self, primals: Tuple[Any, ...]) -> \
            Tuple[ImageState, ImageOutput, ImageKeep]:
        pass
