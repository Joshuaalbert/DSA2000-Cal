import dataclasses
from typing import List

from dsa2000_cal.source_models.fits_stokes_I_source_model import FitsStokesISourceModel
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel


@dataclasses.dataclass(eq=False)
class SkyModel:
    component_models: List[WSCleanSourceModel]
    fits_models: List[FitsStokesISourceModel]

    @property
    def num_sources(self) -> int:
        return len(self.component_models) + len(self.fits_models)
