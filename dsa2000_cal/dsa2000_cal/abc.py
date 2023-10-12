from abc import ABC, abstractmethod

import numpy as np
from pydantic import Field

from dsa2000_cal.utils import SerialisableBaseModel


# Creates an empty Measurement Set using simms and then does a DFT prediction of visibilities for given sky model.

class SourceModel(SerialisableBaseModel):
    image: np.ndarray = Field(
        description="Source model of shape [source, chan, corr]",
    )
    lm: np.ndarray = Field(
        description="Source direction cosines of shape [source, 2]",
    )
    corrs: list[str] = Field(
        description="Correlations in the source model",
        default=['XX', 'XY', 'YX', 'YY'],
    )


class AbstractSkyModel(ABC):
    @abstractmethod
    def get_source(self) -> SourceModel:
        ...
