from abc import ABC, abstractmethod
from typing import List

import numpy as np
from pydantic import Field

from dsa2000_cal.utils import SerialisableBaseModel


# Creates an empty Measurement Set using simms and then does a DFT prediction of visibilities for given sky model.

class SourceModel(SerialisableBaseModel):
    image: np.ndarray = Field(
        description="Source model of shape [source, chan, 2, 2]",
    )
    lm: np.ndarray = Field(
        description="Source direction cosines of shape [source, 2]",
    )
    corrs: List[List[str]] = Field(
        description="Correlations in the source model",
        default=[['XX', 'XY'], ['YX', 'YY']],
    )
    freqs: np.ndarray = Field(
        description="Frequencies of shape [chan]",
    )


class AbstractSkyModel(ABC):
    @abstractmethod
    def get_source(self) -> SourceModel:
        """
        Get the source model.

        Returns:
            source_model: SourceModel
        """
        ...
