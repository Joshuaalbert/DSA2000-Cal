from abc import ABC, abstractmethod

from dsa2000_cal.common.array_types import ComplexArray
from dsa2000_cal.common.types import VisibilityCoords
from dsa2000_cal.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_cal.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel


class AbstractSourceModel(ABC):

    @abstractmethod
    def is_full_stokes(self) -> bool:
        """
        Check if the source model is full stokes.

        Returns:
            True if full stokes, False otherwise
        """
        ...

    @abstractmethod
    def predict(
            self,
            visibility_coords: VisibilityCoords,
            gain_model: GainModel,
            near_field_delay_engine: BaseNearFieldDelayEngine,
            far_field_delay_engine: BaseFarFieldDelayEngine,
            geodesic_model: BaseGeodesicModel
    ) -> ComplexArray:
        """
        Predict the visibility model for the given model data.

        Args:
            visibility_coords: the visibility coordinates
            gain_model: the gain model
            near_field_delay_engine: the near field delay engine
            far_field_delay_engine: the far field delay engine
            geodesic_model: the geodesic model


        Returns:
            the visibility model
        """
        ...
