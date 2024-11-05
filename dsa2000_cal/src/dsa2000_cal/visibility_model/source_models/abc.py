from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from dsa2000_cal.common.types import VisibilityCoords
from dsa2000_cal.common.array_types import ComplexArray, FloatArray
from dsa2000_cal.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_cal.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel

ModelDataType = TypeVar('ModelDataType')


class AbstractSourceModel(ABC, Generic[ModelDataType]):

    @abstractmethod
    def is_full_stokes(self) -> bool:
        """
        Check if the source model is full stokes.

        Returns:
            True if full stokes, False otherwise
        """
        ...

    def get_model_data(self, freqs: FloatArray, times: FloatArray, geodesic_model: BaseGeodesicModel) -> ModelDataType:
        """
        Construct the model data for the given freqs and times.

        Returns:
            the model data ordered for optimal reduction [num_time, num_freqs] + source_shape + [[2,2]]
        """
        ...

    @abstractmethod
    def predict(
            self,
            model_data: ModelDataType,
            visibility_coords: VisibilityCoords,
            gain_model: GainModel,
            near_field_delay_engine: BaseNearFieldDelayEngine,
            far_field_delay_engine: BaseFarFieldDelayEngine,
            geodesic_model: BaseGeodesicModel
    ) -> ComplexArray:
        """
        Predict the visibility model for the given model data.

        Args:
            model_data: the model data [num_time, num_freqs] + source_shape + [[2,2]]
            visibility_coords: the visibility coordinates
            gain_model: the gain model
            near_field_delay_engine: the near field delay engine
            far_field_delay_engine: the far field delay engine
            geodesic_model: the geodesic model


        Returns:
            the visibility model
        """
        ...
