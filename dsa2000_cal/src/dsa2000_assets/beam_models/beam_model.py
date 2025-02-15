from abc import abstractmethod, ABC

from dsa2000_cal.antenna_model.abc import AbstractAntennaModel
from dsa2000_assets.base_content import BaseContent


class AbstractBeamModel(ABC, BaseContent):

    @abstractmethod
    def get_antenna_model(self) -> AbstractAntennaModel:
        """
        Get the beam model.

        Returns:
            beam model
        """
        ...
