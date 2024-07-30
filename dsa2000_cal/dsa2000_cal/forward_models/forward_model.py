import dataclasses
from abc import abstractmethod, ABC

from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet


@dataclasses.dataclass(eq=False)
class AbstractForwardModel(ABC):
    @abstractmethod
    def forward(self, ms: MeasurementSet):
        ...
