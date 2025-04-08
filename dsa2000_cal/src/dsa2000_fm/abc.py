from abc import ABC, abstractmethod
from typing import List, Tuple

from dsa2000_fm.array_layout.sample_constraints import RegionSampler


class AbstractArrayConstraint(ABC):
    @abstractmethod
    def get_constraint_regions(self) -> List[Tuple[RegionSampler, float]]:
        """
        Returns the buffer constraints for the array.

        Returns:
            list of tuples: (RegionSampler, buffer distance m)
        """
        ...

    @abstractmethod
    def get_area_of_interest_regions(self) -> List[Tuple[RegionSampler, float]]:
        """
        Returns the buffer constraints for the array.

        Returns:
            list of tuples: (RegionSampler, buffer distance m)
        """
        ...


