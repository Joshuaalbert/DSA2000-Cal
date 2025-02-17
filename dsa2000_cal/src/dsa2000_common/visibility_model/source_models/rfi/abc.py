from abc import ABC, abstractmethod
from typing import Tuple

from dsa2000_common.common.array_types import ComplexArray, FloatArray


class AbstractRFIAutoCorrelationFunction(ABC):
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the auto-correlation function.

        Returns:
            the shape of the auto-correlation function
        """
        ...

    @abstractmethod
    def eval(self, freq: FloatArray, tau: FloatArray) -> ComplexArray:
        """
        Evaluate the auto-correlation function at the given delay times.

        Args:
            freq: the central frequency of channel in Hz.
            tau: the delay time

        Returns:
            [E[,2,2]] the auto-correlation function
        """
        ...
