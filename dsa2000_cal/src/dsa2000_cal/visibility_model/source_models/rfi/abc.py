from abc import ABC, abstractmethod
from typing import Tuple

from dsa2000_cal.common.array_types import ComplexArray, FloatArray


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
            freq: the frequency
            tau: the delay time

        Returns:
            [[,2,2]] the auto-correlation function
        """
        ...
