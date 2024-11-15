from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic, Tuple

OutputType = TypeVar('OutputType')
KeepType = TypeVar('KeepType')


class AbstractCoreStep(ABC, Generic[OutputType, KeepType]):

    def __hash__(self):
        return hash(self.name)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def state_name(self):
        return f"{self.name}State"

    @property
    def output_name(self):
        return f"{self.name}Output"

    @property
    def keep_name(self):
        return f"{self.name}Keep"

    @abstractmethod
    def step(self, primals: Tuple[Any, ...]) -> Tuple[OutputType, KeepType]:
        """
        Run a single step of the streaming process.

        Args:
            primals: the primals for the current step, a tuple of args from predecessor steps.

        Returns:
            output: the output of the streaming process
            keep: what is keep from each step
        """
        ...
