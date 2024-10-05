from abc import ABC, abstractmethod
from typing import Any, List

import jax

from dsa2000_cal.forward_models.streaming.types import StepReturn, StreamState


class AbstractForwardModelCore(ABC):

    @abstractmethod
    def step(self, key: jax.Array, times: jax.Array, init_params: List[jax.Array] | None,
             solver_state: Any | None, num_calibration_iters: int) -> StepReturn:
        """
        Kernel that defines a single step of forward model.

        Args:
            key: PRNGkey for reproducibility
            times: the times simulated at this step, must be a solution interval worth
            init_params: last cal params
            solver_state: last solver state
            num_calibration_iters: number of calibration iterations to run, 0 means no calibration on this step

        Returns:
            StepReturn: the return values of the step
        """
        ...


class AbstractStreamProcess(ABC):

    @abstractmethod
    def stream(self):
        """
        Run the streaming process.
        """
        ...


class CoreStep(ABC):

    @abstractmethod
    def step(self):
        ...