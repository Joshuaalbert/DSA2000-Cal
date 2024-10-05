import dataclasses
from typing import List, Callable

import jax
from jax import numpy as jnp

from dsa2000_cal.forward_models.streaming.abc import AbstractStreamProcess, AbstractForwardModelCore
from dsa2000_cal.forward_models.streaming.types import StreamState


@dataclasses.dataclass(eq=False)
class StreamProcess(AbstractStreamProcess):
    """
    Represents a streaming process that runs a forward model in a streaming fashion on a single node using all local
    devices.
    """
    core: AbstractForwardModelCore
    num_time: int
    solution_interval: int
    validity_interval: int
    callbacks: List[Callable]

    def create_initial_state(self, key) -> StreamState:
        ...

    def stream(self):
        # Solve once per validity interval
        solve_cadence = self.validity_interval // self.solution_interval

        # Holders for state that will be passed forward
        init_params = None
        solver_state = None
        for cadence_idx in range(0, self.num_time // self.solution_interval):
            # Determine if we solve or not
            if cadence_idx % solve_cadence == 0:
                num_cal_iters = 15
            else:
                num_cal_iters = 0
            # Run step
            times = jnp.arange(self.solution_interval) + cadence_idx * self.solution_interval
            key = jax.random.PRNGKey(cadence_idx)
            step_return = self.core.step(
                key=key,
                num_calibration_iters=num_cal_iters,
                times=times,
                init_params=init_params,
                solver_state=solver_state
            )
            # Update state that will be passed forward
            init_params = step_return.cal_params
            solver_state = step_return.calibration_solver_state
            # Run callbacks
            for callback in self.callbacks:
                callback(step_return)
