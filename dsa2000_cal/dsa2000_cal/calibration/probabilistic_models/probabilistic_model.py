import dataclasses
from abc import ABC, abstractmethod
from typing import Tuple, Callable, List, Any

import astropy.time as at
import jax

from dsa2000_cal.common.types import ComplexArray, FloatArray
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.measurement_sets.measurement_set import VisibilityData, MeasurementSet


@dataclasses.dataclass(eq=False)
class ProbabilisticModelInstance:
    get_init_params_fn: Callable[[], Any]
    log_prob_joint_fn: Callable[[Any], jax.Array]
    forward_fn: Callable[[Any], Tuple[jax.Array, List[jax.Array]]]

    def __add__(self, other: 'ProbabilisticModelInstance') -> 'ProbabilisticModelInstance':
        def get_init_params_fn() -> List[Any]:
            return [self.get_init_params(), other.get_init_params()]

        def log_prob_joint_fn(params: List[Any]):
            return self.log_prob_joint(params[0]) + other.log_prob_joint(params[1])

        def forward_fn(params: List[Any]):
            vis1, gains1 = self.forward(params[0])
            vis2, gains2 = other.forward(params[1])
            return vis1 + vis2, [gains1, gains2]

        return ProbabilisticModelInstance(
            get_init_params_fn=get_init_params_fn,
            log_prob_joint_fn=log_prob_joint_fn,
            forward_fn=forward_fn
        )

    def get_init_params(self) -> Any:
        """
        Get the initial parameters for the gains.

        Returns:
            params: [[num_chan,] ...] the initial parameters, possibly sharded over channel.
        """
        return self.get_init_params_fn()

    def log_prob_joint(self, params: Any) -> FloatArray:
        """
        Compute the joint log probability of the gains and the data.

        Args:
            params: [[num_chan,] ...]

        Returns:
            scalar
        """
        return self.log_prob_joint_fn(params)

    def forward(self, params: Any) -> Tuple[ComplexArray, List[Any]]:
        """
        Forward model for the gains.

        Args:
            params: [[num_chan,] ...]

        Returns:
            vis_model: [num_row, num_chan, 4]
            gains: [num_source, num_ant, num_chan, 2, 2]
        """
        return self.forward_fn(params)


class AbstractProbabilisticModel(ABC):
    """
    Represents a probabilistic model and generates instances, based on data.
    """

    @abstractmethod
    def create_model_instance(self, freqs: jax.Array,
                              times: jax.Array,
                              vis_data: VisibilityData,
                              vis_coords: VisibilityCoords
                              ) -> ProbabilisticModelInstance:
        """
        Returns an instance of a probabilistic model of vis and gains.

        Args:
            freqs: [num_chan] chans to solve gains at
            times: [num_time] times to solve gains at
            vis_data: [num_row, num_chan[, 2, 2]] vis local shard over chan
            vis_coords: [num_row, 2]

        Returns:
            An instance
        """
        ...

    @abstractmethod
    def save_solution(self, solution: Any, file_name: str, times: at.Time, ms: MeasurementSet):
        """
        Save the solution to a folder.

        Args:
            solution: Any
            file_name: str
            times: at.Time
        """
        ...
