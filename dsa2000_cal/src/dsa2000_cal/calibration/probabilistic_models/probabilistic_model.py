import dataclasses
from abc import ABC, abstractmethod
from typing import Tuple, Callable, List, Any

import astropy.time as at
import jax

from src.dsa2000_cal.common.types import ComplexArray, FloatArray
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from src.dsa2000_cal.measurement_sets import VisibilityData, MeasurementSet


@dataclasses.dataclass(eq=False)
class ProbabilisticModelInstance:
    get_init_params_fn: Callable[[], Any]
    log_prob_joint_fn: Callable[[Any], jax.Array]
    forward_fn: Callable[[Any], Tuple[jax.Array, List[jax.Array]]]

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


def combine_probabilistic_model_instances(instances: List[ProbabilisticModelInstance]):
    def get_init_params_fn() -> List[Any]:
        return [instance.get_init_params() for instance in instances]

    def log_prob_joint_fn(params: List[Any]):
        log_prob_joint = [instance.log_prob_joint(param) for instance, param in zip(instances, params)]
        return sum(log_prob_joint[1:], start=log_prob_joint[0])

    def forward_fn(params: List[Any]):
        visibilities = []
        constrained_params = []
        for instance, param in zip(instances, params):
            vis, constrained_param = instance.forward(param)
            visibilities.append(vis)
            constrained_params.append(constrained_param)
        vis = sum(visibilities[1:], start=visibilities[0])
        return vis, constrained_params

    return ProbabilisticModelInstance(
        get_init_params_fn=get_init_params_fn,
        log_prob_joint_fn=log_prob_joint_fn,
        forward_fn=forward_fn
    )


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
