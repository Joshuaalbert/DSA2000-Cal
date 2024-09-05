import dataclasses
from typing import NamedTuple, Any

import jax


class MultiStepLevenbergMarquardtState(NamedTuple):
    iteration: jax.Array  # iteration number
    solution: jax.Array | Any  # current solution, may be a pytree
    residual: jax.Array | Any  # residual, may be a pytree
    damping: jax.Array  # damping factor


@dataclasses.dataclass(eq=False)
class MultiStepLevenbergMarquardt:
    """
    Multi-step Levenberg-Marquardt algorithm.

    Finds a local minimum to the least squares problem defined by a residual function, F(x)=0.

    Implements the algorithm described in [1]. In addition, it applies CG method to solve the normal equations,
    efficient use of JVP and VJP to avoid computing the Jacobian matrix, adaptive step size, and mixed precision.

    References:
        [1] Fan, J., Huang, J. & Pan, J. An Adaptive Multi-step Levenberg–Marquardt Method.
            J Sci Comput 78, 531–548 (2019). https://doi.org/10.1007/s10915-018-0777-8
    """

    def create_initial_state(self) -> MultiStepLevenbergMarquardtState:
        """
        Create the initial state for the algorithm.

        Returns:
            initial state
        """
        pass

    def lm_step(self, state: MultiStepLevenbergMarquardtState) -> MultiStepLevenbergMarquardtState:
        """
        Perform a single step of the algorithm.

        Args:
            state: current state

        Returns:
            new state
        """
        pass

    def approx_lm_step(self, state: MultiStepLevenbergMarquardtState) -> MultiStepLevenbergMarquardtState:
        """
        Perform a single approximate step of the algorithm.

        Args:
            state: current state

        Returns:
            new state
        """
        pass
