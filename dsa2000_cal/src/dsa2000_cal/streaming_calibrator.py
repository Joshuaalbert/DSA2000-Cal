import dataclasses
from typing import NamedTuple, Tuple, Any, List

import jax
import jax.numpy as jnp
import numpy as np
from jaxctx import transform

from dsa2000_cal.ops.residuals import compute_residual_TBC
from dsa2000_cal.probabilistic_models.gain_prior_models import quadratic_interpolation, set_diagonal
from dsa2000_cal.solvers.multi_step_lm import lm_solver
from dsa2000_common.common.array_types import FloatArray, ComplexArray
from dsa2000_common.common.pytree import Pytree


class StreamData(NamedTuple):
    vis_obs: FloatArray  # [T, B, C, 2, 2] visibility observations
    weights: FloatArray  # [T, B, C, 2, 2] weights for the observations
    vis_model: FloatArray  # [D, T, B, C, 2, 2] model visibilities


class IonosphereLayerParams(NamedTuple):
    """
    Parameters for the ionosphere layer.
    """
    length_scale: FloatArray  # [km]
    longitude_pole: FloatArray  # [rad]
    latitude_pole: FloatArray  # [rad]
    bottom_velocity: FloatArray  # [km/s]
    radial_velocity: FloatArray  # [km/s]
    bottom: FloatArray  # [km]
    width: FloatArray  # [km]
    fed_mu: FloatArray  # [1e10 e-/m^3]
    fed_sigma: FloatArray  # [1e10 e-/m^3]


class GainModel(NamedTuple):
    G_amp_model: FloatArray  # [A]
    G_phase_model: FloatArray  # [A]
    B_amp: FloatArray  # [A, Cm, 2]
    B_delay: FloatArray  # [A,2]
    tec: FloatArray  # [D, A]


def assemble_gains(gain_model: GainModel, freqs: FloatArray, model_freqs: FloatArray) -> ComplexArray:
    # Build the total gains from the model
    tec_conv = -8.4479745e6 / freqs  # rad / mTECU [C]
    delay_conv = (2 * np.pi * 1e-9) * freqs  # rad / ns [C]
    B_amp = quadratic_interpolation(freqs, model_freqs, gain_model.B_amp, axis=1)  # [A, C, 2]
    B_phase = gain_model.B_delay[:, None, :] * delay_conv[:, None]  # [A, C, 2]
    prop_phase = gain_model.tec[:, :, None] * tec_conv  # [D, A, C]
    net_amp = gain_model.G_amp_model[:, None, None] * B_amp  # [A, C, 2]
    net_phase = gain_model.G_phase_model[:, None, None] + B_phase + prop_phase[:, :, :, None]  # [D, A, C, 2]
    gains = net_amp * jax.lax.complex(jnp.cos(net_phase), jnp.sin(net_phase))  # [D, A, C, 2]
    gains = set_diagonal(gains)  # [D, A, C, 2, 2]
    return gains


@dataclasses.dataclass(eq=False)
class StreamingCalibrator(Pytree):
    """
    A class that provides implementations of core calibration subroutines for a radio camera.

    It holds a mutatable state that can be updated with new data, as well as transitioning and slewing capabilities.
    The general flow would be:
    t0: create new calibrator | slew(directions, t0)
    t1: transition(t1) | if new data is available, update(data) | if desired, subtract(data) to get residuals.

    In code like:

    cal = StreamingCalibrator()
    while True:
      event, payload = pull.recv_multipart()
      if event == 'new_data':
        t, data = payload
        cal.transition(t)
        cal.update(data)
        if do_subtract:
          residuals = cal.subtract(data)
          push.send_multipart([t, residuals])
      if event == 'slew':
        t, directions = payload
        cal.slew(t, directions)
    """
    # params
    freqs: FloatArray  # [C] frequencies of the data
    model_freqs: FloatArray  # [Cm] model frequencies
    antenna1: FloatArray  # [B] antenna 1 indices
    antenna2: FloatArray  # [B] antenna 2 indices
    x0_radius: FloatArray  # [km]

    skip_post_init: bool = False

    def __post_init__(self):
        """
        Post-initialization to register the class as a pytree.
        """
        if self.skip_post_init:
            return

    @classmethod
    def flatten(cls, this: 'StreamingCalibrator') -> Tuple[List[Any], Tuple[Any, ...]]:
        pass

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> 'StreamingCalibrator':
        pass

    def subtract(self, data: StreamData) -> ComplexArray:
        """
        Subtract the model visibilities from the observed visibilities.

        Args:
            data: StreamData containing the observed visibilities and model visibilities.

        Returns:
            ComplexArray: The residuals after subtracting the model visibilities from the observed visibilities.
        """
        ...

    def update(self, data: StreamData) -> None:
        """
        Update the calibrator with new data and perform calibration.

        Args:
            data: StreamData containing the observed visibilities, weights, and model visibilities.
        """

        def build_gain_model() -> GainModel:
            ...

        transformed_gain_model = transform(build_gain_model)

        init_params = transformed_gain_model.init(
            rngs={"params": jax.random.PRNGKey(0)},
            collections=None
        ).collections

        def get_gain_model(params) -> GainModel:
            return transformed_gain_model.apply({"params": jax.random.PRNGKey(0)}, params, data).fn_val

        def residual_fn(params, data: StreamData, freqs: FloatArray, model_freqs: FloatArray,
                        antenna1, antenna2) -> FloatArray:
            gain_model = get_gain_model(params)
            gains = assemble_gains(gain_model, freqs, model_freqs)
            return compute_residual_TBC(
                vis_model=data.vis_model,  # [D, T, B, C, 2, 2]
                vis_data=data.vis_obs,  # [T, B, C, 2, 2]
                gains=gains[:, None],  # [D, 1, A, C, 2, 2]
                antenna1=antenna1,  # [B]
                antenna2=antenna2,  # [B]
            )

        params, diagnostics = lm_solver(
            residual_fn=residual_fn,
            x0=init_params,
            args=(data,)
        )
        gain_model = get_gain_model(params)

    def transition(self, t: FloatArray) -> None:
        """
        Transition the calibrator state to the specified time.

        Args:
            t: the time to which to transition.
        """
        pass

    def slew(self, directions: FloatArray, t: FloatArray) -> None:
        """
        Slew the calibrator to a new set of directions at a given time.

        Args:
            directions: FloatArray  # [D, 2] the new directions to slew to
            t: FloatArray  # [D] the time at which to slew
        """
        pass


StreamingCalibrator.register_pytree()
