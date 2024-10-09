import dataclasses
from typing import NamedTuple, Tuple, Any

import astropy.time as at
import jax

from dsa2000_cal.forward_models.streaming.abc import AbstractCoreStep
from dsa2000_cal.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model


class SimulateBeamState(NamedTuple):
    beam_model: BaseSphericalInterpolatorGainModel


class SimulateBeamOutput(NamedTuple):
    beam_model: BaseSphericalInterpolatorGainModel


@dataclasses.dataclass(eq=False)
class SimulateBeamStep(AbstractCoreStep[SimulateBeamState, SimulateBeamOutput, None]):
    array_name: str
    full_stokes: bool
    model_times: at.Time | None

    def create_initial_state(self, key: jax.Array) -> SimulateBeamState:
        beam_model = build_beam_gain_model(
            array_name=self.array_name,
            full_stokes=self.full_stokes,
            model_times=self.model_times
        )
        return SimulateBeamState(beam_model=beam_model)

    def step(self, key: jax.Array, state: SimulateBeamState, primals: Any) -> Tuple[
        SimulateBeamState, SimulateBeamOutput, None]:
        # identity transition
        return state, SimulateBeamOutput(beam_model=state.beam_model), None
