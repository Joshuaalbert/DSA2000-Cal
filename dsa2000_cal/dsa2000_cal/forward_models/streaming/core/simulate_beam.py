import dataclasses
import os
from typing import NamedTuple, Tuple, Any

import astropy.time as at
import astropy.units as au
import jax

import dsa2000_cal.common.context as ctx
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
    freqs: au.Quantity
    plot_folder: str

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)

    def get_state(self) -> SimulateBeamState:
        beam_model: BaseSphericalInterpolatorGainModel = ctx.get_state(
            'beam_model',
            init=lambda: build_beam_gain_model(
                array_name=self.array_name,
                full_stokes=self.full_stokes,
                model_times=self.model_times,
                freqs=self.freqs
            )
        )
        beam_model.plot_regridded_beam(save_fig=os.path.join(self.plot_folder, 'regridded_beam.png'))
        return SimulateBeamState(beam_model=beam_model)

    def step(self, primals: Any) -> Tuple[
        SimulateBeamState, SimulateBeamOutput, None]:
        state = self.get_state()
        # identity transition
        return state, SimulateBeamOutput(beam_model=state.beam_model), None
