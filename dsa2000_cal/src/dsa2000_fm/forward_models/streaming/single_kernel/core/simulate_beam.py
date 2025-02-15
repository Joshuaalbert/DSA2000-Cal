import dataclasses
import os
from typing import NamedTuple, Tuple

import astropy.time as at
import astropy.units as au

import dsa2000_cal.common.context as ctx
from dsa2000_fm.forward_models.streaming.single_kernel.abc import AbstractCoreStep
from dsa2000_fm.forward_models.streaming.single_kernel.core.setup_observation import SetupObservationOutput
from dsa2000_common.gain_models import BaseSphericalInterpolatorGainModel
from dsa2000_common.gain_models import build_beam_gain_model


class SimulateBeamState(NamedTuple):
    beam_model: BaseSphericalInterpolatorGainModel


class SimulateBeamOutput(NamedTuple):
    beam_model: BaseSphericalInterpolatorGainModel


@dataclasses.dataclass(eq=False)
class SimulateBeamStep(AbstractCoreStep[SimulateBeamOutput, None]):
    """
    Simulate the beam model.

    Args:
        array_name: The name of the array.
        full_stokes: Whether to simulate the full Stokes parameters.
        times: The times of the observation.
        freqs: The freqs of the observation.
        plot_folder: The folder in which to save plots.
    """
    array_name: str
    full_stokes: bool
    times: at.Time  # [num_model_times]
    ref_time: at.Time
    freqs: au.Quantity  # [num_model_freqs]
    plot_folder: str

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        self.beam_model = build_beam_gain_model(
            array_name=self.array_name, times=self.times,
            ref_time=self.ref_time,
            freqs=self.freqs, full_stokes=self.full_stokes
        )
        # plot_beam_model(beam_model)
        self.beam_model.plot_regridded_beam(save_fig=os.path.join(self.plot_folder, 'regridded_beam.png'))

    def get_state(self) -> SimulateBeamState:
        return SimulateBeamState(beam_model=self.beam_model)

    def step(self, primals: Tuple[SetupObservationOutput]) -> Tuple[SimulateBeamOutput, None]:
        (setup_observation_output,) = primals
        state = ctx.get_state('state', init=lambda: self.get_state())
        # identity transition, so no need to increment state
        return SimulateBeamOutput(beam_model=state.beam_model), None
