import dataclasses
import os
from typing import NamedTuple, Tuple, Any

import astropy.time as at
import astropy.units as au

import dsa2000_cal.common.context as ctx
from dsa2000_cal.forward_models.streaming.abc import AbstractCoreStep
from dsa2000_cal.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model


class SimulateBeamState(NamedTuple):
    beam_model: BaseSphericalInterpolatorGainModel


class SimulateBeamOutput(NamedTuple):
    beam_model: BaseSphericalInterpolatorGainModel


@dataclasses.dataclass(eq=False)
class SimulateBeamStep(AbstractCoreStep[SimulateBeamOutput, None]):
    array_name: str
    full_stokes: bool
    model_times: at.Time | None
    freqs: au.Quantity
    plot_folder: str

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)

    def get_state(self) -> SimulateBeamState:
        def get_build_beam_gain_model():
            beam_model = build_beam_gain_model(
                array_name=self.array_name,
                full_stokes=self.full_stokes,
                model_times=self.model_times,
                freqs=self.freqs
            )

            # def plot_beam_model(beam_model: BaseSphericalInterpolatorGainModel):
            #     def _plot_beam_model(beam_model: BaseSphericalInterpolatorGainModel):
            #         output = os.path.join(self.plot_folder, 'beam_model.png')
            #         if os.path.exists(output):
            #             return np.asarray(False)
            #
            #         beam_model.plot_regridded_beam(save_fig=os.path.join(self.plot_folder, 'regridded_beam.png'))
            #         return np.asarray(True)
            #
            #     return jax.experimental.io_callback(
            #         _plot_beam_model,
            #         jax.ShapeDtypeStruct((), jnp.bool_), beam_model,
            #         ordered=False
            #     )
            #
            # plot_beam_model(beam_model)
            beam_model.plot_regridded_beam(save_fig=os.path.join(self.plot_folder, 'regridded_beam.png'))
            return beam_model

        beam_model: BaseSphericalInterpolatorGainModel = ctx.get_state(
            'beam_model',
            init=lambda: get_build_beam_gain_model()
        )

        return SimulateBeamState(beam_model=beam_model)

    def step(self, primals: Any) -> Tuple[SimulateBeamOutput, None]:
        state = ctx.get_state('state', init=lambda: self.get_state())
        # identity transition
        return SimulateBeamOutput(beam_model=state.beam_model), None
