import dataclasses
import os
from typing import NamedTuple, Tuple, Any

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.numpy as jnp

import dsa2000_cal.common.context as ctx
from dsa2000_cal.common.array_types import FloatArray, IntArray, BoolArray
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_common.delay_models import BaseFarFieldDelayEngine
from dsa2000_common.delay_models import build_far_field_delay_engine
from dsa2000_common.delay_models import build_near_field_delay_engine, \
    BaseNearFieldDelayEngine
from dsa2000_fm.forward_models.streaming.single_kernel.abc import AbstractCoreStep
from dsa2000_common.geodesics.base_geodesic_model import BaseGeodesicModel, build_geodesic_model


class SetupObservationState(NamedTuple):
    freqs: FloatArray
    times: FloatArray
    solution_idx: IntArray
    geodesic_model: BaseGeodesicModel
    far_field_delay_engine: BaseFarFieldDelayEngine
    near_field_delay_engine: BaseNearFieldDelayEngine


class SetupObservationOutput(NamedTuple):
    freqs: FloatArray
    times: FloatArray
    do_solve: BoolArray
    geodesic_model: BaseGeodesicModel
    far_field_delay_engine: BaseFarFieldDelayEngine
    near_field_delay_engine: BaseNearFieldDelayEngine


@dataclasses.dataclass(eq=False)
class SetupObservationStep(AbstractCoreStep[SetupObservationOutput, None]):
    freqs: au.Quantity  # [num_freqs]
    antennas: ac.EarthLocation
    array_location: ac.EarthLocation
    phase_center: ac.ICRS
    obstimes: at.Time  # [num_model_times] over which to compute the zenith
    ref_time: at.Time
    pointings: ac.ICRS | None  # [[num_ant]] or None which means Zenith
    plot_folder: str
    solution_interval: au.Quantity
    validity_interval: au.Quantity
    integration_interval: au.Quantity

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        self.geodesic_model = build_geodesic_model(
            antennas=self.antennas,
            array_location=self.array_location,
            phase_center=self.phase_center,
            obstimes=self.obstimes,
            ref_time=self.ref_time,
            pointings=self.pointings
        )

        self.far_field_delay_engine = build_far_field_delay_engine(
            antennas=self.antennas,
            start_time=self.obstimes[0],
            end_time=self.obstimes[-1],
            ref_time=self.ref_time,
            phase_center=self.phase_center
        )

        self.near_field_delay_engine = build_near_field_delay_engine(
            antennas=self.antennas,
            start_time=self.obstimes[0],
            end_time=self.obstimes[-1],
            ref_time=self.obstimes[0]
        )

    def get_state(self) -> SetupObservationState:
        freqs = quantity_to_jnp(self.freqs)
        integrations_per_solution = int(self.solution_interval / self.integration_interval)
        times = mp_policy.cast_to_time(
            jnp.arange(integrations_per_solution) * quantity_to_jnp(self.integration_interval)
        )
        return SetupObservationState(
            freqs=freqs,
            times=times,
            geodesic_model=self.geodesic_model,
            far_field_delay_engine=self.far_field_delay_engine,
            near_field_delay_engine=self.near_field_delay_engine,
            solution_idx=jnp.zeros((), mp_policy.index_dtype)
        )

    def step(self, primals: Any) -> Tuple[SetupObservationOutput, None]:
        state = ctx.get_state(
            'state',
            init=lambda: self.get_state()
        )

        applies_per_solve = int(self.validity_interval / self.solution_interval)

        do_solve = state.solution_idx % applies_per_solve == 0
        output = SetupObservationOutput(
            freqs=state.freqs,
            times=state.times,
            do_solve=do_solve,
            geodesic_model=state.geodesic_model,
            far_field_delay_engine=state.far_field_delay_engine,
            near_field_delay_engine=state.near_field_delay_engine
        )

        # increment state
        next_times = state.times + quantity_to_jnp(self.solution_interval)
        next_solution_idx = state.solution_idx + jnp.ones((), mp_policy.index_dtype)
        state = state._replace(
            times=next_times,
            solution_idx=next_solution_idx
        )

        ctx.set_state('state', state)

        return output, None
