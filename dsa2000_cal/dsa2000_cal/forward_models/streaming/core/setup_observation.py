import dataclasses
from typing import NamedTuple, Tuple, Any

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax

import dsa2000_cal.common.context as ctx
from dsa2000_cal.common.types import FloatArray, IntArray
from dsa2000_cal.delay_models.far_field import FarFieldDelayEngine
from dsa2000_cal.delay_models.near_field import NearFieldDelayEngine
from dsa2000_cal.forward_models.streaming.abc import AbstractCoreStep
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel, build_geodesic_model


class SetupObservationState(NamedTuple):
    freqs: FloatArray
    geodesic_model: BaseGeodesicModel
    far_field_delay_engine: FarFieldDelayEngine
    near_field_delay_engine: NearFieldDelayEngine


class SetupObservationOutput(NamedTuple):
    solution_idx: IntArray
    validity_idx: IntArray
    times: FloatArray
    freqs: FloatArray
    geodesic_model: BaseGeodesicModel
    far_field_delay_engine: FarFieldDelayEngine
    near_field_delay_engine: NearFieldDelayEngine


@dataclasses.dataclass(eq=False)
class SetupObservationStep(AbstractCoreStep[SetupObservationState, SetupObservationOutput, None]):
    freqs: au.Quantity  # [num_freqs]
    antennas: ac.EarthLocation
    array_location: ac.EarthLocation
    phase_center: ac.ICRS
    obstimes: at.Time  # [num_model_times] over which to compute the zenith
    ref_time: at.Time
    pointings: ac.ICRS | None  # [[num_ant]] or None which means Zenith

    def get_state(self) -> SetupObservationState:
        geodesic_model = ctx.get_parameter(
            'geodesic_model',
            init=lambda: build_geodesic_model(
                antennas=self.antennas,
                array_location=self.array_location,
                phase_center=self.phase_center,
                obstimes=self.obstimes,
                ref_time=self.ref_time,
                pointings=self.pointings,
            )
        )
        # TODO: make pytrees
        far_field_delay_engine = ctx.get_parameter(
            'far_field_delay_engine',
            init=lambda: FarFieldDelayEngine(
                antennas=self.antennas,
                start_time=self.obstimes[0],
                end_time=self.obstimes[-1],
                phase_center=self.phase_center
            )
        )
        near_field_delay_engine = ctx.get_parameter(
            'near_field_delay_engine',
            init=lambda: NearFieldDelayEngine(
                antennas=self.antennas,
                start_time=self.obstimes[0],
                end_time=self.obstimes[-1]
            )
        )
        freqs = ctx.get_parameter('freqs', init=lambda: self.freqs)
        return SetupObservationState(
            freqs=freqs,
            geodesic_model=geodesic_model,
            far_field_delay_engine=far_field_delay_engine,
            near_field_delay_engine=near_field_delay_engine
        )

    def step(self, primals: Any) -> Tuple[
        SetupObservationState, SetupObservationOutput, None]:
        state = self.get_state()
        # identity transition
        return state, SetupObservationOutput(
            freqs=state.freqs,
            geodesic_model=state.geodesic_model,
            far_field_delay_engine=state.far_field_delay_engine,
            near_field_delay_engine=state.near_field_delay_engine
        ), None


def test_setup_observation_step():
    step = SetupObservationStep(
        freqs=au.Quantity([1.0, 2.0], unit='GHz'),
        antennas=ac.EarthLocation.from_geodetic([0, 0] * au.deg, [0, 0] * au.deg, [0, 0] * au.m),
        array_location=ac.EarthLocation.from_geodetic(1.0 * au.deg, 1.0 * au.deg, 1.0 * au.m),
        phase_center=ac.ICRS(ra=0.0 * au.deg, dec=0.0 * au.deg),
        obstimes=at.Time([0.0, 1.0], format='mjd'),
        ref_time=at.Time(0.0, format='mjd'),
        pointings=ac.ICRS(ra=0.0 * au.deg, dec=0.0 * au.deg)
    )
    init = ctx.transform_with_state(step.step).init(jax.random.PRNGKey(0), ())
    print(init)
