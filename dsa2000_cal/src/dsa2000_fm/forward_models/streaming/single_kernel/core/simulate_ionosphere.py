import dataclasses
from typing import NamedTuple, Tuple, Any

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_common.common.array_types import FloatArray
from dsa2000_cal.common.astropy_utils import create_spherical_grid
from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_common.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel, \
    build_spherical_interpolator
from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model
from dsa2000_fm.forward_models.streaming.single_kernel.abc import AbstractCoreStep


class SimulateIonosphereState(NamedTuple):
    window_dtec: FloatArray  # [window_size, num_model_ant, num_model_dir]
    window_geodesics: FloatArray  # [window_size, num_model_ant, num_model_dir, 3]
    window_times: FloatArray  # [window_size]


class SimulateIonosphereOutput(NamedTuple):
    gain_model: BaseSphericalInterpolatorGainModel


@dataclasses.dataclass(eq=False)
class SimulateIonosphereStep(AbstractCoreStep[SimulateIonosphereState, SimulateIonosphereOutput, None]):
    window_resolution: au.Quantity
    window_size: int

    antennas: ac.EarthLocation  # [num_ant]
    ref_time: at.Time
    array_location: ac.EarthLocation
    phase_center: ac.ICRS
    obstimes: at.Time  # [num_model_time]
    pointings: ac.ICRS  # [[num_ant]]
    field_of_view: au.Quantity

    specification: SPECIFICATION
    plot_folder: str
    cache_folder: str

    ref_ant: ac.EarthLocation | None = None
    ref_time: at.Time | None = None

    compute_tec: bool = True  # Faster to compute TEC only and differentiate later
    S_marg: int = 25
    jitter: float = 0.05  # Adds 0.05 mTECU noise to the covariance matrix

    def __post_init__(self):
        # Create direction grid
        directional_grid = create_spherical_grid(
            pointing=self.phase_center,
            angular_radius=self.field_of_view,
            num_shells=3
        )
        # Create antenna grid
        antenna_grid = ...

    def get_state(self) -> SimulateIonosphereState:
        # Create realisation of the ionosphere over window size
        window_times = quantity_to_jnp(np.arange(1, self.window_size + 1) * self.window_resolution)
        geodesic_model = build_geodesic_model(
            antennas=self.antennas,
            array_location=self.array_location,
            phase_center=self.phase_center,
            obstimes=self.obstimes,
            ref_time=self.ref_time,
            pointings=self.pointings,
        )

        geodesic_model.compute_far_field_geodesic()

        state = SimulateIonosphereState(
            window_dtec=...,
            window_geodesics=...,
            window_times=window_times
        )
        return state

    def step(self, primals: Tuple[Any, ...]) -> \
            Tuple[SimulateIonosphereState, SimulateIonosphereOutput, None]:
        gain_model = build_spherical_interpolator(
            antennas=...

        )
