import dataclasses

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.delay_models.base_far_field_delay_engine import build_far_field_delay_engine


from dsa2000_common.delay_models.base_near_field_delay_engine import build_near_field_delay_engine


from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model


@dataclasses.dataclass(eq=False)
class ObservationSetup:
    freqs: au.Quantity  # [num_freqs]
    antennas: ac.EarthLocation
    array_location: ac.EarthLocation
    phase_center: ac.ICRS
    obstimes: at.Time  # [num_model_times] over which to compute the zenith
    ref_time: at.Time
    pointings: ac.ICRS | None  # [[num_ant]] or None which means Zenith

    @staticmethod
    def create_tracking_from_array(array_name: str, ref_time: at.Time, num_timesteps: int,
                                   phase_center: ac.ICRS) -> "ObservationSetup":
        fill_registries()
        array = array_registry.get_instance(array_registry.get_match(array_name))
        return ObservationSetup(
            freqs=array.get_channels(),
            antennas=array.get_antennas(),
            array_location=array.get_array_location(),
            phase_center=phase_center,
            obstimes=ref_time + np.arange(num_timesteps) * array.get_integration_time(),
            ref_time=ref_time,
            pointings=phase_center
        )

    def __post_init__(self):
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
