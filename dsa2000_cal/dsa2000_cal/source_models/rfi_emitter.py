import dataclasses

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.numpy as jnp
import numpy as np
from jax._src.typing import SupportsDType
from pydantic import Field
from tomographic_kernel.frames import ENU

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.common.coord_utils import earth_location_to_uvw
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel


class RFIEmitterModelParam(SerialisableBaseModel):
    location_east: au.Quantity = Field(
        desciption="Distance east of RFI transmitter from center of the telescope [m].",
        default=46.8 * au.km
    )
    location_north: au.Quantity = Field(
        description="Distance south of RFI transmitter from center of the telescope [m].",
        default=0. * au.km
    )
    location_up: au.Quantity = Field(
        description="Height of RFI transmitter [m] above array centre.",
        default=20. * au.m
    )
    polarization: au.Quantity = Field(
        description="Polarization angle of RFI [deg, 0=full XX, 90=full YY].",
        default=10. * au.deg
    )
    power: au.Quantity = Field(
        description="Power of RFI transmitter at the source [W/Hz].",
        default=6.4e-4 * au.W / au.Hz
    )
    frequency: au.Quantity = Field(
        description="Central frequency of RFI signal [Hz].",
        default=705 * au.MHz
    )
    bandwidth: au.Quantity = Field(
        description="Bandwidth of RFI signal [Hz].",
        default=5 * au.MHz
    )

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(RFIEmitterModelParam, self).__init__(**data)
        _check_rfi_emitter_model_params(self)


def _check_rfi_emitter_model_params(params: RFIEmitterModelParam):
    if not params.location_east.unit.is_equivalent(au.m):
        raise ValueError("lte_east must have units of length")
    if not params.location_north.unit.is_equivalent(au.m):
        raise ValueError("lte_north must have units of length")
    if not params.location_up.unit.is_equivalent(au.m):
        raise ValueError("lte_up must have units of length")
    if not params.polarization.unit.is_equivalent(au.deg):
        raise ValueError("lte_polarization must have units of angle")
    if not params.power.unit.is_equivalent(au.W / au.Hz):
        raise ValueError("lte_power must have units of power per Hz")
    if not params.frequency.unit.is_equivalent(au.Hz):
        raise ValueError("lte_frequency must have units of frequency")
    if not params.bandwidth.unit.is_equivalent(au.Hz):
        raise ValueError("lte_bandwidth must have units of frequency")

    if params.power < 0 * au.W / au.Hz:
        raise ValueError("lte_power must be non-negative")

    # Check shapes of arrays, all scalar
    if not params.location_east.isscalar:
        raise ValueError("lte_east must be scalar")
    if not params.location_north.isscalar:
        raise ValueError("lte_north must be scalar")
    if not params.location_up.isscalar:
        raise ValueError("lte_up must be scalar")
    if not params.polarization.isscalar:
        raise ValueError("lte_polarization must be scalar")
    if not params.power.isscalar:
        raise ValueError("lte_power must be scalar")
    if not params.frequency.isscalar:
        raise ValueError("lte_frequency must be scalar")
    if not params.bandwidth.isscalar:
        raise ValueError("lte_bandwidth must be scalar")


@dataclasses.dataclass(eq=False)
class RFIEmitterModel(AbstractSourceModel):
    """
    Predict vis for point source.
    """
    params: RFIEmitterModelParam

    dtype: SupportsDType = jnp.complex64

    @staticmethod
    def from_params(params: RFIEmitterModelParam, **kwargs) -> 'RFIEmitterModel':
        return RFIEmitterModel(params=params, **kwargs)

    def get_lmn(self, antennas: ac.EarthLocation, array_location: ac.EarthLocation, obs_time: at.Time,
                phase_tracking: ac.ICRS) -> au.Quantity:
        """
        Get the lmn coordinates of the source for each antenna.
        """
        enu_frame = ENU(obstime=obs_time, location=array_location)
        array_location_itrs = array_location.get_itrs(obstime=obs_time)
        array_location_enu = array_location_itrs.transform_to(enu_frame)
        # Offset by the RFI emitter location relative to array location
        emitter_location_enu = ENU(
            east=array_location_enu.east + self.params.location_east,
            north=array_location_enu.north + self.params.location_north,
            up=array_location_enu.up + self.params.location_up,
            obstime=obs_time,
            location=array_location
        )

        # Now convert to earth_location via ITRS
        emitter_location_itrs = emitter_location_enu.transform_to(ac.ITRS(obstime=obs_time))
        emitter_location_uvw = earth_location_to_uvw(
            antennas=emitter_location_itrs.earth_location,
            obs_time=obs_time,
            phase_tracking=phase_tracking
        )  # [3]
        # Get the lmn coordinates from the unit vector pointing to the source
        antenna_uvw = earth_location_to_uvw(antennas=antennas,
                                            obs_time=obs_time,
                                            phase_tracking=phase_tracking)  # [num_antennas, 3]

        lmn = emitter_location_uvw - antenna_uvw  # [num_antennas, 3]
        lmn /= np.linalg.norm(lmn, axis=-1, keepdims=True)  # [num_antennas, 3]
        return lmn


def test_rfi_emitter_model():
    params = RFIEmitterModelParam(location_up=1 * au.km, location_east=0 * au.km, location_north=0 * au.km)
    model = RFIEmitterModel.from_params(params)
    assert model.params == params
    antennas = ac.EarthLocation.of_site('vla')
    array_location = ac.EarthLocation.of_site('vla')
    obs_time = at.Time('2021-01-01T00:00:00', scale='utc')
    phase_tracking = ac.AltAz(
        alt=90 * au.deg,
        az=0 * au.deg,
        obstime=obs_time,
        location=array_location
    ).transform_to(
        ac.ICRS()
    )
    lmn = model.get_lmn(antennas, array_location, obs_time, phase_tracking)
    print(lmn)
