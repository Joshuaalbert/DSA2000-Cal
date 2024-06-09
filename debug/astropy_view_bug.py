import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pytest
from astropy.units import Quantity


def create_uvw_frame(obs_time: at.Time, phase_tracking: ac.ICRS) -> ac.SkyOffsetFrame:
    obs_location = ac.EarthLocation.from_geocentric(0 * au.m, 0 * au.m, 0 * au.m)
    obs_position, obs_velocity = obs_location.get_gcrs_posvel(obs_time)

    gcrs_frame = ac.GCRS(
        obstime=obs_time,
        obsgeoloc=obs_position,
        obsgeovel=obs_velocity
    )

    frame_uvw = ac.SkyOffsetFrame(
        origin=phase_tracking.transform_to(gcrs_frame),
        obstime=obs_time,
        obsgeoloc=obs_position,
        obsgeovel=obs_velocity
    )
    return frame_uvw


def lmn_to_icrs(lmn: Quantity, time: at.Time, phase_tracking: ac.ICRS) -> ac.ICRS:
    frame = create_uvw_frame(obs_time=time, phase_tracking=phase_tracking)
    # Swap back order to n, l, m
    cartesian_rep = ac.CartesianRepresentation(lmn[..., 2], lmn[..., 0], lmn[..., 1])
    sources = ac.SkyCoord(cartesian_rep, frame=frame).transform_to(ac.ICRS())
    # enforce instance type
    sources = ac.ICRS(sources.ra, sources.dec)
    return sources


@pytest.mark.parametrize('broadcast_time', [False, True])
@pytest.mark.parametrize('broadcast_phase_tracking', [False, True])
@pytest.mark.parametrize('broadcast_lmn', [False, True])
def test_lmn_to_icrs(broadcast_time, broadcast_phase_tracking, broadcast_lmn):
    np.random.seed(42)
    if broadcast_time:
        time = at.Time(["2021-01-01T00:00:00", "2021-01-01T00:00:00"], format='isot').reshape(
            (1, 1, 2)
        )
    else:
        time = at.Time("2021-01-01T00:00:00", format='isot')
    if broadcast_phase_tracking:
        phase_tracking = ac.ICRS([0, 0, 0, 0] * au.deg, [0, 0, 0, 0] * au.deg).reshape(
            (1, 4, 1)
        )
    else:
        phase_tracking = ac.ICRS(0 * au.deg, 0 * au.deg)
    if broadcast_lmn:
        lmn = np.random.normal(size=(5, 1, 1, 3)) * au.dimensionless_unscaled
    else:
        lmn = np.random.normal(size=(3,)) * au.dimensionless_unscaled
    lmn /= np.linalg.norm(lmn, axis=-1, keepdims=True)

    print(f"lmn shape: {lmn.shape}, time shape: {time.shape}, phase_tracking shape: {phase_tracking.shape}")
    expected_shape = np.broadcast_shapes(lmn.shape[:-1], time.shape, phase_tracking.shape)
    # print(f"expected shape: {expected_shape}")

    sources = lmn_to_icrs(lmn, time, phase_tracking)
    assert sources.shape == expected_shape
