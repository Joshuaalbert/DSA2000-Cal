import numpy as np
from astropy import coordinates as ac, units as au, time as at
from jax import numpy as jnp
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_cal.geodesics.base_geodesic_model import build_geodesic_model


def test_geodesic_model():
    phase_center = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    obstimes = at.Time(['2021-01-01T00:00:00', '2021-01-01T00:00:30'], scale='utc')
    num_ant = 3
    num_time = 2
    num_source = 4
    pointings = ac.ICRS(ra=[0] * num_ant * au.deg, dec=[0] * num_ant * au.deg)
    antennas = ac.EarthLocation.from_geocentric([0] * num_ant * au.m, [0] * num_ant * au.m, [0] * num_ant * au.m)
    geodesic_model = build_geodesic_model(
        phase_center=phase_center,
        obstimes=obstimes,
        pointings=pointings,
        antennas=antennas,
        ref_time=obstimes[0],
        array_location=antennas[0]
    )

    times = jnp.linspace(0.1, 0.5, num_time)

    lmn_sources = jnp.zeros((num_source, 3))
    far_field_geodesics = geodesic_model.compute_far_field_geodesic(times, lmn_sources)
    assert np.shape(far_field_geodesics) == (num_source, num_time, num_ant, 3)
    assert np.all(np.isfinite(far_field_geodesics))

    far_field_geodesics, elevation = geodesic_model.compute_far_field_geodesic(times, lmn_sources,
                                                                               return_elevation=True)
    assert np.shape(far_field_geodesics) == (num_source, num_time, num_ant, 3)
    assert np.all(np.isfinite(far_field_geodesics))
    assert np.shape(elevation) == (num_source, num_time, num_ant)
    assert np.all(np.isfinite(elevation))

    source_positions_enu = jnp.ones((num_source, 3))
    near_field_geodesics = geodesic_model.compute_near_field_geodesics(times, source_positions_enu)
    assert np.shape(near_field_geodesics) == (num_source, num_time, num_ant, 3)

    near_field_geodesics, elevation = geodesic_model.compute_near_field_geodesics(times, source_positions_enu,
                                                                                  return_elevation=True)
    assert np.shape(near_field_geodesics) == (num_source, num_time, num_ant, 3)
    assert np.all(np.isfinite(near_field_geodesics))
    assert np.shape(elevation) == (num_source, num_time, num_ant)
    assert np.all(np.isfinite(elevation))

    # if pointing_lmn does not have ant
    pointings = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    geodesic_model = build_geodesic_model(
        phase_center=phase_center,
        obstimes=obstimes,
        pointings=pointings,
        antennas=antennas,
        ref_time=obstimes[0],
        array_location=antennas[0]
    )

    far_field_geodesics = geodesic_model.compute_far_field_geodesic(times, lmn_sources)
    assert np.shape(far_field_geodesics) == (num_source, num_time, num_ant, 3)

    source_positions_enu = jnp.zeros((num_source, 3))
    near_field_geodesics = geodesic_model.compute_near_field_geodesics(times, source_positions_enu)
    assert np.shape(near_field_geodesics) == (num_source, num_time, num_ant, 3)

    # zenith pointings
    pointings = None
    geodesic_model = build_geodesic_model(
        phase_center=phase_center,
        obstimes=obstimes,
        pointings=pointings,
        antennas=antennas,
        ref_time=obstimes[0],
        array_location=antennas[0]
    )

    far_field_geodesics = geodesic_model.compute_far_field_geodesic(times, lmn_sources)
    assert np.shape(far_field_geodesics) == (num_source, num_time, num_ant, 3)

    source_positions_enu = jnp.zeros((num_source, 3))
    near_field_geodesics = geodesic_model.compute_near_field_geodesics(times, source_positions_enu)
    assert np.shape(near_field_geodesics) == (num_source, num_time, num_ant, 3)


def test_geodesic_model_results():
    obstimes = at.Time(['2021-01-01T00:00:00', '2021-01-01T00:00:30'], scale='utc')
    antennas = ac.EarthLocation.of_site('vla').reshape((-1,))
    phase_center = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    array_location = antennas[0]
    ref_time = obstimes[0]

    geodesic_model = build_geodesic_model(
        phase_center=phase_center,
        obstimes=obstimes,
        pointings=None,
        antennas=antennas,
        ref_time=ref_time,
        array_location=array_location
    )
    times = time_to_jnp(obstimes, ref_time)

    sources = ENU(
        east=[0, 1, 0],
        north=[0, 0, 1],
        up=[1, 0, 0],
        obstime=ref_time,
        location=array_location
    ).transform_to(ac.ICRS())
    lmn_sources = quantity_to_jnp(icrs_to_lmn(sources, phase_center))

    far_field_geodesics = geodesic_model.compute_far_field_geodesic(times[0:1], lmn_sources)
    np.testing.assert_allclose(
        far_field_geodesics[0, 0, 0, :],
        [0, 0, 1],
        atol=1e-6
    )
    np.testing.assert_allclose(
        far_field_geodesics[1, 0, 0, :],
        [1, 0, 0],
        atol=1e-3
    )
    np.testing.assert_allclose(
        far_field_geodesics[2, 0, 0, :],
        [0, 1, 0],
        atol=1e-3
    )

    sources = ENU(
        east=[0, 1, 0] * au.km,
        north=[0, 0, 1] * au.km,
        up=[1, 0, 0] * au.km,
        obstime=ref_time,
        location=array_location
    )
    source_positions_enu = quantity_to_jnp(sources.cartesian.xyz.T)
    near_field_geodesics = geodesic_model.compute_near_field_geodesics(times[0:1], source_positions_enu)

    np.testing.assert_allclose(
        near_field_geodesics[0, 0, 0, :],
        [0, 0, 1],
        atol=1e-6
    )
    np.testing.assert_allclose(
        near_field_geodesics[1, 0, 0, :],
        [1, 0, 0],
        atol=1e-6
    )
    np.testing.assert_allclose(
        near_field_geodesics[2, 0, 0, :],
        [0, 1, 0],
        atol=1e-6
    )
