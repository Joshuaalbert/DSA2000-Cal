import jax.numpy as jnp
import numpy as np
from astropy import coordinates as ac, units as au, time as at
from tomographic_kernel.frames import ENU

from dsa2000_common.common.quantity_utils import time_to_jnp
from dsa2000_common.delay_models.base_near_field_delay_engine import build_near_field_delay_engine


def test_near_field():
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    array_location = ac.EarthLocation.of_site('vla')
    antennas = ENU(
        east=[0, 1] * au.km,
        north=[0, 0] * au.km,
        up=[0, 0] * au.km,
        location=array_location,
        obstime=time
    )
    np.testing.assert_allclose(np.linalg.norm(np.diff(antennas.cartesian.xyz, axis=-1), axis=0), 1 * au.km,
                               atol=1e-3 * au.m)

    antennas = antennas.transform_to(ac.ITRS(obstime=time, location=array_location)).earth_location

    np.testing.assert_allclose(np.linalg.norm(np.diff(antennas.get_itrs().cartesian.xyz, axis=-1), axis=0), 1 * au.km,
                               atol=1e-3 * au.m)

    engine = build_near_field_delay_engine(
        antennas=antennas,
        start_time=time,
        end_time=time,
        ref_time=time,
        verbose=True,
        # resolution=0.01 * au.s
    )
    emitter = ENU(
        east=10 * au.km,
        north=0 * au.km,
        up=0 * au.km,
        obstime=time,
        location=array_location
    ).transform_to(
        ac.ITRS(
            obstime=time,
            location=array_location
        )
    ).earth_location

    x_0_gcrs = engine.construct_x_0_gcrs(
        interp_times=time[None],
        ref_time=time,
        emitter=emitter,
        regular_grid=True
    )

    delay, dist2, dist1 = engine.compute_delay(
        x_0_gcrs,
        time_to_jnp(time, time),
        jnp.asarray(0),
        jnp.asarray(1),
    )

    delay_proj, dist2_proj, dist1_proj = engine.compute_delay(
        engine.construct_x_0_gcrs_from_projection(jnp.asarray(10e3),
                                                  jnp.asarray(0.),
                                                  jnp.asarray(0.)),
        time_to_jnp(time, time),
        jnp.asarray(0),
        jnp.asarray(1)
    )  # delay is dist to travel from 1 to 0
    assert np.shape(delay) == (1,)
    assert np.shape(delay_proj) == (1,)

    print(delay_proj, dist1_proj, dist2_proj)
    print(delay, dist1, dist2)
    np.testing.assert_allclose(dist1_proj, 10e3)
    np.testing.assert_allclose(dist2_proj, 9e3)
    np.testing.assert_allclose(delay_proj, 1000, atol=0.55)

    np.testing.assert_allclose(dist1, 10e3)
    np.testing.assert_allclose(dist2, 9e3)
    np.testing.assert_allclose(delay, 1000, atol=0.55)

    # vector version
    for n in [1, 6]:
        emitter = ENU(
            east=[10] * n * au.km,
            north=[0] * n * au.km,
            up=[0] * n * au.km,
            obstime=time,
            location=array_location
        )
        emitter = emitter.transform_to(ac.ITRS(obstime=time, location=array_location)).earth_location
        delay, dist2_proj, dist1_proj = engine.compute_delay(
            engine.construct_x_0_gcrs(
                interp_times=time[None],
                ref_time=time,
                emitter=emitter,
                regular_grid=True
            ),
            time_to_jnp(time, time),
            jnp.asarray(0),
            jnp.asarray(1)
        )
        assert np.shape(delay) == (n,)
        delay_proj, _, _ = engine.compute_delay(
            engine.construct_x_0_gcrs_from_projection(
                jnp.asarray([10e3] * n),
                jnp.asarray([0.] * n),
                jnp.asarray([0.] * n)),
            time_to_jnp(time, time),
            jnp.asarray(0),
            jnp.asarray(1)
        )
        assert np.shape(delay_proj) == (n,)
        np.testing.assert_allclose(delay_proj, delay, atol=0.001)
