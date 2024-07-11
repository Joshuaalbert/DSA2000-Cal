import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from astropy import coordinates as ac, units as au, time as at
from tomographic_kernel.frames import ENU

from dsa2000_cal.uvw.near_field import NearFieldDelayEngine


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

    antennas = antennas.transform_to(ac.ITRS(obstime=time, location=array_location)).earth_location
    emitter = ENU(
        east=10 * au.km,
        north=0 * au.km,
        up=0 * au.km,
        obstime=time,
        location=array_location
    )

    engine = NearFieldDelayEngine(
        antennas=antennas,
        start_time=time,
        end_time=time,
        verbose=True,
        resolution=0.01 * au.s
    )

    delay, dist2_, dist1_ = engine.compute_delay_from_emitter_jax(
        emitter,
        engine.time_to_jnp(time),
        jnp.asarray(0),
        jnp.asarray(1)
    )

    delay_proj, dist2, dist1 = engine.compute_delay_from_projection_jax(
        jnp.asarray(10.),
        jnp.asarray(0.),
        jnp.asarray(0.),
        engine.time_to_jnp(time),
        jnp.asarray(0),
        jnp.asarray(1)
    )
    assert np.shape(delay) == ()
    assert np.shape(delay_proj) == ()
    np.testing.assert_allclose(delay, 1000, atol=0.55)
    np.testing.assert_allclose(delay_proj, 1000, atol=0.55)
    np.testing.assert_allclose(dist2, dist2_, atol=0.002)
    np.testing.assert_allclose(dist1, dist1_, atol=0.002)

    # vector version
    for n in [1, 6]:
        emitter = ENU(
            east=[10] * n * au.km,
            north=[0] * n * au.km,
            up=[0] * n * au.km,
            obstime=time,
            location=array_location
        )
        delay, dist2, dist1 = engine.compute_delay_from_emitter_jax(
            emitter,
            engine.time_to_jnp(time),
            jnp.asarray(0),
            jnp.asarray(1)
        )
        assert np.shape(delay) == (n,)
        delay_proj, _, _ = engine.compute_delay_from_projection_jax(
            jnp.asarray([10.] * n),
            jnp.asarray([0.] * n),
            jnp.asarray([0.] * n),
            engine.time_to_jnp(time),
            jnp.asarray(0),
            jnp.asarray(1)
        )
        assert np.shape(delay_proj) == (n,)
