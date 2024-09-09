import itertools

import jax
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import time as at, coordinates as ac, units as au
from jax import numpy as jnp
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.coord_utils import earth_location_to_uvw_approx
from dsa2000_cal.delay_models.far_field import FarFieldDelayEngine



def test_far_field_delay_engine():
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

    phase_center = ENU(east=1, north=0, up=0, location=array_location, obstime=time).transform_to(ac.ICRS())

    engine = FarFieldDelayEngine(
        phase_center=phase_center,
        antennas=antennas,
        start_time=time,
        end_time=time,
        verbose=True,
        # resolution=0.01 * au.s
    )

    delay = engine.compute_delay_from_lm_jax(
        l=jnp.asarray(0.),
        m=jnp.asarray(0.),
        t1=engine.time_to_jnp(time),
        i1=jnp.asarray(0),
        i2=jnp.asarray(1),
    )

    assert np.shape(delay) == ()

    print(delay)
    # 64 bit -- 999.9988935488057
    # 32 bit -- 999.9988935488057

    np.testing.assert_allclose(delay, 1000., atol=0.55)



@pytest.mark.parametrize('with_autocorr', [True, False])
def test_compute_uvw(with_autocorr):
    times = at.Time(["2021-01-01T00:00:00"], scale='utc')
    array_location = ac.EarthLocation.of_site('vla')
    antennas = ENU(
        east=[0, 10] * au.km,
        north=[0, 0] * au.km,
        up=[0, 0] * au.km,
        location=array_location,
        obstime=times[0]
    )
    antennas = antennas.transform_to(ac.ITRS(obstime=times[0], location=array_location)).earth_location

    phase_centre = ENU(east=0, north=0, up=1, location=array_location, obstime=times[0]).transform_to(ac.ICRS())

    engine = FarFieldDelayEngine(
        antennas=antennas,
        phase_center=phase_centre,
        start_time=times[0],
        end_time=times[-1],
        verbose=True
    )
    visibilitiy_coords = engine.compute_visibility_coords(
        times=engine.time_to_jnp(times),
        with_autocorr=with_autocorr,
        convention='physical'
    )
    uvw = visibilitiy_coords.uvw[None, :, :] * au.m

    uvw_other = earth_location_to_uvw_approx(
        antennas=antennas[None, :],
        obs_time=times[:, None],
        phase_tracking=phase_centre
    )
    if with_autocorr:
        antenna_1, antenna_2 = jnp.asarray(list(itertools.combinations_with_replacement(range(len(antennas)), 2))).T
    else:
        antenna_1, antenna_2 = jnp.asarray(list(itertools.combinations(range(len(antennas)), 2))).T
    uvw_other = uvw_other[:, antenna_2, :] - uvw_other[:, antenna_1, :]

    if with_autocorr:
        np.testing.assert_allclose(uvw_other[0, 1, 0], 10 * au.km, atol=1 * au.m)
    else:
        np.testing.assert_allclose(uvw_other[0, 0, 0], 10 * au.km, atol=1 * au.m)
    np.testing.assert_allclose(uvw, uvw_other, atol=0.9 * au.m)

    print(engine.compute_uvw_jax(engine.time_to_jnp(times), jnp.asarray([0]), jnp.asarray([1]),
                                 convention='physical'))



@pytest.mark.parametrize('time', [at.Time("2024-01-01T00:00:00", scale='utc'),
                                  at.Time("2024-04-01T00:00:00", scale='utc'),
                                  at.Time("2024-07-01T00:00:00", scale='utc'),
                                  at.Time("2024-10-01T00:00:00", scale='utc')])
@pytest.mark.parametrize('baseline', [3 * au.km, 10 * au.km])
def test_aberated_plane_of_sky(time: at.Time, baseline: au.Quantity):
    # aberation happens when uvw coordinates are assumed to be consistent for all points in the sky, however
    # tau = (-?) c * delay = u l + v m + w sqrt(1 - l^2 - m^2) ==> w = tau(l=0, m=0)
    # d/dl tau = u + w l / sqrt(1 - l^2 - m^2) ==> u = d/dl tau(l=0, m=0)
    # d/dm tau = v + w m / sqrt(1 - l^2 - m^2) ==> v = d/dm tau(l=0, m=0)
    # only true for l=m=0.

    # Let us see the error in delay for the approximation tau(l,m) = u*l + v*m + w*sqrt(1 - l^2 - m^2)
    array_location = ac.EarthLocation.of_site('vla')
    antennas = ENU(
        east=[0, baseline.to('km').value] * au.km,
        north=[0, 0] * au.km,
        up=[0, 0] * au.km,
        location=array_location,
        obstime=time
    )
    antennas = antennas.transform_to(ac.ITRS(obstime=time)).earth_location

    phase_centre = ENU(east=0, north=0, up=1, location=array_location, obstime=time).transform_to(ac.ICRS())

    engine = FarFieldDelayEngine(
        antennas=antennas,
        phase_center=phase_centre,
        start_time=time,
        end_time=time,
        verbose=True
    )
    uvw = engine.compute_uvw_jax(
        times=engine.time_to_jnp(time[None]),
        antenna_1=jnp.asarray([0]),
        antenna_2=jnp.asarray([1])
    )
    uvw = uvw * au.m

    lvec = mvec = jnp.linspace(-1, 1, 100)
    M, L = jnp.meshgrid(lvec, mvec, indexing='ij')
    N = jnp.sqrt(1 - L ** 2 - M ** 2)
    tau_approx = uvw[0, 0] * L + uvw[0, 1] * M + uvw[0, 2] * N
    tau_approx = tau_approx.to('m').value

    tau_exact = jax.vmap(
        lambda l, m: engine.compute_delay_from_lm_jax(
            l=l, m=m,
            t1=engine.time_to_jnp(time),
            i1=jnp.asarray(0),
            i2=jnp.asarray(1))
    )(L.ravel(), M.ravel()).reshape(L.shape)

    tau_diff = tau_exact - tau_approx

    # Plot exact, approx, and difference
    fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True, sharey=True, squeeze=False)

    im = axs[0, 0].imshow(tau_exact,
                          origin='lower',
                          extent=(lvec.min(), lvec.max(), mvec.min(), mvec.max()),
                          interpolation='nearest',
                          cmap='PuOr'
                          )
    axs[0, 0].set_title('Exact delay')
    fig.colorbar(im, ax=axs[0, 0],
                 label='Light travel dist. (m)')

    im = axs[1, 0].imshow(tau_approx,
                          origin='lower',
                          extent=(lvec.min(), lvec.max(), mvec.min(), mvec.max()),
                          interpolation='nearest',
                          cmap='PuOr'
                          )
    axs[1, 0].set_title('Approximated delay')
    fig.colorbar(im, ax=axs[1, 0],
                 label='Light travel dist. (m)')

    im = axs[2, 0].imshow(tau_diff,
                          origin='lower',
                          extent=(lvec.min(), lvec.max(), mvec.min(), mvec.max()),
                          interpolation='nearest',
                          cmap='PuOr'
                          )
    axs[2, 0].set_title(f'Difference: {time}')
    fig.colorbar(im, ax=axs[2, 0],
                 label='Light travel dist. (m)')

    axs[0, 0].set_ylabel('m')
    axs[1, 0].set_ylabel('m')
    axs[2, 0].set_ylabel('m')
    axs[2, 0].set_xlabel('l')

    fig.tight_layout()
    plt.show()

    freq = 70e6 * au.Hz
    difference_deg = tau_diff * freq.to('Hz').value / 299792458.0 * 180 / np.pi

    # The difference in delay in radians
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True, squeeze=False)

    im = axs[0, 0].imshow(difference_deg,
                          origin='lower',
                          extent=(lvec.min(), lvec.max(), mvec.min(), mvec.max()),
                          interpolation='nearest',
                          cmap='coolwarm'
                          )
    # structure time like "1 Jan, 2024"
    axs[0, 0].set_title(
        fr'$\Delta \tau(l,m)$ over {baseline.to("km")} | {freq.to("MHz")} | {time.to_datetime().strftime("%d %b, %Y")}')
    fig.colorbar(im, ax=axs[0, 0],
                 label='Phase difference (deg)')

    axs[0, 0].set_ylabel('m')
    axs[0, 0].set_xlabel('l')

    fig.tight_layout()
    plt.show()



@pytest.mark.parametrize('baseline', [10 * au.km, 100 * au.km, 1000 * au.km])
def test_resolution_error(baseline: au.Quantity):
    # aberation happens when uvw coordinates are assumed to be consistent for all points in the sky, however
    # tau = (-?) c * delay = u l + v m + w sqrt(1 - l^2 - m^2) ==> w = tau(l=0, m=0)
    # d/dl tau = u + w l / sqrt(1 - l^2 - m^2) ==> u = d/dl tau(l=0, m=0)
    # d/dm tau = v + w m / sqrt(1 - l^2 - m^2) ==> v = d/dm tau(l=0, m=0)
    # only true for l=m=0.

    start_time = at.Time("2024-01-01T00:00:00", scale='utc')
    end_time = start_time + (300 * au.s)

    times = start_time + np.arange(0, 300, 1) * au.s

    # Let us see the error in delay for the approximation tau(l,m) = u*l + v*m + w*sqrt(1 - l^2 - m^2)
    array_location = ac.EarthLocation.of_site('vla')
    antennas = ENU(
        east=[0, baseline.to('km').value] * au.km,
        north=[0, 0] * au.km,
        up=[0, 0] * au.km,
        location=array_location,
        obstime=start_time
    )
    antennas = antennas.transform_to(ac.ITRS(obstime=start_time)).earth_location

    phase_centre = ENU(east=0, north=0, up=1, location=array_location, obstime=start_time).transform_to(ac.ICRS())

    engine = FarFieldDelayEngine(
        antennas=antennas,
        phase_center=phase_centre,
        start_time=start_time,
        end_time=end_time,
        verbose=True,
        resolution=0.1 * au.s
    )
    vis_coords = jax.jit(engine.compute_visibility_coords, static_argnames=['with_autocorr'])(
        times=engine.time_to_jnp(times),
        with_autocorr=False
    )
    uvw0 = vis_coords.uvw.reshape((len(times), -1, 3)) * au.m

    engine = FarFieldDelayEngine(
        antennas=antennas,
        phase_center=phase_centre,
        start_time=start_time,
        end_time=end_time,
        verbose=True
    )
    vis_coords = jax.jit(engine.compute_visibility_coords, static_argnames=['with_autocorr'])(
        times=engine.time_to_jnp(times),
        with_autocorr=False
    )
    uvw = vis_coords.uvw.reshape((len(times), -1, 3)) * au.m

    error = jnp.linalg.norm(uvw[:, 0, :] - uvw0[:, 0, :], axis=-1)

    print(f"Max error {np.max(error)} on {baseline}")

    assert np.max(error) < 1e-3 * au.m, f"Max error {np.max(error)}"  # <1 mm error due to interpolation

    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True, squeeze=False)

    for resolution in [5 * au.s, 10 * au.s, 50 * au.s, 100 * au.s, 200 * au.s]:
        engine = FarFieldDelayEngine(
            antennas=antennas,
            phase_center=phase_centre,
            start_time=start_time,
            end_time=end_time,
            verbose=True,
            resolution=resolution
        )
        vis_coords = jax.jit(engine.compute_visibility_coords, static_argnames=['with_autocorr'])(
            times=engine.time_to_jnp(times),
            with_autocorr=False
        )
        uvw = vis_coords.uvw.reshape((len(times), -1, 3)) * au.m
        axs[0, 0].plot(times.jd, uvw[:, 0, 0], label=f'{resolution}')
        axs[1, 0].plot(times.jd, jnp.linalg.norm(uvw[:, 0, :] - uvw0[:, 0, :], axis=-1), label=f'{resolution}')
    axs[0, 0].set_ylabel('u (m)')
    axs[1, 0].set_ylabel('UVW error (m)')
    axs[1, 0].legend()
    fig.tight_layout()
    plt.show()
    plt.close('all')
