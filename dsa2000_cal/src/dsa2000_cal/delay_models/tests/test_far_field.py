import itertools

import jax
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import time as at, coordinates as ac, units as au
from jax import numpy as jnp
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.coord_utils import earth_location_to_uvw_approx
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import time_to_jnp, quantity_to_jnp
from dsa2000_cal.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine, build_far_field_delay_engine

from dsa2000_cal.delay_models.base_far_field_delay_engine import build_far_field_delay_engine


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

    engine = build_far_field_delay_engine(
        phase_center=phase_center,
        antennas=antennas,
        start_time=time,
        end_time=time,
        ref_time=time,
        verbose=True,
        # resolution=0.01 * au.s
    )

    delay = engine.compute_delay_from_lm_jax(
        l=jnp.asarray(0.),
        m=jnp.asarray(0.),
        t1=time_to_jnp(time, time),
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

    phase_center = ENU(east=0, north=0, up=1, location=array_location, obstime=times[0]).transform_to(ac.ICRS())

    engine = build_far_field_delay_engine(
        antennas=antennas,
        phase_center=phase_center,
        start_time=times[0],
        end_time=times[-1],
        ref_time=times[0],
        verbose=True
    )
    visibilitiy_coords = engine.compute_visibility_coords(
        freqs=jnp.asarray([1.]),
        times=time_to_jnp(times, times[0]),
        with_autocorr=with_autocorr,
        convention='physical'
    )
    assert visibilitiy_coords.uvw.dtype == mp_policy.length_dtype
    assert visibilitiy_coords.freqs.dtype == mp_policy.freq_dtype
    assert visibilitiy_coords.times.dtype == mp_policy.time_dtype

    uvw = visibilitiy_coords.uvw * au.m # [T, B, 3]

    uvw_other = earth_location_to_uvw_approx(
        antennas=antennas[None, :],
        obs_time=times[:, None],
        phase_center=phase_center
    ) # [T, A, 3]
    if with_autocorr:
        antenna1, antenna2 = jnp.asarray(list(itertools.combinations_with_replacement(range(len(antennas)), 2))).T
    else:
        antenna1, antenna2 = jnp.asarray(list(itertools.combinations(range(len(antennas)), 2))).T
    uvw_other = uvw_other[:, antenna2, :] - uvw_other[:, antenna1, :] # [T, B, 3]

    if with_autocorr:
        np.testing.assert_allclose(uvw_other[0, 1, 0], 10 * au.km, atol=1 * au.m)
    else:
        np.testing.assert_allclose(uvw_other[0, 0, 0], 10 * au.km, atol=1 * au.m)
    np.testing.assert_allclose(uvw, uvw_other, atol=0.9 * au.m)

    print(engine.compute_uvw(time_to_jnp(times, times[0]), jnp.asarray([0]), jnp.asarray([1]),
                             convention='physical'))


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
    freqs = quantity_to_jnp(np.linspace(700, 2000, 4) * au.MHz)

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

    phase_center = ENU(east=0, north=0, up=1, location=array_location, obstime=start_time).transform_to(ac.ICRS())

    engine = build_far_field_delay_engine(
        antennas=antennas,
        phase_center=phase_center,
        start_time=start_time,
        end_time=end_time,
        ref_time=start_time,
        verbose=True,
        resolution=0.1 * au.s
    )
    vis_coords = jax.jit(engine.compute_visibility_coords, static_argnames=['with_autocorr'])(
        times=time_to_jnp(times, times[0]),
        freqs=freqs,
        with_autocorr=False
    )
    uvw0 = np.asarray(vis_coords.uvw) * au.m # [T, B, 3]

    engine = build_far_field_delay_engine(
        antennas=antennas,
        phase_center=phase_center,
        start_time=start_time,
        end_time=end_time,
        ref_time=start_time,
        verbose=True
    )
    vis_coords = jax.jit(engine.compute_visibility_coords, static_argnames=['with_autocorr'])(
        times=time_to_jnp(times, times[0]),
        freqs=freqs,
        with_autocorr=False
    )
    uvw = np.asarray(vis_coords.uvw) * au.m

    error = jnp.linalg.norm(uvw[:, 0, :] - uvw0[:, 0, :], axis=-1)

    print(f"Max error {np.max(error)} on {baseline}")

    assert np.max(error) < 1e-3 * au.m, f"Max error {np.max(error)}"  # <1 mm error due to interpolation

    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True, squeeze=False)

    for resolution in [5 * au.s, 10 * au.s, 50 * au.s, 100 * au.s, 200 * au.s]:
        engine = build_far_field_delay_engine(
            antennas=antennas,
            phase_center=phase_center,
            start_time=start_time,
            end_time=end_time,
            ref_time=start_time,
            verbose=True,
            resolution=resolution
        )
        vis_coords = jax.jit(engine.compute_visibility_coords, static_argnames=['with_autocorr'])(
            times=time_to_jnp(times, times[0]),
            freqs=freqs,
            with_autocorr=False
        )
        uvw = np.asarray(vis_coords) * au.m
        axs[0, 0].plot(times.jd, uvw[:, 0, 0], label=f'{resolution}')
        axs[1, 0].plot(times.jd, jnp.linalg.norm(uvw[:, 0, :] - uvw0[:, 0, :], axis=-1), label=f'{resolution}')
    axs[0, 0].set_ylabel('u (m)')
    axs[1, 0].set_ylabel('UVW error (m)')
    axs[1, 0].legend()
    fig.tight_layout()
    plt.show()
    plt.close('all')


def test_build_far_field_delay_engine():
    far_field_delay_model = build_far_field_delay_engine(
        antennas=ac.ITRS(
            x=[0, 0, 0] * au.m, y=[1, 1, 1] * au.m, z=[0, 0, 0] * au.m,
            obstime=at.Time.now()
        ).earth_location,
        start_time=at.Time.now(),
        end_time=at.Time.now() + 1 * au.s,
        ref_time=at.Time.now(),
        phase_center=ac.ICRS(ra=0 * au.deg, dec=0 * au.deg),
        resolution=1 * au.s,
        verbose=True
    )
    print(far_field_delay_model)

    @jax.jit
    def f(ffdm: BaseFarFieldDelayEngine):
        return ffdm

    ffdm = f(far_field_delay_model)

    print(ffdm)

    # Test with lm_offset
    num_ant = 3
    lm_offset = au.Quantity(np.random.uniform(-10, 10, (num_ant, 2)), unit=au.arcmin)

    far_field_delay_model = build_far_field_delay_engine(
        antennas=ac.ITRS(
            x=[0, 0, 0] * au.m, y=[1, 1, 1] * au.m, z=[0, 0, 0] * au.m,
            obstime=at.Time.now()
        ).earth_location,
        start_time=at.Time.now(),
        end_time=at.Time.now() + 1 * au.s,
        ref_time=at.Time.now(),
        phase_center=ac.ICRS(ra=0 * au.deg, dec=0 * au.deg),
        resolution=1 * au.s,
        verbose=True,
        lm_offset=lm_offset
    )

    print(far_field_delay_model)
