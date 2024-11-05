import itertools

import astropy.constants as const
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pytest
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import time_to_jnp, quantity_to_jnp
from dsa2000_cal.common.types import VisibilityCoords
from dsa2000_cal.common.wgridder import image_to_vis
from dsa2000_cal.delay_models.base_far_field_delay_engine import build_far_field_delay_engine
from dsa2000_cal.delay_models.base_near_field_delay_engine import build_near_field_delay_engine
from dsa2000_cal.delay_models.uvw_utils import perley_icrs_from_lmn
from dsa2000_cal.geodesics.base_geodesic_model import build_geodesic_model
from dsa2000_cal.visibility_model.source_models.celestial.base_point_source_model import build_point_source_model


def build_mock_point_source_model(num_freqs: int, num_source: int, full_stokes: bool,
                                  phase_tracking: ac.ICRS):
    model_freqs = np.linspace(700, 2000, num_freqs) * au.MHz

    # Wgridder test data

    N = 512
    max_baseline = 20. * au.km
    min_wavelength = const.c / model_freqs.max()

    diff_scale = float((min_wavelength / max_baseline).to(au.dimensionless_unscaled))
    dl = dm = diff_scale / 7.
    l0 = m0 = 0.

    lvec = (-0.5 * N + np.arange(N)) * dl + l0
    mvec = (-0.5 * N + np.arange(N)) * dm + m0
    L, M = np.meshgrid(lvec, mvec, indexing='ij')
    dirty = np.zeros((N, N))
    for i in range(num_source):
        dirty[N // (i + 2), N // (i + 2)] = 1.

    select = np.where(dirty)
    l = L[select]
    m = M[select]
    n = np.sqrt(1. - l ** 2 - m ** 2)

    ra, dec = perley_icrs_from_lmn(l, m, n, phase_tracking.ra.rad, phase_tracking.dec.rad)
    ra = np.asarray(ra) * au.rad
    dec = np.asarray(dec) * au.rad

    wgridder_data = dict(
        center_l=l0,
        center_m=m0,
        pixsize_l=dl,
        pixsize_m=dm,
        dirty=dirty
    )

    ## Mock model data

    if full_stokes:
        A = np.ones((num_freqs, num_source, 2, 2)) * au.Jy
    else:
        A = np.ones((num_freqs, num_source)) * au.Jy
    model_data = build_point_source_model(
        model_freqs=model_freqs,
        ra=ra,
        dec=dec,
        A=A
    )

    return model_data, wgridder_data


def build_mock_obs_setup(ant: int, time: int):
    array_location = ac.EarthLocation.of_site('vla')
    ref_time = at.Time('2021-01-01T00:00:00', scale='utc')
    obstimes = ref_time + np.arange(time) * au.s
    phase_tracking = ENU(0, 0, 1, location=array_location, obstime=ref_time).transform_to(ac.ICRS())
    pointing = phase_tracking
    antennas = ENU(
        east=np.random.uniform(low=-10, high=10, size=ant) * au.km,
        north=np.random.uniform(low=-10, high=10, size=ant) * au.km,
        up=np.random.uniform(low=-10, high=10, size=ant) * au.m,
        location=array_location,
        obstime=ref_time
    ).transform_to(ac.ITRS(location=array_location, obstime=ref_time)).earth_location

    geodesic_model = build_geodesic_model(
        antennas=antennas,
        array_location=array_location,
        phase_center=phase_tracking,
        obstimes=obstimes,
        ref_time=ref_time,
        pointings=pointing
    )

    far_field_delay_engine = build_far_field_delay_engine(
        antennas=antennas,
        phase_center=phase_tracking,
        start_time=obstimes.min(),
        end_time=obstimes.max(),
        ref_time=ref_time
    )

    near_field_delay_engine = build_near_field_delay_engine(
        antennas=antennas,
        start_time=obstimes.min(),
        end_time=obstimes.max(),
        ref_time=ref_time
    )

    baseline_pairs = np.asarray(list(itertools.combinations_with_replacement(range(ant), 2)),
                                dtype=np.int32)
    num_baselines = len(baseline_pairs)
    antenna_1 = baseline_pairs[:, 0]
    antenna_2 = baseline_pairs[:, 1]
    time_obs = time_to_jnp(obstimes, ref_time)

    time_obs = np.tile(time_obs[:, None], (1, num_baselines)).flatten()
    antenna_1 = np.tile(antenna_1[None, :], (time, 1)).flatten()
    antenna_2 = np.tile(antenna_2[None, :], (time, 1)).flatten()
    time_idx = np.tile(np.arange(time)[:, None], (1, num_baselines)).flatten()
    uvw = far_field_delay_engine.compute_uvw_jax(
        times=time_obs,
        antenna_1=antenna_1,
        antenna_2=antenna_2
    )

    visibility_coords = VisibilityCoords(
        uvw=mp_policy.cast_to_length(uvw),
        time_obs=mp_policy.cast_to_time(time_obs),
        antenna_1=mp_policy.cast_to_index(antenna_1),
        antenna_2=mp_policy.cast_to_index(antenna_2),
        time_idx=mp_policy.cast_to_index(time_idx)
    )
    return phase_tracking, time_obs, visibility_coords, geodesic_model, far_field_delay_engine, near_field_delay_engine


@pytest.mark.parametrize("full_stokes", [True, False])
def test_point_predict(full_stokes: bool):
    time = 15
    ant = 24
    phase_tracking, time_obs, visibility_coords, geodesic_model, far_field_delay_engine, near_field_delay_engine = build_mock_obs_setup(
        ant, time
    )
    row = len(visibility_coords.antenna_1)

    num_model_freqs = 3
    num_freqs = 4
    num_sources = 5
    point_source_model, wgridder_data = build_mock_point_source_model(num_model_freqs, num_sources, full_stokes,
                                                                      phase_tracking)

    freqs = quantity_to_jnp(np.linspace(700, 2000, num_freqs) * au.MHz)
    if full_stokes:
        assert point_source_model.is_full_stokes()
    else:
        assert not point_source_model.is_full_stokes()

    model_data = point_source_model.get_model_data(freqs, time_obs, geodesic_model)
    assert np.shape(model_data.freqs) == (num_freqs,)
    assert np.shape(model_data.times) == (1,)
    assert np.shape(model_data.lmn) == (num_sources, 3)
    if full_stokes:
        assert np.shape(model_data.image) == (1, num_freqs, num_sources, 2, 2)
    else:
        assert np.shape(model_data.image) == (1, num_freqs, num_sources)

    visibilities = point_source_model.predict(model_data=model_data, visibility_coords=visibility_coords,
                                              gain_model=None, near_field_delay_engine=near_field_delay_engine,
                                              far_field_delay_engine=far_field_delay_engine,
                                              geodesic_model=geodesic_model
                                              )
    assert np.all(np.isfinite(visibilities))
    if full_stokes:
        assert np.shape(visibilities) == (row, num_freqs, 2, 2)
    else:
        assert np.shape(visibilities) == (row, num_freqs)

    wgridder_vis = image_to_vis(
        uvw=visibility_coords.uvw,
        freqs=freqs,
        epsilon=1e-4,
        **wgridder_data
    )

    if full_stokes:
        np.testing.assert_allclose(wgridder_vis.real, visibilities.real[..., 0, 0], atol=1e-3)
        np.testing.assert_allclose(wgridder_vis.imag, visibilities.imag[..., 0, 0], atol=1e-3)
    else:
        np.testing.assert_allclose(wgridder_vis.real, visibilities.real, atol=1e-3)
        np.testing.assert_allclose(wgridder_vis.imag, visibilities.imag, atol=1e-3)
