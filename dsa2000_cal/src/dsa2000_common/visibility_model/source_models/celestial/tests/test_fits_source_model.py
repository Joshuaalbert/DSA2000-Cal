import astropy.constants as const
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pytest
from dsa2000_common.common.enu_frame import ENU

from dsa2000_common.common.ellipse_utils import Gaussian
from dsa2000_common.common.quantity_utils import time_to_jnp, quantity_to_jnp
from dsa2000_common.common.wgridder import image_to_vis
from dsa2000_common.delay_models.base_far_field_delay_engine import build_far_field_delay_engine
from dsa2000_common.delay_models.base_near_field_delay_engine import build_near_field_delay_engine
from dsa2000_common.gain_models.base_spherical_interpolator import build_spherical_interpolator
from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model
from dsa2000_common.visibility_model.source_models.celestial.base_fits_source_model import build_fits_source_model, \
    divide_fits_into_facets


def build_mock_fits_source_model(num_model_freqs: int, num_source: int, full_stokes: bool,
                                 phase_center: ac.ICRS, num_facets_per_side):
    model_freqs = np.linspace(700, 2000, num_model_freqs) * au.MHz

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
    l = np.asarray([lvec[N // (i + 2)] for i in range(num_source)])
    m = np.asarray([mvec[N // (i + 2)] for i in range(num_source)])
    major_axis = dl * 20 * np.ones(num_source)
    minor_axis = dm * 10 * np.ones(num_source)
    pos_angle = np.random.uniform(low=0, high=np.pi, size=num_source)

    dirty = np.zeros((N, N))
    lm = np.stack([L.flatten(), M.flatten()], axis=-1)

    for i in range(num_source):
        gaussian = Gaussian(
            x0=np.stack([l[i], m[i]]),
            major_fwhm=major_axis[i],
            minor_fwhm=minor_axis[i],
            pos_angle=pos_angle[i],
            total_flux=1.0
        )
        dirty += jax.vmap(gaussian.compute_flux_density)(lm).reshape(L.shape) * dl * dm

    wgridder_data = dict(
        center_l=l0,
        center_m=m0,
        pixsize_l=dl,
        pixsize_m=dm,
        dirty=dirty
    )

    ## Mock model data

    if full_stokes:
        images = np.tile(dirty[None, :, :, None, None], (num_model_freqs, 1, 1, 2, 2)) * au.Jy
    else:
        images = np.tile(dirty[None, :, :], (num_model_freqs, 1, 1)) * au.Jy

    ras = au.Quantity(np.tile(phase_center.ra[None], (num_model_freqs,)))
    decs = au.Quantity(np.tile(phase_center.dec[None], (num_model_freqs,)))
    dls = au.Quantity(np.tile([dl] * au.rad, (num_model_freqs,)))
    dms = au.Quantity(np.tile([dm] * au.rad, (num_model_freqs,)))

    images, ras, decs, dls, dms = divide_fits_into_facets(
        images, ras, decs, dls, dms, num_facets_per_side
    )

    if full_stokes:
        assert np.shape(images) == (
            num_facets_per_side ** 2, num_model_freqs, N // num_facets_per_side, N // num_facets_per_side, 2, 2)
    else:
        assert np.shape(images) == (
            num_facets_per_side ** 2, num_model_freqs, N // num_facets_per_side, N // num_facets_per_side)
    assert np.shape(ras) == (num_facets_per_side ** 2, num_model_freqs)
    assert np.shape(decs) == (num_facets_per_side ** 2, num_model_freqs)
    assert np.shape(dls) == (num_facets_per_side ** 2, num_model_freqs)
    assert np.shape(dms) == (num_facets_per_side ** 2, num_model_freqs)

    model_data = build_fits_source_model(
        model_freqs=model_freqs,
        image=images,
        ra=ras,
        dec=decs,
        dl=dls,
        dm=dms
    )

    model_data.plot(phase_center=phase_center)
    return model_data, wgridder_data


def build_mock_gain_model(with_gains, full_stokes, antennas: ac.EarthLocation):
    if with_gains:
        model_freqs = np.linspace(700, 2000, 5) * au.MHz
        model_theta = np.linspace(0, np.pi, 5) * au.rad
        model_phi = np.linspace(0, 2 * np.pi, 5) * au.rad
        ref_time = at.Time('2021-01-01T00:00:00', scale='utc')
        model_times = ref_time + np.arange(2) * au.s
        if full_stokes:
            model_gains = np.ones(
                (len(model_times), len(model_theta), len(model_freqs), 2, 2)
            ) * au.dimensionless_unscaled  # [num_model_times, num_model_dir, [num_ant,] num_model_freqs, 2, 2]
            model_gains[..., 0, 1] *= 0.
            model_gains[..., 1, 0] *= 0.

        else:
            model_gains = np.ones(
                (len(model_times), len(model_theta), len(model_freqs))
            ) * au.dimensionless_unscaled  # [num_model_times, num_model_dir, [num_ant,] num_model_freqs]
        return build_spherical_interpolator(
            antennas=antennas,
            model_freqs=model_freqs,
            model_theta=model_theta,
            model_phi=model_phi,
            model_times=model_times,
            model_gains=model_gains,
            ref_time=ref_time,
            tile_antennas=True,
        )
    else:
        return None


def build_mock_obs_setup(ant: int, time: int, num_freqs: int):
    array_location = ac.EarthLocation.of_site('vla')
    ref_time = at.Time('2021-01-01T00:00:00', scale='utc')
    obstimes = ref_time + np.arange(time) * au.s
    phase_center = ENU(0, 0, 1, location=array_location, obstime=ref_time).transform_to(ac.ICRS())
    freqs = np.linspace(700, 2000, num_freqs) * au.MHz

    pointing = phase_center
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
        phase_center=phase_center,
        obstimes=obstimes,
        ref_time=ref_time,
        pointings=pointing
    )

    far_field_delay_engine = build_far_field_delay_engine(
        antennas=antennas,
        phase_center=phase_center,
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

    visibility_coords = far_field_delay_engine.compute_visibility_coords(
        freqs=quantity_to_jnp(freqs),
        times=time_to_jnp(obstimes, ref_time)
    )
    return phase_center, antennas, visibility_coords, geodesic_model, far_field_delay_engine, near_field_delay_engine


@pytest.mark.parametrize("full_stokes", [True, False])
@pytest.mark.parametrize("with_gains", [True, False])
def test_run_fits_predict(full_stokes: bool, with_gains: bool):
    time = 15
    ant = 100
    num_freqs = 4
    phase_center, antennas, visibility_coords, geodesic_model, far_field_delay_engine, near_field_delay_engine = build_mock_obs_setup(
        ant, time, num_freqs
    )
    gain_model = build_mock_gain_model(with_gains, full_stokes, antennas)
    num_times, num_baselines, _ = np.shape(visibility_coords.uvw)

    num_model_freqs = 3
    num_sources = 5
    num_facets_per_side = 2
    fits_source_model, wgridder_data = build_mock_fits_source_model(num_model_freqs, num_sources, full_stokes,
                                                                    phase_center, num_facets_per_side)

    if full_stokes:
        assert fits_source_model.is_full_stokes()
    else:
        assert not fits_source_model.is_full_stokes()

    visibilities_np = fits_source_model.predict_np(
        visibility_coords=visibility_coords,
        gain_model=gain_model,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        geodesic_model=geodesic_model
    )

    visibilities = fits_source_model.predict(
        visibility_coords=visibility_coords,
        gain_model=gain_model,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        geodesic_model=geodesic_model
    )

    np.testing.assert_allclose(visibilities_np, visibilities, atol=1e-3)
    assert np.all(np.isfinite(visibilities))
    if full_stokes:
        assert np.shape(visibilities) == (num_times, num_baselines, num_freqs, 2, 2)
    else:
        assert np.shape(visibilities) == (num_times, num_baselines, num_freqs)

    wgridder_vis = image_to_vis(
        uvw=visibility_coords.uvw.reshape((-1, 3)),
        freqs=visibility_coords.freqs,
        epsilon=1e-4,
        **wgridder_data
    )
    wgridder_vis = wgridder_vis.reshape((num_times, num_baselines, num_freqs))

    if full_stokes:
        np.testing.assert_allclose(wgridder_vis.real, visibilities.real[..., 0, 0], atol=1e-3)
        np.testing.assert_allclose(wgridder_vis.imag, visibilities.imag[..., 0, 0], atol=1e-3)
    else:
        np.testing.assert_allclose(wgridder_vis.real, visibilities.real, atol=1e-3)
        np.testing.assert_allclose(wgridder_vis.imag, visibilities.imag, atol=1e-3)

    def f(sm):
        return sm

    f_jit = jax.jit(f).lower(fits_source_model).compile()

    jax.block_until_ready(f_jit(fits_source_model))
