import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.astropy_utils import create_spherical_spiral_grid
from dsa2000_common.common.coord_utils import icrs_to_lmn
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.mixed_precision_utils import mp_policy, ComplexMP, kron_product_2x2_complex_mp
from dsa2000_common.common.one_factor import get_one_factors
from dsa2000_common.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_common.common.sum_utils import kahan_sum
from dsa2000_common.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine, \
    build_far_field_delay_engine
from dsa2000_common.delay_models.base_near_field_delay_engine import build_near_field_delay_engine
from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model
from dsa2000_common.visibility_model.source_models.celestial.base_point_source_model import BasePointSourceModel


def compute_image_to_image_full_stokes(A_in, lmn_in, lmn_out, freqs, uvw, antenna1, antenna2, gains: ComplexMP):
    """
    Compute image at output locations in full stokes.

    Args:
        A_in: [D, 2, 2]
        lmn_in: [D, 3]
        lmn_out: [E, 3]
        freqs: [C]
        uvw: [B, 3]
        antenna1: [B]
        antenna2: [B]
        gains: [D, A, C, 2, 2]

    Returns:
        [E, 2, 2]
    """
    gains = gains.swapaxes(1, 2)  # [D, C, A, 2, 2]
    g1 = gains.take(antenna1, axis=2, unique_indices=True, indices_are_sorted=True)  # sorted and unique
    g2 = gains.take(antenna2, axis=2, unique_indices=True, indices_are_sorted=False)  # unique (not sorted)

    dtype = gains.dtype

    def outer_body_fn(accumulate, x):
        (A_in, lmn_in, g1, g2) = x  # [2, 2], [3], [C, B, 2, 2], [C, B, 2, 2]

        def inner_body_fn(accumulate, x):
            (freq, g1, g2) = x  # [], [B, 2, 2], [B, 2, 2]
            c = 299792458.
            wavelength = (c / freq).astype(dtype)
            _uvw = uvw / wavelength  # [B, 3]
            phase = -2 * jnp.pi * (jnp.sum(_uvw * lmn_in, axis=-1) - _uvw[:, 2])  # [B]
            phase = phase.astype(dtype)
            fringe = ComplexMP(jnp.cos(phase), jnp.sin(phase)) / lmn_in[2]
            vis = fringe[:, None, None] * kron_product_2x2_complex_mp(
                g1, A_in, g2.conj().swapaxes(-1, -2)
            )  # [B, 2, 2]

            out_phase = 2 * jnp.pi * (jnp.sum(_uvw * lmn_out[:, None, :], axis=-1) - _uvw[:, 2])  # [E, B]
            out_phase = out_phase.astype(dtype)
            out_fringe = ComplexMP(jnp.cos(out_phase), jnp.sin(out_phase)) * lmn_out[:, None, 2]
            out_image = jnp.sum((out_fringe[:, :, None, None] * vis).real, axis=1)  # [E, 2, 2]
            out_image /= jnp.asarray(uvw.shape[0], out_image.dtype)
            accumulate += out_image
            return accumulate, None

        inner_accumulate = jnp.zeros_like(accumulate)
        out_image, _ = jax.lax.scan(inner_body_fn, inner_accumulate, (freqs, g1, g2))
        out_image /= jnp.asarray(len(freqs), out_image.dtype)
        return accumulate + out_image, None

    outer_accumulate = jnp.zeros(lmn_out.shape[:1] + (2, 2), dtype=gains.dtype)
    out_image, _ = jax.lax.scan(outer_body_fn, outer_accumulate, (A_in, lmn_in, g1, g2))
    return out_image


def compute_image_to_vis_full_stokes(A_in, lmn_in, freqs, uvw, antenna1, antenna2, gains: ComplexMP):
    """
    Compute image at output locations in full stokes.

    Args:
        A_in: [D, 2, 2]
        lmn_in: [D, 3]
        freqs: [C]
        uvw: [B, 3]
        antenna1: [B]
        antenna2: [B]
        gains: [D, A, C, 2, 2]

    Returns:
        [E, 2, 2]
    """
    gains = gains.swapaxes(1, 2)  # [D, C, A, 2, 2]
    g1 = gains.take(antenna1, axis=2, unique_indices=True, indices_are_sorted=True)  # sorted and unique
    g2 = gains.take(antenna2, axis=2, unique_indices=True, indices_are_sorted=False)  # unique (not sorted)

    B = np.shape(antenna1)[0]
    C = np.shape(freqs)[0]
    dtype = gains.dtype
    A_scale = jnp.max(A_in, axis=[-1, -2], keepdims=True)
    A_in_scaled = (A_in / A_scale).astype(dtype)
    freqs = freqs.astype(jnp.float64)
    lmn_in = lmn_in.astype(jnp.float64)
    uvw = uvw.astype(jnp.float64)

    def wrap(angle):
        return (angle + 2 * np.pi) % (2 * np.pi)

    def outer_accum_fn(x):
        (A_in, A_scale, lmn_in, g1, g2) = x  # [2, 2], [3], [C, B, 2, 2], [C, B, 2, 2]

        def inner_body_fn(carry, x):
            (freq, g1, g2) = x  # [], [B, 2, 2], [B, 2, 2]
            c = 299792458.
            wavelength = c / freq
            _uvw = uvw / wavelength  # [B, 3]
            phase = -2 * jnp.pi * (jnp.sum(_uvw * lmn_in, axis=-1) - _uvw[:, 2])  # [B]
            phase = wrap(phase)  # Wrapping at 64bit preserves precision
            phase = phase.astype(dtype)
            scale = jnp.reciprocal(lmn_in[2]).astype(dtype)
            fringe = ComplexMP(scale * jnp.cos(phase), scale * jnp.sin(phase), dtype)
            vis = fringe[:, None, None] * kron_product_2x2_complex_mp(
                g1, A_in, g2.conj().swapaxes(-1, -2)
            )  # [B, 2, 2]
            return None, vis.astype(dtype)

        _, out_vis = jax.lax.scan(inner_body_fn, None, (freqs, g1, g2))
        out_vis = A_scale * out_vis.swapaxes(0, 1)  # [B, C, 2, 2]
        return out_vis.astype(dtype)

    init_accumulate = ComplexMP(jnp.zeros((B, C, 2, 2), dtype=dtype), jnp.zeros((B, C, 2, 2), dtype=dtype), dtype)
    out_vis, _ = kahan_sum(outer_accum_fn, init_accumulate, (A_in_scaled, A_scale, lmn_in, g1, g2))
    return out_vis


def single_factor_compute_image_to_image_full_stokes(antenna1, antenna2, A_in, lmn_in, lmn_out, times, freqs,
                                                     gains: ComplexMP, far_field_delay_engine: BaseFarFieldDelayEngine):
    def accum_fn(time):
        uvw = far_field_delay_engine.compute_uvw(jnp.repeat(time[None], len(antenna1)), antenna1, antenna2)  # [T, B, 3]
        _image = compute_image_to_image_full_stokes(
            A_in=A_in,
            lmn_in=lmn_in,
            lmn_out=lmn_out,
            freqs=freqs,
            uvw=uvw,
            antenna1=antenna1,
            antenna2=antenna2,
            gains=gains,
        )
        return _image

    init_accumulate = jnp.zeros(lmn_out.shape[:1] + (2, 2), dtype=gains.dtype)
    out_image, _ = kahan_sum(accum_fn, init_accumulate, times)
    out_image /= jnp.asarray(len(times), out_image.dtype)
    return out_image


@jax.jit
def reduce_factor_compute_image_to_image_full_stokes(factors, A_in, lmn_in, lmn_out, times, freqs,
                                                     gains: ComplexMP, far_field_delay_engine: BaseFarFieldDelayEngine):
    def accum_fn(factor):
        antenna1, antenna2 = factor.T
        _image = single_factor_compute_image_to_image_full_stokes(
            antenna1=antenna1,
            antenna2=antenna2,
            A_in=A_in,
            lmn_in=lmn_in,
            lmn_out=lmn_out,
            times=times,
            freqs=freqs,
            gains=gains,
            far_field_delay_engine=far_field_delay_engine,
        )
        return _image

    init_accumulate = jnp.zeros(lmn_out.shape[:1] + (2, 2), dtype=gains.dtype)
    out_image, _ = kahan_sum(accum_fn, init_accumulate, factors)
    return out_image


def build_mock_obs_setup(array_name: str, num_sol_ints_time: int, frac_aperture: float = 1.):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))

    array_location = array.get_array_location()

    ref_time = at.Time('2021-01-01T00:00:00', scale='utc')
    num_times_per_sol_int = 1
    num_times = num_times_per_sol_int * num_sol_ints_time
    obstimes = ref_time + np.arange(num_times) * array.get_integration_time()

    phase_center = ENU(0, 0, 1, location=array_location, obstime=ref_time).transform_to(ac.ICRS())
    freqs = array.get_channels()[:1]

    # Point dishes exactly at phase center
    pointing = phase_center

    antennas = array.get_antennas()
    if frac_aperture < 1.:
        keep_ant_idxs = np.random.choice(len(antennas), max(2, int(frac_aperture * len(antennas))), replace=False)
        antennas = antennas[keep_ant_idxs]

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

    system_equivalent_flux_density, chan_width, integration_time = array.get_system_equivalent_flux_density(), array.get_channel_width(), array.get_integration_time()

    chan_width *= 40  # simulate wider band to lower nosie
    integration_time *= 4  # simulate longer integration time

    return ref_time, obstimes, freqs, phase_center, antennas, geodesic_model, far_field_delay_engine, near_field_delay_engine, system_equivalent_flux_density, chan_width, integration_time


def test_compute_image_to_image_full_stokes():
    D = 1
    E = 1

    A_in = jnp.asarray(np.random.uniform(0., 0.001, (D, 2, 2), jnp.float16)).at[:, 0, 1].set(0.).at[:, 1, 0].set(0.)

    ref_time, obstimes, obsfreqs, phase_center, antennas, geodesic_model, far_field_delay_engine, near_field_delay_engine, system_equivalent_flux_density, chan_width, integration_time = build_mock_obs_setup(
        'dsa2000_optimal_v1', num_sol_ints_time=1, frac_aperture=0.1
    )

    directions_in = create_spherical_spiral_grid(
        pointing=phase_center,
        num_points=D,
        angular_radius=1 * au.deg
    )
    lmn_in = quantity_to_jnp(icrs_to_lmn(directions_in, phase_center), 'rad')

    directions_out = create_spherical_spiral_grid(
        pointing=phase_center,
        num_points=E,
        angular_radius=1 * au.deg
    )
    lmn_out = quantity_to_jnp(icrs_to_lmn(directions_out, phase_center), 'rad')

    times = time_to_jnp(obstimes, ref_time)
    freqs = quantity_to_jnp(obsfreqs, 'Hz')

    A = len(antennas)
    C = len(obsfreqs)
    gains = ComplexMP(jnp.ones((D, A, C, 2, 2)).at[..., 0, 1].set(0.).at[..., 1, 0].set(0.),
                      np.random.normal(size=(D, A, C, 2, 2)))

    factors = np.asarray(get_one_factors(A))

    # A_out = reduce_factor_compute_image_to_image_full_stokes(
    #     factors,
    #     A_in, lmn_in, lmn_out, times, freqs,
    #     gains, far_field_delay_engine
    # )
    # print(A_out)

    source_model = BasePointSourceModel(
        model_freqs=freqs,
        ra=directions_in.ra.rad,
        dec=directions_in.dec.rad,
        A=jnp.repeat(A_in[:, None, :, :], len(freqs), axis=1)
    )

    visibility_coords = far_field_delay_engine.compute_visibility_coords(freqs, times, with_autocorr=False)

    vis_out = compute_image_to_vis_full_stokes(
        A_in=A_in,
        lmn_in=lmn_in,
        freqs=freqs,
        uvw=visibility_coords.uvw[0, :, :],
        antenna1=visibility_coords.antenna1,
        antenna2=visibility_coords.antenna2,
        gains=gains
    )

    print(vis_out)

    # vis = source_model.predict(visibility_coords, None, near_field_delay_engine, far_field_delay_engine, geodesic_model)
    # # print(vis)
    # vis = vis.reshape((-1, C, 2, 2))
    # # image at point
    # image_out = explicit_gridder(uvw=visibility_coords.uvw.reshape((-1, 3)), freqs=freqs, vis=vis, lmn=lmn_out)
    # print(image_out)


@pytest.mark.parametrize('dtype,atol', [(jnp.float16, 1e-2), (jnp.float32, 1e-5), (jnp.float64, 1e-6)])
def test_compute_image_to_vis_full_stokes(dtype, atol):
    np.random.seed(0)
    D = 10

    A_in = jnp.array(np.random.uniform(0., 1, (D, 2, 2)), np.float64).at[:, 0, 1].set(0.).at[:, 1, 0].set(0.)

    (ref_time, obstimes, obsfreqs, phase_center, antennas, geodesic_model, far_field_delay_engine,
     near_field_delay_engine, system_equivalent_flux_density, chan_width, integration_time) = build_mock_obs_setup(
        'dsa2000_optimal_v1', num_sol_ints_time=1, frac_aperture=0.1
    )

    directions_in = create_spherical_spiral_grid(
        pointing=phase_center,
        num_points=D,
        angular_radius=1 * au.deg
    )
    lmn_in = quantity_to_jnp(icrs_to_lmn(directions_in, phase_center), 'rad')

    times = time_to_jnp(obstimes, ref_time)
    freqs = quantity_to_jnp(obsfreqs, 'Hz')

    A = len(antennas)
    C = len(obsfreqs)
    gains = ComplexMP(
        jnp.ones((D, A, C, 2, 2)).at[..., 0, 1].set(0.).at[..., 1, 0].set(0.),
        jnp.zeros((D, A, C, 2, 2)).at[..., 0, 1].set(0.).at[..., 1, 0].set(0.), dtype
    )

    source_model = BasePointSourceModel(
        model_freqs=freqs,
        ra=directions_in.ra.rad,
        dec=directions_in.dec.rad,
        A=jnp.repeat(A_in[:, None, :, :], len(freqs), axis=1)
    )

    visibility_coords = far_field_delay_engine.compute_visibility_coords(freqs, times[:1], with_autocorr=False)

    vis_out = compute_image_to_vis_full_stokes(
        A_in=A_in,
        lmn_in=lmn_in,
        freqs=freqs,
        uvw=visibility_coords.uvw[0, :, :],
        antenna1=visibility_coords.antenna1,
        antenna2=visibility_coords.antenna2,
        gains=gains
    )

    vis = source_model.predict(
        visibility_coords, None, near_field_delay_engine, far_field_delay_engine, geodesic_model)
    vis = vis.reshape((-1, C, 2, 2))
    np.testing.assert_allclose(vis.real, vis_out.real, atol=atol)
    np.testing.assert_allclose(vis.imag, vis_out.imag, atol=atol)


@pytest.mark.parametrize('dtype,atol', [(jnp.float16, 1e-2), (jnp.float32, 1e-5)])
@pytest.mark.parametrize('D', [1, 10, 100])
def test_compute_image_to_vis_full_stokes_self_with_gains(D, dtype, atol):
    np.random.seed(0)

    A_in = jnp.array(np.random.uniform(0., 1, (D, 2, 2)), np.float64).at[:, 0, 1].set(0.).at[:, 1, 0].set(0.)

    (ref_time, obstimes, obsfreqs, phase_center, antennas, geodesic_model, far_field_delay_engine,
     near_field_delay_engine, system_equivalent_flux_density, chan_width, integration_time) = build_mock_obs_setup(
        'dsa2000_optimal_v1', num_sol_ints_time=1, frac_aperture=0.1
    )

    directions_in = create_spherical_spiral_grid(
        pointing=phase_center,
        num_points=D,
        angular_radius=1 * au.deg
    )
    lmn_in = quantity_to_jnp(icrs_to_lmn(directions_in, phase_center), 'rad')

    times = time_to_jnp(obstimes, ref_time)
    freqs = quantity_to_jnp(obsfreqs, 'Hz')
    visibility_coords = far_field_delay_engine.compute_visibility_coords(freqs, times[:1], with_autocorr=False)

    A = len(antennas)
    C = len(obsfreqs)
    gains = ComplexMP(
        jnp.ones((D, A, C, 2, 2)).at[..., 0, 1].set(0.).at[..., 1, 0].set(0.),
        jnp.asarray(np.random.normal(size=(D, A, C, 2, 2))).at[..., 0, 1].set(0.).at[..., 1, 0].set(0.),
        jnp.float64
    )

    vis_out64 = compute_image_to_vis_full_stokes(
        A_in=A_in,
        lmn_in=lmn_in,
        freqs=freqs,
        uvw=visibility_coords.uvw[0, :, :],
        antenna1=visibility_coords.antenna1,
        antenna2=visibility_coords.antenna2,
        gains=gains.astype(jnp.float64)
    )

    vis_out = compute_image_to_vis_full_stokes(
        A_in=A_in,
        lmn_in=lmn_in,
        freqs=freqs,
        uvw=visibility_coords.uvw[0, :, :],
        antenna1=visibility_coords.antenna1,
        antenna2=visibility_coords.antenna2,
        gains=gains.astype(dtype)
    )

    np.testing.assert_allclose(vis_out.real, vis_out64.real, atol=atol)
    np.testing.assert_allclose(vis_out.imag, vis_out64.imag, atol=atol)


def explicit_gridder(uvw, freqs, vis, lmn):
    c = 299792458.  # m/s
    l, m, n = lmn[..., 0], lmn[..., 1], lmn[..., 2]
    dirty = np.zeros(np.shape(lmn)[:-1] + (2, 2), mp_policy.image_dtype)
    for row, (u, v, w) in enumerate(uvw):
        for col, freq in enumerate(freqs):
            wavelength = c / freq
            phase = 2j * np.pi * (u * l + v * m + w * (n - 1)) / wavelength
            dirty += (vis[row, col] * np.exp(phase[..., None, None])).real
    dirty /= np.size(vis)
    return dirty
