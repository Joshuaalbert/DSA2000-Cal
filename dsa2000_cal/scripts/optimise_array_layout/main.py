import itertools
import os
from functools import partial
from typing import Tuple, Optional

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import pylab as plt
from jaxns import Prior, Model, save_pytree
from jaxns.framework.special_priors import SpecialPrior
from scipy.spatial import KDTree
from tomographic_kernel.frames import ENU

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardt
from dsa2000_cal.common.astropy_utils import mean_itrs
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.assets.array_constraints.array_constraint_content import ArrayConstraint

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

from dsa2000_cal.common.array_types import FloatArray

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from dsa2000_cal.common.mixed_precision_utils import mp_policy

tfpd = tfp.distributions


# Compose PSF
def rotation_matrix_change_dec(delta_dec: FloatArray):
    # Rotate up or down changing DEC, but keeping RA constant.
    # Used for projecting ENU system
    c, s = jnp.cos(delta_dec), jnp.sin(delta_dec)
    R = jnp.asarray(
        [
            [1., 0., 0.],
            [0., c, -s],
            [0., s, c]
        ]
    )
    return R


def rotate_coords(antennas: FloatArray, dec_from: FloatArray, dec_to: FloatArray) -> FloatArray:
    # East to east
    delta_dec = dec_to - dec_from
    east, north, up = antennas[..., 0], antennas[..., 1], antennas[..., 2]
    east_prime = east
    north_prime = jnp.cos(delta_dec) * north - jnp.sin(delta_dec) * up
    up_prime = jnp.sin(delta_dec) * north + jnp.cos(delta_dec) * up
    return jnp.stack([east_prime, north_prime, up_prime], axis=-1)


def deproject_antennas(antennas_projected: FloatArray, latitude: FloatArray, transit_dec: FloatArray) -> FloatArray:
    antennas = rotate_coords(antennas_projected, transit_dec, latitude)
    # antennas = antennas.at[..., 2].set(0.)
    return antennas


def project_antennas(antennas: FloatArray, latitude: FloatArray, transit_dec: FloatArray) -> FloatArray:
    antennas_projected = rotate_coords(antennas, latitude, transit_dec)
    # antennas_projected = antennas_projected.at[..., 2].set(0.)
    return antennas_projected


def compute_psf(antennas: FloatArray, lmn: FloatArray, freq: FloatArray, latitude: FloatArray,
                transit_dec: FloatArray, with_autocorr: bool = False) -> FloatArray:
    """
    Compute the point spread function of the array. Uses short cut,

    B(l,m) = (sum_i e^(-i2pi (u_i l + v_i m)))^2/N^2

    To remove auto-correlations, there are N values of 1 to subtract from N^2 values, then divide by (N-1)N
    PSF(l,m) = (N^2 B(l,m) - N)/(N-1)/N = (N B(l,m) - 1)/(N-1) where B(l,m) in [0, 1].
    Thus the amount of negative is (-1/(N-1))

    Args:
        antennas: [N, 3]
        lmn: [..., 3]
        freq: []

    Returns:
        psf: [...]
    """

    # # Create a mesh for the shard_map
    # mesh = create_mesh((len(jax.devices()),), ('shard',), devices=jax.devices())
    #
    # @partial(
    #     shard_map,
    #     mesh=mesh,
    #     in_specs=(PartitionSpec(),
    #               PartitionSpec('shard', ),
    #               PartitionSpec(),
    #               PartitionSpec(),
    #               PartitionSpec(),
    #               ),
    #     out_specs=PartitionSpec('shard', )
    # )
    def compute_shard_psf(antennas, lmn_shard, freq, latitude, transit_dec):
        antennas = project_antennas(antennas, latitude, transit_dec)
        wavelength = mp_policy.cast_to_length(299792458. / freq)
        r = antennas / wavelength
        delay = -2 * jnp.pi * jnp.sum(r * lmn_shard[..., None, :], axis=-1)  # [..., N]
        N = antennas.shape[-2]
        voltage_beam = jax.lax.complex(jnp.cos(delay), jnp.sin(delay))  # [..., N]
        voltage_beam = jnp.mean(voltage_beam, axis=-1)  # [...]
        power_beam = jnp.abs(voltage_beam) ** 2
        if with_autocorr:
            return power_beam
        return jnp.reciprocal(N - 1) * (N * power_beam - 1)

    _, psf = jax.lax.scan(
        lambda carry, lmn: (None, compute_shard_psf(antennas, lmn, freq, latitude, transit_dec)),
        None,
        lmn
    )

    return psf


def compute_ideal_psf_distribution(key, lmn: FloatArray, freq: FloatArray, latitude: FloatArray,
                                   transit_dec: FloatArray, base_projected_array: FloatArray, num_samples: int):
    def body_fn(carry, key):
        x, x2 = carry
        psf = sample_ideal_psf(
            key,
            lmn,
            freq,
            latitude,
            transit_dec,
            base_projected_array,
            with_autocorr=True
        )
        log_psf = 10 * jnp.log10(psf)
        x = x + log_psf
        x2 = x2 + log_psf ** 2
        return (x, x2), None

    init_x = jnp.zeros(lmn.shape[:-1])
    (x, x2), _ = jax.lax.scan(
        body_fn,
        (init_x, init_x),
        jax.random.split(key, num_samples)
    )
    mean = x / num_samples
    std = jnp.sqrt(jnp.abs(x2 / num_samples - mean ** 2))
    return mean, std


def sample_ideal_psf(key, lmn: FloatArray, freq: FloatArray, latitude: FloatArray,
                     transit_dec: FloatArray, base_projected_array: FloatArray, with_autocorr: bool) -> FloatArray:
    """
    Compute the ideal point spread function of the array

    Args:
        lmn: [Nr, Ntheta, 3]
        freq: []
        latitude: []
        transit_dec: []

    Returns:
        psf: [Nr, Ntheta]
    """
    antenna_projected_dist = tfpd.Normal(loc=0, scale=50.)

    antennas_enu = base_projected_array + antenna_projected_dist.sample(base_projected_array.shape, key)
    psf = compute_psf(antennas_enu, lmn, freq, latitude, transit_dec, with_autocorr=with_autocorr)
    return psf


def sample_aoi(num_samples, array_location: ac.EarthLocation, additional_distance):
    radius = np.linalg.norm(array_location.get_itrs().cartesian.xyz.to(au.m).value)
    array_constraint = ArrayConstraint()
    samples = []
    aoi_data = array_constraint.get_area_of_interest_regions()
    constraint_data = array_constraint.get_constraint_regions()
    aoi_samplers, aoi_buffers = zip(*aoi_data)
    constraint_samplers, constraint_buffers = zip(*constraint_data)
    areas = np.asarray([s.total_area for s in aoi_samplers])
    aoi_probs = areas / areas.sum()
    c = 0
    while len(samples) < num_samples:
        sampler_idx = np.random.choice(len(aoi_samplers), p=aoi_probs)
        sampler = aoi_samplers[sampler_idx]
        buffer = aoi_buffers[sampler_idx]
        sample_proposal = sampler.get_samples_within(1)[0]
        # Check that it far enough from AOI perimeter
        _, angular_dist = sampler.closest_approach_to_boundary(*sample_proposal)
        dist = np.pi / 180. * angular_dist * radius
        if dist < buffer + additional_distance:
            continue

        # Check that it is far enough from constraint regions
        far_enough_away = True
        for constraint_sampler, buffer in zip(constraint_samplers, constraint_buffers):
            _, angular_dist = constraint_sampler.closest_approach(*sample_proposal)
            dist = np.pi / 180. * angular_dist * radius
            if dist <= buffer + additional_distance:
                far_enough_away = False
                break

        if far_enough_away:
            samples.append(sample_proposal)
        c += 1
    samples = np.asarray(samples)
    new_locations = ac.EarthLocation.from_geodetic(
        lon=samples[:, 0] * au.deg,
        lat=samples[:, 1] * au.deg,
        height=array_location.geodetic.height
    )  # [num_relocate]
    return new_locations


def relocate_antennas(antennas: ac.EarthLocation, obstime: at.Time, array_location: ac.EarthLocation,
                      additional_buffer, force_relocate) -> ac.EarthLocation:
    # Find closest constraint point
    closest_point_dist, closest_point_dist_including_buffer, closest_type = get_closest_point_dist(
        locations=antennas, array_location=array_location, additional_distance=additional_buffer
    )  # [N] in meters
    too_close_to_constraint = closest_point_dist_including_buffer <= 0.
    too_close = np.logical_or(too_close_to_constraint, force_relocate)
    if not np.any(too_close):
        return antennas
    for idx in range(len(too_close)):
        if force_relocate[idx]:
            closest_type[idx] = "another antenna"
            closest_point_dist_including_buffer[idx] = 0.
            closest_point_dist[idx] = 0.

    num_relocate = np.sum(too_close)
    with open('relocation.log', 'w') as f:
        s = f"Relocating {num_relocate} antennas"
        print(s)
        f.write(f"{s}\n")
        for idx in range(len(too_close)):
            if too_close[idx]:
                s = (f"Antenna {idx} (lon={antennas[idx].lon},lat={antennas[idx].lat},height={antennas[idx].height}) "
                     f"is too close {closest_point_dist[idx]:.1f}m "
                     f"({-closest_point_dist_including_buffer[idx]:.1f}m within buffer) to {closest_type[idx]}")
                # print(s)
                f.write(f"{s}\n")

    # sample new locations
    new_locations = sample_aoi(
        num_relocate, array_location, additional_buffer
    )  # [num_relocate]
    new_enu = new_locations.get_itrs(
        obstime=obstime, location=array_location
    ).transform_to(
        ENU(obstime=obstime, location=array_location)
    ).cartesian.xyz.to('m').value.T  # [num_relocate, 3]
    antennas_enu = antennas.get_itrs(
        obstime=obstime, location=array_location
    ).transform_to(
        ENU(obstime=obstime, location=array_location)
    ).cartesian.xyz.to('m').value.T  # [N, 3]
    # Set new locations
    antennas_enu[too_close, :] = new_enu
    # Back to earth location
    antennas = ENU(
        antennas_enu[:, 0] * au.m,
        antennas_enu[:, 1] * au.m,
        antennas_enu[:, 2] * au.m,
        obstime=obstime,
        location=array_location
    ).transform_to(ac.ITRS(obstime=obstime, location=array_location)).earth_location
    return antennas


def get_closest_point_dist(locations: ac.EarthLocation, array_location: ac.EarthLocation, additional_distance):
    radius = np.linalg.norm(array_location.get_itrs().cartesian.xyz.to(au.m).value)
    array_constraint = ArrayConstraint()
    aoi_data = array_constraint.get_area_of_interest_regions()
    constraint_data = array_constraint.get_constraint_regions()

    aoi_samplers, aoi_buffers = zip(*aoi_data)
    constraint_samplers, constraint_buffers = zip(*constraint_data)
    points_lon = locations.lon.to(au.deg).value
    points_lat = locations.lat.to(au.deg).value
    closest_type = []
    closest_point_dist = []
    closest_point_dist_including_buffer = []
    for point in zip(points_lon, points_lat):
        dist = np.inf
        dist_including_buffer = np.inf
        _type = None
        for aoi_sampler, buffer in zip(aoi_samplers, aoi_buffers):
            _, angular_dist = aoi_sampler.closest_approach_to_boundary(*point)
            _dist = np.pi / 180. * angular_dist * radius
            _dist_including_buffer = _dist - buffer - additional_distance

            if _dist < dist:
                dist = _dist
                dist_including_buffer = _dist_including_buffer
                _type = aoi_sampler.name

            if not aoi_sampler.contains(*point):
                dist = 0.
                dist_including_buffer = _dist - buffer - additional_distance
                _type = "Outside AOI"

        for constraint_sampler, buffer in zip(constraint_samplers, constraint_buffers):
            _, angular_dist = constraint_sampler.closest_approach(*point)
            _dist = np.pi / 180. * angular_dist * radius
            _dist_including_buffer = _dist - buffer - additional_distance
            if _dist < dist:
                dist = _dist
                dist_including_buffer = _dist_including_buffer
                _type = constraint_sampler.name

        closest_point_dist.append(dist)
        closest_point_dist_including_buffer.append(dist_including_buffer)
        closest_type.append(_type)

    return np.asarray(closest_point_dist), np.asarray(closest_point_dist_including_buffer), closest_type


def get_uniform_ball_prior(antennas_enu: np.ndarray, obstime: at.Time, array_location: ac.EarthLocation):
    """
    Construct a prior for the antenna locations

    Args:
        antennas_enu: [N, 3]
        obstime: at.Time
        array_location: ac.EarthLocation

    Returns:
        ball_centre: [N, 2]
        ball_radius: [N]
    """
    # convert to earth location
    antennas_locations = ENU(
        antennas_enu[:, 0] * au.m,
        antennas_enu[:, 1] * au.m,
        antennas_enu[:, 2] * au.m,
        obstime=obstime,
        location=array_location
    ).transform_to(ac.ITRS(obstime=obstime, location=array_location)).earth_location
    # Find closest constraint point
    _, ball_radius, _ = get_closest_point_dist(
        locations=antennas_locations,
        array_location=array_location,
        additional_distance=0.  # If more than specified buffer desired, increase this
    )  # [N] in meters
    # Now make sure antennas are at least 5m apart
    tree = KDTree(
        antennas_enu
    )
    dist, _ = tree.query(antennas_enu, k=2)
    dist = dist[:, 1]
    min_sep = 8.
    ball_radius = np.maximum(0., np.minimum(ball_radius, dist - min_sep))
    # Construct prior
    ball_centre = antennas_enu
    return ball_centre, ball_radius


class BiUnitRadiusPrior(SpecialPrior):
    def __init__(self, *, max_radius: FloatArray, name: Optional[str] = None):
        super(BiUnitRadiusPrior, self).__init__(name=name)
        self.max_radius = max_radius

    def _dtype(self):
        return np.result_type(self.max_radius)

    def _base_shape(self) -> Tuple[int, ...]:
        return np.shape(self.max_radius)

    def _shape(self) -> Tuple[int, ...]:
        return jnp.shape(self.max_radius)

    def _forward(self, U) -> FloatArray:
        return self._quantile(U)

    def _inverse(self, X) -> FloatArray:
        return self._cdf(X)

    def _log_prob(self, X) -> FloatArray:
        return jnp.abs(X / self.max_radius)

    def _quantile(self, U):
        twoUm1 = U + U - 1
        return twoUm1 * jnp.sqrt(jnp.abs(twoUm1)) * self.max_radius

    def _cdf(self, Y):
        Y = Y / self.max_radius
        return 0.5 * jnp.sign(Y) * jnp.square(Y) + 0.5


@partial(jax.jit)
def solve(ball_origin, ball_radius, lmn, latitude, freqs, decs,
          target_log_psf_mean, target_log_psf_stddev):
    def prior_model():
        # Uniform ball prior
        theta = yield Prior(tfpd.Uniform(
            jnp.zeros_like(ball_radius),
            2. * jnp.pi * jnp.ones_like(ball_radius)),
            name='theta').parametrised(random_init=True)
        direction = jnp.stack([jnp.cos(theta), jnp.sin(theta), jnp.zeros_like(theta)], axis=-1)

        # Allow going up and down, with initial point at zero
        radius = yield Prior(tfpd.Uniform(-ball_radius, ball_radius), name='radius').parametrised()
        # radius = yield BiUnitRadiusPrior(max_radius=ball_radius, name='radius').parametrised()
        antennas_enu = ball_origin + radius[:, None] * direction  # [N, 3]
        return antennas_enu

    def log_likelihood(x):
        return 0.

    model = Model(prior_model, log_likelihood)

    U = model.sample_U(jax.random.key(0))

    def residuals(params):
        (x,) = model(params).prepare_input(U)

        psf = jax.vmap(
            lambda freq, dec: compute_psf(x, lmn, freq, latitude, dec, with_autocorr=True)
        )(freqs, decs)

        log_psf = 10 * jnp.log10(psf)

        return (log_psf - target_log_psf_mean) / target_log_psf_stddev

    # solver = ApproxCGNewton(
    #     obj_fn=objective,
    #     num_approx_steps=2,
    #     num_iterations=10,
    #     verbose=True,
    #     min_cg_maxiter=100,
    #     init_cg_maxiter=100
    # )

    solver = MultiStepLevenbergMarquardt(
        residual_fn=residuals,
        num_iterations=10,
        num_approx_steps=0,
        p_any_improvement=0.01,
        p_less_newton=0.25,
        p_more_newton=0.8,
        c_more_newton=0.2,
        c_less_newton=1.5,
        verbose=True
    )
    state = solver.create_initial_state(model.params)
    state, diagnostics = solver.solve(state)
    return model(state.x).prepare_input(U)[0], state, diagnostics


def plot_relocated_antennas(antennas_before: ac.EarthLocation, antennas_after: ac.EarthLocation, obstime: at.Time,
                            array_location: ac.EarthLocation):
    # bring both to ENU then antennas before with arrows to antennas after, colored by distance moved
    antennas_before_enus = antennas_before.get_itrs(
        obstime=obstime, location=array_location
    ).transform_to(
        ENU(obstime=obstime, location=array_location)
    ).cartesian.xyz.to('m').value.T  # [N, 3]
    antennas_after_enus = antennas_after.get_itrs(
        obstime=obstime, location=array_location
    ).transform_to(
        ENU(obstime=obstime, location=array_location)
    ).cartesian.xyz.to('m').value.T  # [N, 3]
    dists = np.linalg.norm(antennas_after_enus - antennas_before_enus, axis=-1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(antennas_before_enus[:, 0], antennas_before_enus[:, 1], c=dists, cmap='jet')
    ar = ax.quiver(
        antennas_before_enus[:, 0], antennas_before_enus[:, 1],
        antennas_after_enus[:, 0] - antennas_before_enus[:, 0],
        antennas_after_enus[:, 1] - antennas_before_enus[:, 1],
        dists, scale=1, scale_units='xy', cmap='jet'
    )
    plt.colorbar(ar, label='Distance moved (m)')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title('Antenna relocation')
    fig.savefig('antenna_relocation.png')
    plt.close('all')


def create_lmn1():
    lmn = []
    for inner, outer, dl, frac in [
        (0. * au.arcmin, 1. * au.arcmin, (3.3 / 7) * au.arcsec, 1.),
        # (1. * au.arcmin, 0.5 * au.deg, (3.3/7) * au.arcsec, 0.001),
        # (0.5 * au.deg, 1.5 * au.deg, (3.3/7) * au.arcsec, 0.0001),
    ]:
        lvec = mvec = np.arange(-outer.to('rad').value, outer.to('rad').value, dl.to('rad').value)
        L, M = np.meshgrid(lvec, mvec, indexing='ij')
        L = L.flatten()
        M = M.flatten()
        LM = L ** 2 + M ** 2
        _lmn = np.stack([L, M, 1 - jnp.sqrt(1 - LM)], axis=-1)
        keep = np.logical_and(np.sqrt(LM) >= inner.to('rad').value, LM < outer.to('rad').value)
        _lmn = _lmn[keep]
        print(f"Got {_lmn.shape[0]} samples")
        if frac < 1:
            select_idx = np.random.choice(_lmn.shape[0], int(frac * _lmn.shape[0]), replace=False)
            _lmn = _lmn[select_idx]
        print(f"Got {_lmn.shape[0]} samples from {inner} to {outer} with {dl} spacing")
        lmn.append(_lmn)
    lmn = jnp.concatenate(lmn, axis=0)
    print(f"Total {lmn.shape[0]} samples")
    return lmn


def create_lmn2():
    batch_size: int = 1024
    num_radial_bins: int = 3600 // 8
    num_theta_bins: int = 10
    lmax: au.Quantity = 1 * au.deg
    lmin: au.Quantity = 8 * au.arcsec
    fwhm: au.Quantity = 3.3 * au.arcsec
    lmax = quantity_to_jnp(lmax, 'rad')
    lmin = quantity_to_jnp(lmin, 'rad')
    fwhm = quantity_to_jnp(fwhm, 'rad')
    radii = jnp.concatenate([0.5 * fwhm[None], jnp.linspace(lmin, lmax, num_radial_bins)])
    theta = jnp.linspace(0., 2 * np.pi, num_theta_bins, endpoint=False)

    R, Theta = jnp.meshgrid(
        radii, theta,
        indexing='ij'
    )

    L = R * jnp.cos(Theta)
    M = R * jnp.sin(Theta)
    N = jnp.sqrt(1 - L ** 2 - M ** 2)
    lmn = jnp.stack([L, M, 1 - N], axis=-1)  # [Nr, Nt, 3]

    return lmn


def create_target(key, lmn, freqs, decs, num_samples: int):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))

    antennas = array.get_antennas()
    array_location = array.get_array_location()
    obstime = at.Time('2022-01-01T00:00:00', scale='utc')
    antennas_enu = antennas.get_itrs(obstime=obstime, location=array_location).transform_to(
        ENU(0, 0, 1, obstime=obstime, location=array_location)
    )
    antennas_enu_xyz = antennas_enu.cartesian.xyz.T
    latitude = array_location.geodetic.lat.rad
    antennas_enu_xyz[:, 1] /= np.cos(latitude)

    antennas_enu_xyz = jnp.asarray(antennas_enu_xyz)

    return jax.vmap(lambda freq, dec: compute_ideal_psf_distribution(
        key, lmn, freq, latitude, dec, antennas_enu_xyz, num_samples)
                    )(freqs, decs)


def create_initial_data(antennas: ac.EarthLocation, obstime: at.Time, array_location: ac.EarthLocation):
    antenna_enu_xyz = antennas.get_itrs(
        obstime=obstime, location=array_location).transform_to(
        ENU(obstime=obstime, location=array_location)).cartesian.xyz.to('m').T
    antennas0 = quantity_to_jnp(antenna_enu_xyz, 'm')  # [N, 3]
    latitude = quantity_to_jnp(array_location.geodetic.lat)
    return antennas0, latitude


def main(init_config: str | None = None):
    key = jax.random.PRNGKey(0)
    np.random.seed(0)
    lmn = create_lmn1()

    freqs = [700, 1350, 2000] * au.MHz
    decs = [-30, 0, 30, 60, 90] * au.deg

    freqs, decs = np.meshgrid(freqs.to('Hz').value, decs.to('rad').value, indexing='ij')
    freqs = jnp.asarray(freqs.flatten())
    decs = jnp.asarray(decs.flatten())

    target_log_psf_mean, target_log_psf_stddev = create_target(key, lmn, freqs, decs, 20)

    if init_config is not None:
        coords = []
        with open(init_config, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                x, y, z = line.strip().split(',')
                coords.append((float(x), float(y), float(z)))
        coords = np.asarray(coords)
        antennas = ac.EarthLocation.from_geocentric(
            coords[:, 0] * au.m,
            coords[:, 1] * au.m,
            coords[:, 2] * au.m
        )
        array_location = mean_itrs(antennas.get_itrs()).earth_location
    else:
        fill_registries()
        array = array_registry.get_instance(array_registry.get_match('dsa2000_31b'))
        antennas = array.get_antennas()
        array_location = array.get_array_location()
    obstime = at.Time('2021-01-01T00:00:00', format='isot', scale='utc')

    antennas = shift_antennas_too_close_to_boundary(antennas, array_location, obstime)
    antennas0, latitude = create_initial_data(antennas, obstime, array_location)
    x_init = antennas0
    for iteration in range(100):
        ball_centre, ball_radius = get_uniform_ball_prior(x_init, obstime, array_location)
        x, state, diagnostics = solve(ball_centre, ball_radius, lmn, latitude, freqs, decs, target_log_psf_mean, target_log_psf_stddev)
        x_init = x
        # Save state and solution
        save_pytree(state, 'state.json')
        save_pytree(diagnostics, f'diagnostics_{iteration}.json')
        with open(f'solution_{iteration}.txt', 'w') as f:
            f.write("#X_ITRS,Y_ITRS,Z_ITRS\n")
            antenna_locs = ENU(
                np.asarray(x[:, 0]) * au.m,
                np.asarray(x[:, 1]) * au.m,
                np.asarray(x[:, 2]) * au.m,
                location=array_location,
                obstime=obstime
            ).transform_to(ac.ITRS(obstime=obstime, location=array_location)).earth_location
            for row in antenna_locs:
                f.write(f"{row.x.to('m').value},{row.y.to('m').value},{row.z.to('m').value}\n")
        plot_solution(iteration, antennas0, latitude, x, ball_centre, ball_radius)


def plot_solution(iteration, antennas0, latitude, x, ball_centre, ball_radius):
    x0 = antennas0
    lmn = create_lmn2()
    freq = quantity_to_jnp(1350 * au.MHz)

    # row 1: Plot prior, ball centre and radius
    # row 2: Plot the antenna locations, and their movement from x0
    # row 3: Plot the UVW radial profile
    fig, ax = plt.subplots(3, 1, figsize=(6, 15))
    sc = ax[0].scatter(ball_centre[:, 0], ball_centre[:, 1], s=1, c=ball_radius, cmap='jet', vmin=0.)
    plt.colorbar(sc, ax=ax[0], label='Allowed movement (m)')
    ax[0].set_xlabel('East [m]')
    ax[0].set_ylabel('North [m]')
    ax[0].set_title('Prior')
    # Plot x0 and gradient from from x0 to x
    ax[1].scatter(x0[:, 0], x0[:, 1], s=1, c='black', alpha=0.1)
    arrow_length = jnp.linalg.norm(x - x0, axis=-1)
    ar = ax[1].quiver(
        x0[:, 0],
        x0[:, 1],
        x[:, 0] - x0[:, 0],
        x[:, 1] - x0[:, 1],
        arrow_length, scale=1,
        scale_units='xy',
        cmap='jet'
    )
    # color bar
    plt.colorbar(ar, label='Displacement (m)')
    ax[1].set_xlabel('East [m]')
    ax[1].set_ylabel('North [m]')
    ax[1].set_xlim(-10e3, 10e3)
    ax[1].set_ylim(-10e3, 10e3)
    ax[1].set_title(f'Solution: {iteration}')
    antenna1, antenna2 = np.asarray(list(itertools.combinations_with_replacement(range(x.shape[0]), 2)),
                                    dtype=jnp.int32).T
    uvw_radial = np.linalg.norm(x[antenna2] - x[antenna1], axis=-1)
    ax[2].hist(uvw_radial.flatten(), bins=np.arange(0, 20e3, 10))
    ax[2].set_xlabel('UVW radial distance [m]')
    ax[2].set_ylabel('Number of pairs')
    ax[2].set_title(f'UVW radial distance histogram: {iteration}')
    fig.savefig(f'array_solution_{iteration}.png', dpi=300)
    plt.close('all')

    # row 1: Plot the PSF, vmin=-80 vmax=0
    # row 2: Plot residuals, PSF - PSF0
    # row 3: Plot FWHM of both
    fig, ax = plt.subplots(3, 1, figsize=(6, 15))
    psf0 = 10. * np.log10(jax.jit(compute_psf)(x0, lmn, freq, latitude, 0.))  # [Nr, Nt]
    psf = 10. * np.log10(jax.jit(compute_psf)(x, lmn, freq, latitude, 0.))  # [Nr, Nt]
    fwhm = 10 ** (psf[0, :] / 10)
    fwhm0 = 10 ** (psf0[0, :] / 10)
    residuals = psf - psf0
    thetas = np.linspace(0, 2 * np.pi, lmn.shape[1], endpoint=False)
    sc = ax[0].scatter(
        lmn[..., 0].flatten(), lmn[..., 1].flatten(), c=psf.flatten(), s=1, cmap='jet',
        vmin=-70, vmax=-20
    )
    plt.colorbar(sc, ax=ax[0], label='Power (dB)')
    ax[0].set_xlabel('l (proj.rad)')
    ax[0].set_ylabel('m (proj.rad)')
    ax[0].set_title('PSF')
    sc = ax[1].scatter(
        lmn[..., 0].flatten(), lmn[..., 1].flatten(), c=residuals.flatten(), s=1, cmap='jet'
    )
    plt.colorbar(sc, ax=ax[1], label='Power (dB)')
    ax[1].set_xlabel('l (proj.rad)')
    ax[1].set_ylabel('m (proj.rad)')
    ax[1].set_title('PSF residuals (smaller is better)')
    ax[2].plot(thetas, fwhm0, label='FWHM0')
    ax[2].plot(thetas, fwhm, label='FWHM')
    ax[2].set_xlabel('Theta [rad]')
    ax[2].set_ylabel('Beam power')
    ax[2].set_title('FWHM')
    ax[2].legend()
    fig.savefig(f'psf_solution_{iteration}.png', dpi=300)
    plt.close('all')

    # row 1: Plot the PSF vs radius
    # row 2: Plot the residuals, PSF - PSF0 vs radius
    radii = np.linalg.norm(lmn[..., :2], axis=-1).flatten() * 180. / np.pi
    fig, ax = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    ax[0].scatter(radii, psf.flatten(), s=1)
    ax[0].set_xlabel('Radius [proj. degrees]')
    ax[0].set_ylabel('Beam power (dB)')
    ax[0].set_title('PSF vs Radius')
    ax[0].set_ylim(-80, 0)
    ax[1].scatter(radii, residuals.flatten(), s=1)
    ax[1].set_xlabel('Radius [proj. degrees]')
    ax[1].set_ylabel('Beam power (dB)')
    ax[1].set_title('PSF residuals vs Radius')
    fig.savefig(f'psf_vs_radius_solution_{iteration}.png', dpi=300)
    plt.close('all')

    # row 1: Plot the PSF at DEC=-90
    # row 2: Plot the PSF at DEC=-45
    # row 3: Plot the PSF at DEC=0
    # row 4: Plot the PSF at DEC=+45
    # row 5: Plot the PSF at DEC=+90
    psf = 10. * np.log10(jax.jit(compute_psf)(x, lmn, freq, latitude, - np.pi/2))  # [Nr, Nt]
    psf_45 = 10. * np.log10(jax.jit(compute_psf)(x, lmn, freq, latitude, - np.pi/4))  # [Nr, Nt]
    psf_m45 = 10. * np.log10(jax.jit(compute_psf)(x, lmn, freq, latitude, 0.))  # [Nr, Nt]
    psf_90 = 10. * np.log10(jax.jit(compute_psf)(x, lmn, freq, latitude, np.pi/4))  # [Nr, Nt]
    psf_m90 = 10. * np.log10(jax.jit(compute_psf)(x, lmn, freq, latitude, np.pi/2))  # [Nr, Nt]
    fig, ax = plt.subplots(5, 1, figsize=(6, 20), sharex=True, sharey=True)
    for i, (p, title) in enumerate(
            [(psf_m90, 'DEC=-90'), (psf_m45, 'DEC=-45'), (psf, 'DEC=0'), (psf_45, 'DEC=+45'), (psf_90, 'DEC=+90')]):
        sc = ax[i].scatter(
            lmn[..., 0].flatten(), lmn[..., 1].flatten(), c=p.flatten(), s=1, cmap='jet',
            vmin=-70, vmax=-20
        )
        plt.colorbar(sc, ax=ax[i], label='Power (dB)')
        ax[i].set_xlabel('l (proj.rad)')
        ax[i].set_ylabel('m (proj.rad)')
        ax[i].set_title(f'PSF at {title}')
    fig.savefig(f'psf_dec_solution_{iteration}.png', dpi=300)

    # row 1: Plot the PSF vs radius at DEC=-90
    # row 2: Plot the PSF vs radius at DEC=-45
    # row 3: Plot the PSF vs radius at DEC=0
    # row 4: Plot the PSF vs radius at DEC=+45
    # row 5: Plot the PSF vs radius at DEC=+90
    radii = np.linalg.norm(lmn[..., :2], axis=-1).flatten() * 180. / np.pi
    fig, ax = plt.subplots(5, 1, figsize=(6, 20), sharex=True)
    for i, (p, title) in enumerate(
            [(psf_m90, 'DEC=-90'), (psf_m45, 'DEC=-45'), (psf, 'DEC=0'), (psf_45, 'DEC=+45'), (psf_90, 'DEC=+90')]):
        ax[i].scatter(radii, p.flatten(), s=1)
        ax[i].set_xlabel('Radius [proj. degrees]')
        ax[i].set_ylabel('Beam power (dB)')
        ax[i].set_title(f'PSF vs Radius at {title}')
        ax[i].set_ylim(-70, 0)
    fig.savefig(f'psf_vs_radius_dec_solution_{iteration}.png', dpi=300)
    plt.close('all')


def shift_antennas_too_close_to_boundary(antennas, array_location, obstime):
    # Shift antennas if too close to boundary, so that search priors can have non-zero radius
    antennas_before = antennas
    # Minimal allowed movement
    init_freedom = 10.
    antennas_before_enu = antennas_before.get_itrs(
        obstime=obstime, location=array_location
    ).transform_to(
        ENU(obstime=obstime, location=array_location)
    ).cartesian.xyz.to('m').value.T
    tree = KDTree(antennas_before_enu)
    dist, _ = tree.query(antennas_before_enu, k=2)
    dist = dist[:, 1]
    force_relocate = dist < 8. + init_freedom
    while True:
        relocated_antennas = relocate_antennas(antennas, obstime, array_location,
                                               additional_buffer=init_freedom, force_relocate=force_relocate)
        relocated_antennas_enu = relocated_antennas.get_itrs(
            obstime=obstime, location=array_location
        ).transform_to(
            ENU(obstime=obstime, location=array_location)
        ).cartesian.xyz.to('m').value.T
        tree = KDTree(relocated_antennas_enu)
        dist, _ = tree.query(relocated_antennas_enu, k=2)
        dist = dist[:, 1]
        if np.any(dist < 8. + init_freedom):
            print(f"Some {np.sum(dist < 8. + init_freedom)} antennas are still too close to each other, retrying")
            # force_relocate = dist < 8. + init_freedom
            continue
        break
    antennas = relocated_antennas
    plot_relocated_antennas(antennas_before, antennas, obstime, array_location)
    return antennas


if __name__ == '__main__':
    main(init_config='init_config.txt')


