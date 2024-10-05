import os

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"
from scipy.spatial import KDTree

import dataclasses
from jaxns import Model, Prior

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
from dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardt
from jaxns import save_pytree
import jax.numpy as jnp
import numpy as np
from functools import partial
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from tomographic_kernel.frames import ENU

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp

tfpd = tfp.distributions

from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map
from typing import Tuple
from dsa2000_cal.common.jax_utils import create_mesh


def rotate_coords_to_dec0(antennas: jax.Array, latitude: jax.Array) -> jax.Array:
    east, north, up = antennas[..., 0], antennas[..., 1], antennas[..., 2]
    east_prime = east
    north_prime = jnp.cos(latitude) * north - jnp.sin(latitude) * up
    up_prime = jnp.sin(latitude) * north + jnp.cos(latitude) * up
    return jnp.stack([east_prime, north_prime, up_prime], axis=-1)


def compute_psf(antennas: jax.Array, lmn: jax.Array, freq: jax.Array, latitude: jax.Array) -> jax.Array:
    """
    Compute the point spread function of the array

    Args:
        antennas: [N, 3]
        lmn: [..., , 3]
        freq: []

    Returns:
        psf: [...]
    """
    mesh = create_mesh((len(jax.devices()),), ('shard',), devices=jax.devices())

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(PartitionSpec(), PartitionSpec('shard', ), PartitionSpec(), PartitionSpec()),
        out_specs=PartitionSpec('shard', )
    )
    def compute_shard_psf(antennas, lmn_shard, freq, latitude):
        antennas = rotate_coords_to_dec0(antennas, latitude)
        wavelength = mp_policy.cast_to_length(299792458. / freq)
        r = antennas / wavelength
        delay = mp_policy.cast_to_vis(
            -2j * jnp.pi * jnp.sum(r[..., :2] * lmn_shard[..., None, :2], axis=-1))  # [..., N]
        N = antennas.shape[0]
        voltage_beam = jnp.mean(jnp.exp(delay), axis=-1)  # [...]
        power_beam = jnp.reciprocal(N - 1) * (N * (voltage_beam * voltage_beam.conj()).real - 1)
        psf = power_beam  # [...]
        return psf

    return compute_shard_psf(antennas, lmn, freq, latitude)


@dataclasses.dataclass(eq=False)
class OptimisationProblem:
    batch_size: int = 1024
    num_radial_bins: int = 3600 // 8
    num_theta_bins: int = 10
    lmax: au.Quantity = 1 * au.deg
    lmin: au.Quantity = 8 * au.arcsec
    fwhm: au.Quantity = 3.3 * au.arcsec
    freq: au.Quantity = 1350. * au.MHz

    def __post_init__(self):
        self.devices = jax.devices()
        self.mesh = create_mesh((len(self.devices),), ('shard',), devices=self.devices)

    def create_data(self, antennas: ac.EarthLocation, obstime: at.Time, array_location: ac.EarthLocation):
        lmax = quantity_to_jnp(self.lmax, 'rad')
        lmin = quantity_to_jnp(self.lmin, 'rad')
        fwhm = quantity_to_jnp(self.fwhm, 'rad')
        radii = jnp.concatenate([0.5 * fwhm[None], jnp.linspace(lmin, lmax, self.num_radial_bins)])
        theta = jnp.linspace(0., 2 * np.pi, self.num_theta_bins, endpoint=False)

        R, Theta = jnp.meshgrid(
            radii, theta,
            indexing='ij'
        )

        L = R * jnp.cos(Theta)
        M = R * jnp.sin(Theta)
        N = jnp.sqrt(1 - L ** 2 - M ** 2)
        lmn = jnp.stack([L, M, N], axis=-1)  # [Nr, Nt, 3]

        antenna_enu_xyz = antennas.get_itrs(
            obstime=obstime, location=array_location).transform_to(
            ENU(obstime=obstime, location=array_location)).cartesian.xyz.to('m').T

        antennas0 = quantity_to_jnp(antenna_enu_xyz, 'm')  # [N, 3]

        freq = quantity_to_jnp(self.freq, 'Hz')

        latitude = quantity_to_jnp(array_location.geodetic.lat)

        return antennas0, lmn, freq, latitude


def compute_mu_sigma_X(mu_Y, sigma_Y):
    # Compute the standard deviation of X
    sigma_X = jnp.sqrt(jnp.log(1 + (sigma_Y ** 2) / (mu_Y ** 2)))

    # Compute the mean of X
    mu_X = jnp.log(mu_Y) - 0.5 * (sigma_X ** 2)

    return mu_X, sigma_X


def compute_residuals(antenna_locations: jax.Array, lmn: jax.Array,
                      freq: jax.Array, latitude: jax.Array) -> Tuple[jax.Array, jax.Array]:
    psf = compute_psf(antenna_locations, lmn, freq, latitude)  # [Nr, Nt]
    fwhm_ring = psf[0, :]  # [Nt]
    sidelobes = psf[1:, :]  # [Nr-1, Nt]
    # Only positive psf values can be optimised by configuration
    pos_mask = sidelobes > 0.
    residual_fwhm = (jnp.log(fwhm_ring) - jnp.log(0.5)) / 0.02
    residual_fwhm = jnp.where(jnp.isnan(residual_fwhm), 0., residual_fwhm)
    log_sidelobe = jnp.log(sidelobes)
    stopped_log_sidelobe = jax.lax.stop_gradient(log_sidelobe)
    threshold = jnp.asarray(np.log(1e-3), psf.dtype)
    residual_sidelobes = jnp.where(pos_mask, (log_sidelobe - stopped_log_sidelobe - jnp.log(0.98)) / 1., 0.)
    # Only the top 10% of sidelobes are optimised at a given time.
    residual_sidelobes = jnp.where(
        log_sidelobe < threshold,
        0.,
        residual_sidelobes
    )
    return residual_fwhm, residual_sidelobes


def construct_prior(x0, forbidden_points):
    """
    Construct a prior for the antenna locations

    Args:
        x0: [N, 2]
        forbidden_points: [S, 2]

    Returns:
        sigma: [N]
    """
    # For each point find the closest forbidden point, and compute distance using kdtree
    tree = KDTree(forbidden_points)
    d, _ = tree.query(x0)
    sigma = 0.33 * d
    return sigma


def sample_ball(origin, radius, num_samples: int):
    radius = radius * np.random.uniform(0, 1, num_samples) ** 0.5
    random_direction = np.random.normal(size=(num_samples, 3))
    random_direction[:, 2] = 0.
    random_direction /= np.linalg.norm(random_direction, axis=-1)[:, None]
    return origin + radius[:, None] * random_direction


# Define constaints by 


@partial(jax.jit)
def solve(init_state, x0, sigma, lmn, freq, latitude):
    def prior_model():

        x = yield Prior(tfpd.Normal(x0[:, :2], sigma[:, None]),
                        'x').parametrised()
        x = jnp.concatenate([x, jnp.zeros((x0.shape[0], 1), x.dtype)], axis=-1)
        return x

    def log_likelihood(x):
        residuals = compute_residuals(x, lmn, freq, latitude)
        return sum([-jnp.sum(jnp.square(r)) for r in jax.tree.leaves(residuals)])

    model = Model(prior_model, log_likelihood)
    U = model.sample_U(jax.random.key(0))

    def residuals(params):
        (x,) = model(params).prepare_input(U)
        return compute_residuals(x, lmn, freq, latitude)

    solver = MultiStepLevenbergMarquardt(
        residual_fn=residuals,
        num_iterations=1,
        num_approx_steps=0,
        delta=2,
        mu1=100.,
        p_any_improvement=0.01,
        p_less_newton=0.25,
        p_sufficient_improvement=0.5,
        p_more_newton=0.8,
        c_more_newton=0.2,
        c_less_newton=1.5,
        verbose=True
    )
    if init_state is None:
        state = solver.create_initial_state(model.params)
    else:
        state = solver.update_initial_state(init_state)
    state, diagnostics = solver.solve(state)
    return model(state.x).prepare_input(U)[0], state, diagnostics


def main():
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000_31b'))
    antennas = array.get_antennas()
    array_location = array.get_array_location()
    obstime = at.Time('2021-01-01T00:00:00', format='isot', scale='utc')

    problem = OptimisationProblem(num_radial_bins=len(jax.devices()) * 10 - 1, num_theta_bins=len(jax.devices()) * 10,
                                  lmax=2 * au.arcmin)

    antennas0, lmn, freq, latitude = problem.create_data(
        antennas=antennas, obstime=obstime, array_location=array_location
    )

    # Plot antennas in ENU
    antennas_enu_xyz = antennas.get_itrs(
        obstime=obstime, location=array_location).transform_to(
        ENU(obstime=obstime, location=array_location)).cartesian.xyz.to('m').T.value

    antennas_enu_xyz_rot = rotate_coords_to_dec0(antennas_enu_xyz, latitude)

    plt.scatter(antennas_enu_xyz[:, 0], antennas_enu_xyz[:, 1], s=1, c='black', label='ENU')
    plt.scatter(antennas_enu_xyz_rot[:, 0], antennas_enu_xyz_rot[:, 1], s=1, c='red', label='DEC=0 projection')
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.legend()
    plt.title('Antenna locations')
    plt.xlim(-10e3, 10e3)
    plt.ylim(-10e3, 10e3)
    plt.savefig('antennas.png')
    plt.close('all')

    # Plot UVW radial profile rotated to Dec=0 in bins of 10m

    uvw_radial = np.linalg.norm(antennas_enu_xyz_rot[:, None, :2] - antennas_enu_xyz_rot[None, :, :2], axis=-1)
    bins = np.arange(0, 20e3, 10)
    uvw_radial_flat = uvw_radial.flatten()

    plt.hist(uvw_radial_flat, bins=bins)
    plt.xlabel('UVW radial distance [m]')
    plt.ylabel('Number of pairs')
    plt.title('UVW radial distance histogram')
    plt.savefig('uvw_radial_hist.png')
    plt.close('all')

    psf0 = jax.jit(compute_psf)(antennas0, lmn, freq, latitude)  # [Nr, Nt]
    print(psf0[0, :])  # FWHM
    thetas = np.linspace(0, 2 * np.pi, problem.num_theta_bins, endpoint=False)
    plt.plot(thetas, psf0[0, :], label='FWHM')
    plt.xlabel('Theta [rad]')
    plt.ylabel('Beam power')
    plt.title('FWHM')
    plt.legend()
    plt.savefig('fwhm.png')
    plt.close('all')

    sc = plt.scatter(
        lmn[..., 0].flatten(), lmn[..., 1].flatten(), c=jnp.log10(psf0.flatten()), s=1, cmap='jet'
    )
    plt.colorbar(sc)
    plt.xlabel('l (proj.rad)')
    plt.ylabel('m (proj.rad)')
    plt.title('PSF')
    plt.close('all')

    radii = np.linalg.norm(lmn[..., :2], axis=-1).flatten()
    log_psf_radii = 10 * np.log10(psf0).flatten()

    plt.scatter(radii, log_psf_radii, s=1)
    plt.xlabel('Radius [proj.rad]')
    plt.ylabel('Beam power (dB)')
    plt.title('PSF vs Radius')
    plt.savefig('psf_vs_radius.png')
    plt.close('all')

    # Setup 

    x0 = np.array(antennas0.copy())

    exclusion_center = np.asarray([-7000, 2000, 0.])
    exclusion_radius = 1500

    # Create forbidden samples
    forbidden_points = sample_ball(exclusion_center, exclusion_radius, 2000)
    # Move these antennas to another place
    t = KDTree(forbidden_points)
    d, _ = t.query(x0)
    point_in_forbidden = d < 100.
    # Choose another than is not forbidden
    choose_prob = np.bitwise_not(point_in_forbidden) / jnp.sum(np.bitwise_not(point_in_forbidden))
    good = np.random.choice(x0.shape[0], np.sum(point_in_forbidden), p=choose_prob, replace=True)
    x0[point_in_forbidden, :2] = x0[good, :2] + np.random.normal(size=(good.size, 2)) * 100.
    # For sigma make the distance to exclusion center 3 sigma, from x0
    sigma = construct_prior(x0, forbidden_points)  # [n]

    sc = plt.scatter(x0[:, 0], x0[:, 1], s=1, c='black')
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.title(r'Aperture constraint')

    # Draw grey circle for exclusion zone
    circle = plt.Circle(exclusion_center, exclusion_radius, color='grey', fill=True, alpha=0.5)
    plt.gca().add_artist(circle)

    plt.savefig('aperture_constraint.png')
    plt.subplots_adjust(right=0.8)
    plt.close('all')

    sc = plt.scatter(x0[:, 0], x0[:, 1], s=1, c=sigma, cmap='jet')
    plt.colorbar(sc, label=r'$\sigma$ (m)')
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.title(r'$x_0$ and $\sigma$')

    # Draw grey circle for exclusion zone
    circle = plt.Circle(exclusion_center, exclusion_radius, color='grey', fill=True, alpha=0.5)
    plt.gca().add_artist(circle)

    plt.savefig('x0_sigma.png')
    plt.close('all')

    # Run

    problem = OptimisationProblem(num_radial_bins=len(jax.devices()) * 20 - 1, num_theta_bins=len(jax.devices()) * 20,
                                  lmax=3 * au.deg)

    antennas0, lmn, freq, latitude = problem.create_data(
        antennas=antennas, obstime=obstime, array_location=array_location
    )

    state = None
    x_iter = x0
    for iteration in range(100):
        sigma = construct_prior(x_iter, forbidden_points)
        x, state, diagnostics = solve(state, x0, sigma, lmn, freq, latitude)
        x_iter = x
        save_pytree(state, 'state.json')
        with open('solution_{iteration}.json', 'w') as f:
            f.write(jax.tree_util.tree_map(lambda x: x.tolist(), x))

        # Plot x0 and gradient from from x0 to x
        # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # plt.scatter(x0[:, 0], x0[:, 1], s=1, c=sigma, cmap='jet')
        arrow_length = jnp.linalg.norm(x - x0, axis=-1)
        ar = plt.quiver(x0[:, 0], x0[:, 1], x[:, 0] - x0[:, 0], x[:, 1] - x0[:, 1], arrow_length, scale=1,
                        scale_units='xy',
                        cmap='jet')
        # color bar
        plt.colorbar(ar, label='Displacement (m)')
        # grey circle
        circle = plt.Circle(exclusion_center, exclusion_radius, color='grey', fill=True, alpha=0.5)
        plt.gca().add_artist(circle)
        plt.xlabel('East [m]')
        plt.ylabel('North [m]')
        plt.xlim(-10e3, 10e3)
        plt.ylim(-10e3, 10e3)
        plt.title('Solution')
        plt.tight_layout()
        plt.savefig(f'array_solution_{iteration}.png')
        plt.close('all')

    problem = OptimisationProblem(num_radial_bins=len(jax.devices()) * 10 - 1, num_theta_bins=len(jax.devices()) * 10,
                                  lmax=3 * au.deg)

    antennas0, lmn, freq, latitude = problem.create_data(
        antennas=antennas, obstime=obstime, array_location=array_location
    )

    psf0 = jax.jit(compute_psf)(antennas0, lmn, freq, latitude)  # [Nr, Nt]
    psf = jax.jit(compute_psf)(x, lmn, freq, latitude)  # [Nr, Nt]
    thetas = np.linspace(0, 2 * np.pi, problem.num_theta_bins, endpoint=False)
    plt.plot(thetas, psf[0, :], label='FWHM')
    plt.xlabel('Theta [rad]')
    plt.ylabel('Beam power')
    plt.title('FWHM')
    plt.legend()
    plt.savefig('fwhm.png')
    plt.close('all')

    sc = plt.scatter(
        lmn[..., 0].flatten(), lmn[..., 1].flatten(), c=jnp.log10(psf.flatten()), s=1, cmap='jet'
    )
    plt.colorbar(sc)
    plt.xlabel('l (proj.rad)')
    plt.ylabel('m (proj.rad)')
    plt.title('PSF')
    plt.close('all')

    radii = np.linalg.norm(lmn[..., :2], axis=-1).flatten()
    log_psf_radii = 10 * np.log10(psf).flatten()
    log_psf0_radii = 10 * np.log10(psf0).flatten()
    log_residuals = (log_psf_radii - log_psf0_radii)

    plt.scatter(radii, log_residuals, s=1)
    plt.xlabel('Radius [proj.rad]')
    plt.ylabel('Beam power (dB)')
    plt.title('PSF residuals vs Radius')
    plt.savefig('psf_residuals.png')
    plt.close('all')

    ##


if __name__ == '__main__':
    main()
