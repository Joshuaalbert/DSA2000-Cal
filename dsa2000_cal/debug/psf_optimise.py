import itertools

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from tomographic_kernel.frames import ENU

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_common.common.mixed_precision_utils import mp_policy
from dsa2000_common.common.quantity_utils import time_to_jnp
from dsa2000_common.delay_models.base_far_field_delay_engine import build_far_field_delay_engine




def compute_mu_sigma_X(mu_Y, sigma_Y):
    # Compute the standard deviation of X
    sigma_X = jnp.sqrt(jnp.log(1 + (sigma_Y ** 2) / (mu_Y ** 2)))

    # Compute the mean of X
    mu_X = jnp.log(mu_Y) - 0.5 * (sigma_X ** 2)

    return mu_X, sigma_X


def compute_psf_from_uvw(uv, lm: jax.Array, freq: jax.Array, batch_size):
    wavelength = mp_policy.cast_to_length(299792458. / freq)
    uv /= wavelength
    row = uv.shape[0]

    # Split uv into batches
    uv_batches = uv.reshape(row // batch_size, batch_size, 2)

    def body(psf, uv_batch):
        delay = jnp.sum(uv_batch * lm[..., None, :], axis=-1)  # [Nr, Nt, batch_size]
        psf += mp_policy.cast_to_image(jnp.sum(jnp.cos(2 * jnp.pi * delay), axis=-1))  # [Nr, Nt]
        return psf, None

    psf_init = jnp.zeros(lm.shape[:-1], dtype=mp_policy.image_dtype)
    psf, _ = jax.lax.scan(body, psf_init, uv_batches)
    psf /= row
    return psf


def compute_psf(antenna_locations: jax.Array, lm: jax.Array, freq: jax.Array, batch_size: int = 1) -> jax.Array:
    antenna1, antenna2 = jnp.asarray(list(itertools.combinations_with_replacement(range(n), 2)),
                                     dtype=mp_policy.index_dtype).T
    uv = antenna_locations[antenna2] - antenna_locations[antenna1]  # [row, 2]
    return compute_psf_from_uvw(uv, lm, freq, batch_size)


def compute_dynamic_range(antenna_locations: jax.Array, lm: jax.Array, freq: jax.Array,
                          batch_size: int = 1):
    psf = compute_psf(antenna_locations, lm, freq, batch_size)
    psf_mean = jnp.mean(psf, axis=-1)
    psf_std = jnp.std(psf, axis=-1)
    # Convert mean and std to log-normal
    psf_mean_log, psf_std_log = compute_mu_sigma_X(psf_mean, psf_std)
    # Require 1/psf_mean_log to be greater than a threshold
    log_dr_mean = -psf_mean_log
    log_dr_std = psf_std_log
    return log_dr_mean, log_dr_std


tfpd = tfp.distributions

if __name__ == '__main__':
    freq = 700 * au.MHz
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    antennas = array.get_antennas()
    n = len(antennas)
    location = antennas[0]
    obstime = at.Time('2021-01-01T00:00:00', format='isot', scale='utc')
    ref_time = obstime

    phase_center = ENU(0, 0, 1, obstime=obstime, location=location).transform_to(ac.ICRS())
    engine = build_far_field_delay_engine(
        antennas=antennas,
        start_time=obstime,
        end_time=obstime + (10.3 * 60) * au.s,
        ref_time=ref_time,
        phase_center=phase_center,
        resolution=60 * au.s,
        verbose=True
    )

    antenna1, antenna2 = jnp.asarray(list(itertools.combinations_with_replacement(range(n), 2)),
                                     dtype=mp_policy.index_dtype).T

    times = obstime + np.arange(0, 10.3 * 60, 60) * au.s
    f_jit = jax.jit(engine.compute_uvw)
    uv = []
    for t in times:
        _times = jnp.repeat(time_to_jnp(t, ref_time)[None], len(antenna1))
        uvw = f_jit(
            _times,
            antenna1,
            antenna2
        )
        uv.append(uvw[:, :2])
    uv = jnp.concatenate(uv, axis=0)

    dr_constraint = 1e6  #

    num_radial_bins = 100
    num_theta_bins = 10
    lmax = (1 * au.arcmin).to('rad').value
    theta = jnp.linspace(0., 2 * np.pi, num_theta_bins)
    radii = jnp.linspace(0., lmax, num_radial_bins)

    R, Theta = jnp.meshgrid(
        radii, theta,
        indexing='ij'
    )

    L = R * jnp.cos(Theta)  # [Nr, Nt]
    M = R * jnp.sin(Theta)  # [Nr, Nt]

    lm = jnp.stack([L, M], axis=-1)  # [Nr, Nt, 2]

    # convert to ENU

    antennas_itrs = antennas.get_itrs(obstime=obstime, location=location)
    antennas_enu = antennas_itrs.transform_to(ENU(obstime=obstime, location=location))

    antennas_enu_xyz = antennas_enu.cartesian.xyz.T  # [n, 3]
    min_east = jnp.min(antennas_enu_xyz[:, 0])
    max_east = jnp.max(antennas_enu_xyz[:, 0])
    d_east = (max_east - min_east) / 10
    min_north = jnp.min(antennas_enu_xyz[:, 1])
    max_north = jnp.max(antennas_enu_xyz[:, 1])
    d_north = (max_north - min_north) / 10

    plt.scatter(antennas_enu_xyz[:, 0], antennas_enu_xyz[:, 1],
                c='black', s=1)
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.title('Initial antenna locations')
    plt.show()

    plt.scatter(uv[:, 0], uv[:, 1],
                c='black', s=1)
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.title('UV coverage')
    plt.show()
    #
    # antenna_locations0 = antennas_enu_xyz[:, :2]
    #
    # # Time it
    # t0 = time.time()
    # psf0 = jax.block_until_ready(
    #     jax.jit(lambda *args: compute_psf_from_uvw(*args, batch_size=n // 2))(
    #         uv, lm,
    #         quantity_to_jnp(freq)
    #     )
    # )
    # print('Time taken for JIT compiled function:', time.time() - t0)
    #
    # sc = plt.scatter(
    #     lm[..., 0].flatten(), lm[..., 1].flatten(), c=psf0.flatten(), s=1, cmap='inferno'
    # )
    # plt.colorbar(sc)
    # plt.xlabel('l (proj.rad)')
    # plt.ylabel('m (proj.rad)')
    # plt.title('PSF')
    # plt.show()
    #
    # psf0_mean = jnp.mean(psf0, axis=-1)
    # psf0_std = jnp.std(psf0, axis=-1)
    # # Convert mean and std to log-normal
    # psf0_mean_log, psf0_std_log = compute_mu_sigma_X(psf0_mean, psf0_std)
    #
    # radii_arcmin = (radii.tolist() * au.rad).to('arcmin').value
    #
    # plt.plot(radii_arcmin, psf0_mean_log)
    # plt.fill_between(radii_arcmin, psf0_mean_log - psf0_std_log, psf0_mean_log + psf0_std_log, alpha=0.2)
    # plt.xlabel('Radius [arcmin]')
    # plt.ylabel('Mean PSF')
    # plt.title('Mean PSF vs Radius')
    # plt.show()
    #
    #
    # def prior_model():
    #     ones = jnp.ones((n, 2))
    #     dx = yield Prior(
    #         tfpd.Uniform(
    #             -ones * jnp.asarray([d_east, d_north]),
    #             ones * jnp.asarray([d_east, d_north])
    #         ),
    #         name='antenna_locations'
    #     ).parametrised()
    #     antenna_locations = antenna_locations0 + dx
    #     psf = compute_psf(
    #         antenna_locations=antenna_locations,
    #         lm=lm,
    #         freq=quantity_to_jnp(freq)
    #     )
    #     return psf, antenna_locations
    #
    #
    # def log_likelihood(psf, model_psf):
    #     ...
