from functools import partial

import jax
import numpy as np
from astropy import time as at
from jax import numpy as jnp
from tomographic_kernel.frames import ENU

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from main import rotation_matrix_change_dec, rotate_coords, compute_ideal_psf_distribution


def test_rotation_matrix_change_dec():
    R = rotation_matrix_change_dec(jnp.pi / 2.)
    x = jnp.asarray([1., 0., 0.])
    np.testing.assert_allclose(R @ x, jnp.asarray([1., 0., 0.]), atol=1e-6)

    x = jnp.asarray([0., 1., 0.])
    np.testing.assert_allclose(R @ x, jnp.asarray([0., 0., 1.]), atol=1e-6)

    x = jnp.asarray([0., 0., 1.])
    np.testing.assert_allclose(R @ x, jnp.asarray([0., -1., 0.]), atol=1e-6)


def test_rotate_coords():
    # case: east hat at dec=0, viewed from dec=pi/2 gives east hat
    antennas_projected = jnp.asarray([1., 0., 0.])
    antennas_enu = rotate_coords(antennas_projected, 0., jnp.pi / 2)
    np.testing.assert_allclose(antennas_enu, jnp.asarray([1., 0., 0.]), atol=1e-6)
    # case: north hat at dec=0, viewed from dec=pi/2 gives w-vec (up)
    antennas_projected = jnp.asarray([0., 1., 0.])
    antennas_enu = rotate_coords(antennas_projected, 0., jnp.pi / 2)
    np.testing.assert_allclose(antennas_enu, jnp.asarray([0., 0., 1.]), atol=1e-6)
    # case: up hat at dec=0, viewed from dec=pi/2 gives -v hat (south)
    antennas_projected = jnp.asarray([0., 0., 1.])
    antennas_enu = rotate_coords(antennas_projected, 0., jnp.pi / 2)
    np.testing.assert_allclose(antennas_enu, jnp.asarray([0., -1., 0.]), atol=1e-6)


def test_ideal_psf():
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

    lvec = mvec = 1 / 60 * np.pi / 180 * jnp.linspace(-1, 1, 500)
    L, M = jnp.meshgrid(lvec, mvec, indexing='ij')
    lmn = jnp.stack([L, M, 1 - jnp.sqrt(1 - L ** 2 - M ** 2)], axis=-1)
    freq = 1350e6
    transit_dec = 0.
    key = jax.random.PRNGKey(0)
    log_psf_mean, log_psf_std = jax.jit(partial(compute_ideal_psf_distribution, num_samples=20))(
        key, lmn, freq, latitude, transit_dec, antennas_enu_xyz
    )

    import pylab as plt
    plt.imshow(
        log_psf_mean.T,
        origin='lower',
        extent=[lvec[0], lvec[-1], mvec[0], mvec[-1]],
        vmin=-60, vmax=10 * np.log10(0.5),
        aspect='auto',
        cmap='jet',
        interpolation='nearest'
    )
    plt.colorbar()
    plt.xlabel("l")
    plt.ylabel("m")
    plt.title('Ideal PSF mean')
    plt.show()

    plt.imshow(
        log_psf_std.T,
        origin='lower',
        extent=[lvec[0], lvec[-1], mvec[0], mvec[-1]],
        aspect='auto',
        cmap='jet',
        interpolation='nearest'
    )
    plt.colorbar()
    plt.xlabel("l")
    plt.ylabel("m")
    plt.title('Ideal PSF std')
    plt.show()
