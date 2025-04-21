import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_common.common.ellipse_utils import Gaussian, ellipse_rotation
from dsa2000_common.common.fourier_utils import ApertureTransform


def test_ellipse_definitions():
    ellipse = Gaussian(
        x0=jnp.asarray([0., 0.]),
        major_fwhm=1,
        minor_fwhm=0.5,
        pos_angle=0.,
        total_flux=Gaussian.total_flux_from_peak(1, major_fwhm=1, minor_fwhm=0.5)
    )
    # [minor, major]
    assert ellipse.compute_flux_density(np.array([0., 0.])) == 1
    assert ellipse.compute_flux_density(np.array([0.0, 0.5])) == 1 / 2
    assert ellipse.compute_flux_density(np.array([0.25, 0.])) == 1 / 2

def test_ellipse_fourier():
    arcsec2rad = np.pi / 180. / 3600.
    major = 40 * arcsec2rad
    minor = 20 * arcsec2rad
    peak = 1

    dx = 10/0.22 # m
    n = (int(16000/dx/0.22)//2) * 2
    dl = 1/(n * dx)  # m
    xvec = (-0.5 * n + np.arange(n)) * dx
    lvec = (-0.5 * n + np.arange(n)) * dl
    U,V = np.meshgrid(xvec, xvec, indexing='ij')
    UV = np.stack([U, V], axis=-1)
    L,M = np.meshgrid(lvec, lvec, indexing='ij')

    g = Gaussian(
        x0=jnp.asarray([0., 0.]),
        major_fwhm=major,
        minor_fwhm=minor,
        pos_angle=0.,
        total_flux=Gaussian.total_flux_from_peak(peak, major_fwhm=major, minor_fwhm=minor)
    )

    vis = jax.vmap(jax.vmap(g.fourier))(UV)
    a = ApertureTransform()
    img = a.to_image(vis, axes=(-2, -1), dx=dx, dy=dx).real
    import pylab as plt
    plt.imshow(img.T[n//2 - 100: n//2 + 100, n//2 - 100: n//2 + 100], origin='lower', extent=[lvec.min(), lvec.max(), lvec.min(), lvec.max()],
                interpolation='none', )
    plt.colorbar()
    plt.show()

    np.testing.assert_allclose(np.max(img), 1, atol=1e-5) # peak flux density


def test_ellipse():
    n = 2096
    ellipse = Gaussian(
        x0=jnp.asarray([1., 0.]),
        major_fwhm=2.5,
        minor_fwhm=1.5,
        pos_angle=np.pi / 4.,
        total_flux=1.
    )

    x_vec = jnp.linspace(-5, 5, n)
    dx = x_vec[1] - x_vec[0]
    X, Y = jnp.meshgrid(x_vec, x_vec, indexing='ij')
    x = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
    flux_density = jax.vmap(ellipse.compute_flux_density)(x).reshape(X.shape)  # [Nl, Nm]
    flux = flux_density * dx ** 2
    import matplotlib.pyplot as plt
    plt.imshow(flux.T, origin='lower', extent=[x_vec.min(), x_vec.max(), x_vec.min(), x_vec.max()],
               interpolation='none', )
    plt.colorbar()
    plt.show()

    total_flux_estimate = jnp.sum(flux)
    np.testing.assert_allclose(total_flux_estimate, ellipse.total_flux, atol=1e-2)
    np.testing.assert_allclose(jnp.max(flux), ellipse.peak_flux_density() * dx ** 2, atol=1e-2)

    # Compute beam area
    mask = flux >= 0.5 * flux.max()
    beam_area_approx = jnp.sum(mask) * dx ** 2
    np.testing.assert_allclose(beam_area_approx, ellipse.beam_area(), atol=1e-2)

    plt.imshow(mask.T, origin='lower', extent=[x_vec.min(), x_vec.max(), x_vec.min(), x_vec.max()],
               interpolation='none', )
    plt.colorbar()
    plt.show()

    k_vec = jnp.fft.fftshift(jnp.fft.fftfreq(n, d=dx))

    K1, K2 = jnp.meshgrid(k_vec, k_vec, indexing='ij')
    fourier_flux_estimate = jnp.fft.fftshift(jnp.fft.fft2(flux))

    k = jnp.stack([K1.flatten(), K2.flatten()], axis=-1)
    fourier = jax.vmap(ellipse.fourier)(k).reshape(K1.shape)

    np.testing.assert_allclose(np.abs(fourier), np.abs(fourier_flux_estimate), atol=1e-3)

    plt.imshow(jnp.angle(fourier_flux_estimate).T, origin='lower',
               extent=[k_vec.min(), k_vec.max(), k_vec.min(), k_vec.max()],
               interpolation='none',
               # alpha=jnp.abs(fourier_flux_estimate).T
               )
    plt.colorbar()
    plt.show()

    plt.imshow(jnp.angle(fourier).T, origin='lower',
               extent=[k_vec.min(), k_vec.max(), k_vec.min(), k_vec.max()],
               interpolation='none',
               # alpha=jnp.abs(fourier).T
               )
    plt.colorbar()
    plt.show()

    diff = fourier_flux_estimate - fourier

    plt.imshow(np.log10(jnp.abs(diff).T), origin='lower',
               interpolation='none',
               extent=[k_vec.min(), k_vec.max(), k_vec.min(), k_vec.max()])
    plt.colorbar()
    plt.show()

    fourier_inv = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(fourier))).real
    plt.imshow(fourier_inv.T, origin='lower',
               interpolation='none',
               extent=[x_vec.min(), x_vec.max(), x_vec.min(), x_vec.max()])
    plt.colorbar()
    plt.show()
    np.testing.assert_allclose(fourier_inv, flux, atol=1e-3)


def test_ellipse_rotation():
    # (x,y) -> (x',y')
    up = np.asarray([0, 1])
    right = np.asarray([1, 0])
    left = np.asarray([-1, 0])
    down = np.asarray([0, -1])
    np.testing.assert_allclose(ellipse_rotation(np.pi / 2.) @ up, right, atol=1e-6)
    np.testing.assert_allclose(ellipse_rotation(np.pi) @ up, down, atol=1e-6)
    np.testing.assert_allclose(ellipse_rotation(-np.pi / 2.) @ up, left, atol=1e-6)
