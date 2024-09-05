import pylab as plt
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.fourier_utils import ApertureTransform, find_optimal_fft_size
import numpy as np


@pytest.mark.parametrize('convention', ['physical', 'casa'])
def test_fourier_conventions(convention):
    dx = 0.1
    n = 100
    x = jnp.arange(-n, n + 1) * dx
    nu = jnp.fft.fftshift(jnp.fft.fftfreq(2 * n + 1, dx))
    dnu = nu[1] - nu[0]
    f_aperture = jnp.exp(-x ** 2) + x

    am = ApertureTransform(convention=convention)

    f_image = am.to_image(f_aperture, axes=(0,), dx=dx)
    plt.plot(nu,
             jnp.abs(f_image))  # This shows the gaussian shifted with peak split up! I expected it to be in the middle
    plt.title(convention + ': image')
    plt.show()

    rec_f_aperture = am.to_aperture(f_image, axes=(0,), dnu=dnu)
    # These agree and the gaussian is at the centre of both plots.
    plt.plot(x, jnp.abs(rec_f_aperture))
    plt.plot(x, jnp.abs(f_aperture))
    plt.title(convention + ': aperture')
    plt.show()

    # This passes for both conventions
    np.testing.assert_allclose(f_aperture, rec_f_aperture, atol=1e-4)

    # If we run with 'casa' convention, the plots all have mode in centre

    f_image = jnp.exp(-nu ** 2) + nu

    f_aperture = am.to_aperture(f_image, axes=(0,), dnu=dnu)
    plt.plot(nu,
             jnp.abs(
                 f_aperture))  # This shows the gaussian shifted with peak split up! I expected it to be in the middle
    plt.title(convention + ': aperture')
    plt.show()

    rec_f_image = am.to_image(f_aperture, axes=(0,), dx=dx)
    # These agree and the gaussian is at the centre of both plots.
    plt.plot(x, jnp.abs(rec_f_image))
    plt.plot(x, jnp.abs(f_image))
    plt.title(convention + ': image')
    plt.show()

    # This passes for both conventions
    np.testing.assert_allclose(f_image, rec_f_image, atol=1e-5)


def test_find_next_magic_size():
    for i in range(1, 100):
        assert find_optimal_fft_size(i, required_radix={4}) % 4 == 0

    import pylab as plt
    x = []
    y = []
    for i in range(1, 10000, 1):
        x.append(i)
        y.append(find_optimal_fft_size(i, required_radix={4}))
    plt.scatter(x, y)
    plt.show()
    print(sorted(set(y)))
