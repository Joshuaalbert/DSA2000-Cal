import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_cal.common.fourier_utils import ApertureTransform, find_optimal_fft_size
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model


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


def test_aperture_transform():
    a = ApertureTransform()

    image = jax.random.normal(jax.random.PRNGKey(0), shape=(128, 128), dtype=jnp.float32)
    dl = dm = 0.001
    dx = dy = 1 / (128 * 0.001)

    # Test with_shifts vs without_shifts
    f_ap = a._to_aperture_physical(image, axes=(-2, -1), dl=dl, dm=dm)
    f_ap_shifts = a._to_aperture_physical_with_shifts(image, axes=(-2, -1), dl=dl, dm=dm)
    np.testing.assert_allclose(f_ap, f_ap_shifts, atol=1e-3)

    f_image = a._to_image_physical(f_ap, axes=(-2, -1), dx=dx, dy=dy)
    f_image_shifts = a._to_image_physical_with_shifts(f_ap, axes=(-2, -1), dx=dx, dy=dy)
    np.testing.assert_allclose(f_image, f_image_shifts, atol=1e-3)

    f_ap = a.to_aperture(image, axes=(-2, -1), dl=dl, dm=dm)

    import pylab as plt

    plt.imshow(jnp.abs(f_ap), interpolation='nearest')
    plt.colorbar()
    plt.show()
    plt.imshow(np.angle(f_ap), interpolation='nearest')
    plt.colorbar()
    plt.show()

    f_image = a.to_image(f_ap, axes=(-2, -1), dx=dx, dy=dy)
    np.testing.assert_allclose(f_image, image, atol=1e-6)

    plt.imshow(jnp.abs(f_image), interpolation='nearest')
    plt.colorbar()
    plt.show()
    plt.imshow(np.imag(f_image), interpolation='nearest')
    plt.colorbar()
    plt.show()

    residual = f_image - image
    plt.imshow(jnp.abs(residual), interpolation='nearest')
    plt.colorbar()
    plt.show()
    plt.imshow(np.angle(residual), interpolation='nearest')
    plt.colorbar()
    plt.show()


def test_aperture_transform_2():
    a = ApertureTransform()

    n = 128

    image = jax.random.normal(jax.random.PRNGKey(0), shape=(n, n), dtype=jnp.float32)
    dl = dm = 0.001
    dx = dy = 1 / (128 * 0.001)

    # Test with_shifts vs without_shifts
    f_ap = a._to_aperture_physical(image, axes=(-2, -1), dl=dl, dm=dm)
    f_ap_shifts = a._to_aperture_physical_with_shifts(image, axes=(-2, -1), dl=dl, dm=dm)
    np.testing.assert_allclose(f_ap, f_ap_shifts, atol=1e-3)

    f_image = a._to_image_physical(f_ap, axes=(-2, -1), dx=dx, dy=dy)
    f_image_shifts = a._to_image_physical_with_shifts(f_ap, axes=(-2, -1), dx=dx, dy=dy)
    np.testing.assert_allclose(f_image, f_image_shifts, atol=1e-3)
