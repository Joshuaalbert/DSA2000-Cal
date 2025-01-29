import numpy as np
import pylab as plt
import pytest

from dsa2000_cal.common.fourier_utils import ApertureTransform, find_optimal_fft_size


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


@pytest.mark.parametrize('convention', ['physical', 'engineering'])
def test_aperture_transform(convention):
    a = ApertureTransform(convention=convention)

    n = 128
    dl = dm = 0.001
    mvec = lvec = (-n * 0.5 + np.arange(n)) * dl
    L, M = np.meshgrid(lvec, mvec, indexing='ij')
    image = np.exp(-0.5 * (L ** 2 + M ** 2) / 0.01 ** 2 + 1j * (L + M) / 0.01).astype(np.float32)
    plt.imshow(
        np.abs(image).T, interpolation='nearest',
        origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1])
    )
    plt.show()

    plt.imshow(
        np.angle(image).T, interpolation='nearest',
        origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1])
    )
    plt.show()

    aperture = a.to_aperture(image, axes=(-2, -1), dl=dl, dm=dm)
    dx = dy = 1 / (n * dl)
    xvec = yvec = (-n * 0.5 + np.arange(n)) * dx
    X, Y = np.meshgrid(xvec, yvec, indexing='ij')
    plt.imshow(
        np.abs(aperture).T, interpolation='nearest',
        origin='lower', extent=(xvec[0], xvec[-1], yvec[0], yvec[-1])
    )
    plt.show()
    plt.imshow(
        np.angle(aperture).T, interpolation='nearest',
        origin='lower', extent=(xvec[0], xvec[-1], yvec[0], yvec[-1])
    )
    plt.show()

    image_recon = a.to_image(aperture, axes=(-2, -1), dx=dx, dy=dy)
    plt.imshow(
        np.abs(image_recon).T, interpolation='nearest',
        origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1])
    )
    plt.show()
    plt.imshow(
        np.angle(image_recon).T, interpolation='nearest',
        origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1])
    )
    plt.show()
    np.testing.assert_allclose(image_recon.real, image.real, atol=1e-6)
    np.testing.assert_allclose(image_recon.imag, image.imag, atol=1e-6)
