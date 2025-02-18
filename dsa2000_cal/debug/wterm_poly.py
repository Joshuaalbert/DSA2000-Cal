from functools import partial

import astropy.constants as const
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_common.common.quantity_utils import quantity_to_jnp


def wterm(l, m, w):
    n = jnp.square(1 - jnp.square(l) - jnp.square(m))
    return jnp.exp(-2j * jnp.pi * w * n) / n


def break_up_ffts():
    import numpy as np
    import matplotlib.pyplot as plt

    # Function to shift the FFT of a sub-image
    def shift_fft(sub_fft, shift):
        n, m = sub_fft.shape
        u = np.fft.fftfreq(n).reshape(-1, 1)
        v = np.fft.fftfreq(m).reshape(1, -1)
        shift_matrix = np.exp(-2j * np.pi * (shift[0] * u + shift[1] * v))
        return sub_fft * shift_matrix

    # Create a sample image of size [2n, 2n]
    n = 4
    full_image = np.random.random((2 * n, 2 * n))

    # Divide the image into four sub-images
    I1 = full_image[0:n, 0:n]
    I2 = full_image[0:n, n:2 * n]
    I3 = full_image[n:2 * n, 0:n]
    I4 = full_image[n:2 * n, n:2 * n]

    # Compute the FFT of each sub-image
    F1 = np.fft.fft2(I1)
    F2 = np.fft.fft2(I2)
    F3 = np.fft.fft2(I3)
    F4 = np.fft.fft2(I4)

    # Adjust the FFTs with shifts
    F2_shifted = shift_fft(F2, (0, n))
    F3_shifted = shift_fft(F3, (n, 0))
    F4_shifted = shift_fft(F4, (n, n))

    # Combine the FFTs into a full FFT array
    combined_fft = np.zeros((2 * n, 2 * n), dtype=complex)
    combined_fft[0:n, 0:n] = F1
    combined_fft[0:n, n:2 * n] = F2_shifted
    combined_fft[n:2 * n, 0:n] = F3_shifted
    combined_fft[n:2 * n, n:2 * n] = F4_shifted

    # Compute the FFT of the full image
    full_fft = np.fft.fft2(full_image)

    # Compare the results
    print("Are the two FFTs identical? ", np.allclose(full_fft, combined_fft))

    # For visualization purposes
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("FFT of Full Image")
    plt.imshow(np.log(np.abs(np.fft.fftshift(full_fft)) + 1), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Combined FFT of Sub-Images")
    plt.imshow(np.log(np.abs(np.fft.fftshift(combined_fft)) + 1), cmap='gray')

    plt.show()


# def unwrapped_wkernel(l, w):
#     n = jnp.sqrt(1 - l ** 2)
#     return jnp.exp(-2j * jnp.pi * w * (n - 1.) / 0.5) / n

def test_w_abs_approx():
    def approx(l):
        return 1 + l ** 2 / 2 + (3 * l ** 4) / 8 + (5 * l ** 6) / 16

    def f(l):
        return 1 / jnp.sqrt(1 - l ** 2)

    lvec = jnp.linspace(0., 0.999, 1000)

    import pylab as plt

    plt.plot(lvec, jnp.log10(f(lvec) - approx(lvec)))
    plt.axvline(jnp.interp(0.001, f(lvec) - approx(lvec), lvec), color='r', label='0.1% error')
    plt.title("Approximation error 6th order")
    plt.xlabel('l')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()

    def approx(l):
        return 1 + l ** 2 / 2 + (3 * l ** 4) / 8 + (5 * l ** 6) / 16 + (35 * l ** 8) / 128 + (63 * l ** 10) / 256

    def f(l):
        return 1 / jnp.sqrt(1 - l ** 2)

    lvec = jnp.linspace(0., 0.999, 1000)

    import pylab as plt

    plt.plot(lvec, jnp.log10(f(lvec) - approx(lvec)))
    plt.axvline(jnp.interp(0.001, f(lvec) - approx(lvec), lvec), color='r', label='0.1% error')
    plt.title("Approximation error 10th order")
    plt.xlabel('l')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()


def test_w_angle_approximation():
    def w_angle(l, w):
        n = jnp.sqrt(1 - l ** 2)
        return -2 * jnp.pi * w * (n - 1.)

    def approx(l, w):
        # l ^ 2 \pi  w + (l ^ 4 \pi w) / 4 + (l ^ 6 \pi w) / 8 + (5 l ^ 8 \pi w) / 64 + (7 l ^ {10} \pi w) / 128
        return l * jnp.pi * w + (l ** 3 * jnp.pi * w) / 4 + (l ** 5 * jnp.pi * w) / 8 + (
                5 * l ** 7 * jnp.pi * w) / 64 + (
                7 * l ** 9 * jnp.pi * w) / 128

    lvec = jnp.linspace(0., 0.999, 1000)
    w = 1e3

    import pylab as plt

    plt.plot(lvec, jnp.abs(w_angle(lvec, w) - approx(lvec, w)))

    plt.title("Approximation error 10th order")
    plt.xlabel('l')
    plt.ylabel('log10(error)')
    plt.show()


def test_wkernel_variation():
    freq = 70e6 * au.Hz
    wavelength = quantity_to_jnp(const.c / freq)
    wvec = jnp.linspace(0., 10e3, 10000)
    lvec = jnp.linspace(0, 0.999, 1000)

    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def wkernel(l, w):
        n = jnp.sqrt(1 - l ** 2)
        return jnp.exp(-2j * jnp.pi * w * (n - 1.) / wavelength) / n

    wterm = wkernel(lvec, wvec)  # [nl, nw]

    import pylab as plt

    wterm_angle = jnp.angle(wterm)  # [nl, nw]

    wterm_angle_unwrapped = jnp.unwrap(wterm_angle, axis=0)
    print(jnp.sum(jnp.isnan(wterm_angle_unwrapped)))
    print(wterm_angle_unwrapped)

    np.testing.assert_allclose(jnp.angle(jnp.exp(1j * wterm_angle_unwrapped)), wterm_angle, atol=1e-6)

    diff_unwrapped = jnp.diff(wterm_angle_unwrapped, axis=0)

    plt.imshow(diff_unwrapped.T,
               origin='lower',
               extent=(lvec[0], lvec[-1], wvec[0], wvec[-1]),
               interpolation='nearest',
               aspect='auto',
               cmap='hsv')
    plt.colorbar()
    plt.xlabel('l [proj. rad]')
    plt.ylabel('w [m]')
    plt.show()

    plt.imshow(wterm_angle.T,
               origin='lower',
               extent=(lvec[0], lvec[-1], wvec[0], wvec[-1]),
               interpolation='nearest',
               aspect='auto',
               cmap='hsv')
    plt.colorbar()
    plt.xlabel('l [proj. rad]')
    plt.ylabel('w [m]')
    plt.title(r"${\rm{Arg}W(l,m)}$")
    plt.show()

    plt.imshow(wterm_angle_unwrapped.T,
               origin='lower',
               extent=(lvec[0], lvec[-1], wvec[0], wvec[-1]),
               interpolation='nearest',
               aspect='auto',
               cmap='jet')
    plt.colorbar()
    plt.xlabel('l [proj. rad]')
    plt.ylabel('w [m]')
    plt.title(r"Unwrapped ${\rm{Arg}W(l,m)}$")
    plt.show()

    plt.imshow(jnp.log(jnp.abs(wterm).T),
               origin='lower',
               extent=(lvec[0], lvec[-1], wvec[0], wvec[-1]),
               interpolation='nearest',
               aspect='auto',
               cmap='inferno')
    plt.colorbar()
    plt.xlabel('l [proj. rad]')
    plt.ylabel('w [m]')
    plt.show()


if __name__ == '__main__':
    break_up_ffts()
