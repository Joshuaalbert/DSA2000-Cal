import jax.numpy as jnp
import numpy
import pylab as plt

def wterm(l,m,w):
    n = jnp.square(1 - jnp.square(l) - jnp.square(m))
    return jnp.exp(-2j * jnp.pi * w * n)/n

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


if __name__ == '__main__':
    break_up_ffts()