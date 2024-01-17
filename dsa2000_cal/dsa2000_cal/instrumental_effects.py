import numpy as np
import matplotlib.pyplot as plt

# Example aperture distribution function phi(x, y)
def aperture_distribution(x, y):
    # Define your function here. For example:
    return np.exp(-((x**2 + y**2) / 2))

# Define the grid
x = np.linspace(-10, 10, 256)
y = np.linspace(-10, 10, 256)
X, Y = np.meshgrid(x, y)

# Compute the aperture distribution
aperture = aperture_distribution(X, Y)

def compute_far_field(aperture_distribution, x_range, y_range):
    """
    Compute the far-field pattern of a radio dish given the aperture plane distribution.

    Parameters:
    aperture_distribution (2D numpy array): The aperture distribution in the x-y plane.
    x_range (tuple): The range of x coordinates (min_x, max_x) for the aperture distribution.
    y_range (tuple): The range of y coordinates (min_y, max_y) for the aperture distribution.

    Returns:
    tuple: A tuple containing:
        - 2D numpy array of the far-field pattern.
        - 1D numpy arrays for the l and m coordinates.
    """
    # Create the grid for the aperture distribution
    x = np.linspace(x_range[0], x_range[1], aperture_distribution.shape[1])
    y = np.linspace(y_range[0], y_range[1], aperture_distribution.shape[0])

    dx = (x_range[1] - x_range[0]) / (aperture_distribution.shape[1] - 1)
    dy = (y_range[1] - y_range[0]) / (aperture_distribution.shape[0] - 1)

    # Apply FFT to the aperture distribution
    fft_result = np.fft.fftshift(np.fft.fft2(aperture_distribution))

    # Calculate spatial frequencies (l, m) corresponding to the far-field
    l = np.fft.fftshift(np.fft.fftfreq(fft_result.shape[0], d=(x[1] - x[0])))
    m = np.fft.fftshift(np.fft.fftfreq(fft_result.shape[1], d=(y[1] - y[0])))

    return fft_result, l, m

def test_compute_far_field():
    # Apply FFT to the aperture distribution
    fft_result = np.fft.fftshift(np.fft.fft2(aperture))

    # Calculate spatial frequencies (l, m) corresponding to the far-field
    l = np.fft.fftshift(np.fft.fftfreq(fft_result.shape[0], d=dx))
    m = np.fft.fftshift(np.fft.fftfreq(fft_result.shape[1], d=dy))

    # Plot the results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(aperture), extent=(x.min(), x.max(), y.min(), y.max()))
    plt.title('Aperture Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(fft_result), extent=(l.min(), l.max(), m.min(), m.max()))
    plt.title('Far Field Pattern')
    plt.xlabel('l')
    plt.ylabel('m')
    plt.colorbar()
    plt.show()
