import numpy as np

def test_complex_fourier_transform():
    # Parameters
    N = 2048  # Number of samples
    Delta_x = 0.01  # Spatial sampling interval (meters)
    Delta_k = 1 / (N * Delta_x)  # Frequency sampling interval (1/m)
    x_min = -N * Delta_x / 2  # Minimum x value
    k_min = -N * Delta_k / 2  # Minimum k value

    # Spatial and frequency grids
    x = x_min + np.arange(N) * Delta_x
    k = k_min + np.arange(N) * Delta_k

    # Function parameters
    sigma = 0.1  # Width of the Gaussian (meters)
    x0 = 0.5  # Shift in x-space (meters)
    k0 = 50.0  # Modulation in k-space (1/m)

    # Define the complex function in x-space
    f_x = np.exp(- (x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * 2 * np.pi * k0 * x)

    # Analytical Fourier Transform in k-space
    F_k_analytical = sigma * np.sqrt(2 * np.pi) * \
                     np.exp(- (sigma ** 2) * (2 * np.pi * (k - k0)) ** 2 / 2) * \
                     np.exp(-1j * 2 * np.pi * (k - k0) * x0)

    # Perform FFT with proper shifting and scaling
    # Shift f_x for FFT input
    f_x_shifted = np.fft.ifftshift(f_x)

    # Compute FFT
    F_k_unshifted = np.fft.fft(f_x_shifted)

    # Shift FFT output
    F_k_shifted = np.fft.fftshift(F_k_unshifted)

    # Scale the FFT output
    F_k_scaled = F_k_shifted * Delta_x

    # Phase correction due to grid offset
    phase_correction = np.exp(-1j * 2 * np.pi * x_min * k)
    F_k_corrected = F_k_scaled * phase_correction

    # Compute the error between numerical and analytical results
    amplitude_error = np.abs(F_k_corrected) - np.abs(F_k_analytical)
    phase_error = np.angle(F_k_corrected) - np.angle(F_k_analytical)

    # Normalize errors
    amplitude_error_norm = np.max(np.abs(amplitude_error)) / np.max(np.abs(F_k_analytical))
    phase_error_norm = np.max(np.abs(phase_error))

    # Define acceptable tolerance
    amplitude_tolerance = 1e-6
    phase_tolerance = 1e-6  # Radians

    # Unit test assertions
    # np.testing.assert_allclose(amplitude_error_norm, 0, atol=amplitude_tolerance)
    # np.testing.assert_allclose(phase_error_norm, 0, atol=phase_tolerance)

    # For visual inspection (optional)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(k, np.abs(F_k_corrected), label='Numerical')
    plt.plot(k, np.abs(F_k_analytical), '--', label='Analytical')
    plt.title('Amplitude of Fourier Transform')
    plt.xlabel('k (rad/m)')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(k, np.angle(F_k_corrected), label='Numerical')
    plt.plot(k, np.angle(F_k_analytical), '--', label='Analytical')
    plt.title('Phase of Fourier Transform')
    plt.xlabel('k (rad/m)')
    plt.legend()
    plt.tight_layout()
    plt.show()

