import numpy as np
from scipy.optimize import least_squares
from scipy.special import eval_genlaguerre


# Define Zernike Polynomials
def zernike_polynomial(n, m, theta, phi):
    """
    Calculate the Zernike polynomial for given n, m, theta, and phi.
    """

    def R(n, m, r):
        """
        Calculate the radial polynomial component of the Zernike polynomial.
        """
        radial_poly = eval_genlaguerre((n - abs(m)) // 2, abs(m), r ** 2)
        return radial_poly

    r = np.sin(theta)
    if m >= 0:
        f_abs = R(n, m, r) * np.cos(m * phi)
    else:
        f_abs = R(n, m, r) * np.sin(m * phi)

    return f_abs


# Function to fit Zernike polynomials to data
def fit_zernike_polynomials(theta, phi, f, n_max):
    """
    Fit Zernike polynomials to a set of (theta, phi, f(theta, phi)) data points.

    Parameters:
    theta (np.ndarray): Array of theta values.
    phi (np.ndarray): Array of phi values.
    f (np.ndarray): Array of f(theta, phi) values.
    n_max (int): Maximum order of Zernike polynomials to fit.

    Returns:
    coefficients (np.ndarray): Array of fitted Zernike polynomial coefficients.
    """
    # Generate Zernike polynomials up to n_max
    zernike_polys = []
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            if (n - abs(m)) % 2 == 0:
                zernike_polys.append(zernike_polynomial(n, m, theta, phi))

    zernike_polys = np.array(zernike_polys)

    # Define the objective function for least-squares fitting
    def objective(coeffs):
        zernike_fit = np.sum(coeffs[:, None] * zernike_polys, axis=0)
        return np.concatenate([np.real(zernike_fit - f), np.imag(zernike_fit - f)])

    # Initial guess for coefficients
    initial_guess = np.zeros(zernike_polys.shape[0], dtype=float)

    # Perform least-squares fitting
    result = least_squares(objective, initial_guess.view(np.float64))

    # Extract fitted coefficients
    fitted_coeffs = result.x

    screen = np.sum(fitted_coeffs[:, None] * zernike_polys, axis=0)

    return fitted_coeffs, screen


if __name__ == '__main__':
    # Example usage
    theta = np.random.uniform(0, np.pi, 1000)
    phi = np.random.uniform(0, 2 * np.pi, 1000)
    f = np.sin(theta)**2 * np.exp(phi * 1j)

    n_max = 15
    coefficients, screen = fit_zernike_polynomials(theta, phi, f, n_max)
    print("Fitted coefficients:", coefficients)

    import pylab as plt

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    plt.scatter(x, y, c=np.abs(f))
    plt.show()
    plt.scatter(x, y, c=np.abs(screen))
    plt.show()

    plt.scatter(x, y, c=np.angle(f))
    plt.show()
    plt.scatter(x, y, c=np.angle(screen))
    plt.show()
