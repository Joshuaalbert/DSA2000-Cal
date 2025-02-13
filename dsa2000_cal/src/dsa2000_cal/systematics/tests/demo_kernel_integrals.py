import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from dsa2000_cal.systematics.ionosphere import GaussianLineIntegral, IonosphereLayerIntegral


def gaussian_numerical_line_integral(t1, t2, a, u, mu, Sigma):
    """Numerical integration of line integral through 3D Gaussian"""
    Sigma_inv = np.linalg.inv(Sigma)

    # norm_factor = 1 / ((2 * np.pi) ** 1.5 * np.sqrt(jnp.linalg.det(Sigma)))

    def integrand(t):
        x = a + t * u
        dx = x - mu
        return np.exp(-0.5 * dx.T @ Sigma_inv @ dx)

    result, _ = quad(integrand, t1, t2, epsabs=1e-9, epsrel=1e-9)
    return result


def numerical_double_integral(x1, x2, s1_hat, s2_hat, s1m, s1p, s2m, s2p, resolution: int, a, u, mu, Sigma):
    Sigma_inv = jnp.linalg.inv(Sigma)

    def gaussian(x):
        dx = x
        return jnp.exp(-0.5 * dx.T @ Sigma_inv @ dx)

    def integrand(s1, s2):
        x = x1 + s1 * s1_hat - x2 - s2 * s2_hat
        return gaussian(x)

    s1_array = jnp.linspace(s1m, s1p, resolution + 1)
    s2_array = jnp.linspace(s2m, s2p, resolution + 1)
    ds1 = (s1p - s1m) / resolution
    ds2 = (s2p - s2m) / resolution
    integral = jnp.sum(jax.vmap(lambda s1: jax.vmap(lambda s2: integrand(s1, s2))(s2_array))(s1_array)) * ds1 * ds2
    return integral

def compare_gaussian_line_integral():
    # Test parameters
    mu = np.array([1.0, -0.5, 2.0])
    Sigma = np.array([[2.0, 0.5, 0.9],
                      [0.5, 1.5, -0.3],
                      [0.9, -0.3, 3.0]])
    a = np.array([0.5, 1.0, -1.0])
    u = np.array([1.0, 0.5, -0.5])

    # Finite integral test
    t1, t2 = -1.0, 1.0
    closed = GaussianLineIntegral(mu, Sigma).finite_integral(a, u, t1, t2)
    numerical = gaussian_numerical_line_integral(t1, t2, a, u, mu, Sigma)

    print(f"Finite Integral Results ({t1} to {t2}):")
    print(f"Closed-form: {closed:.9f}")
    print(f"Numerical:   {numerical:.9f}")
    print(f"Difference:  {np.abs(closed - numerical):.2e}\n")

    # Infinite integral test (using large bounds)
    t_inf = 1e3  # Practical approximation of infinity
    closed_inf = GaussianLineIntegral(mu, Sigma).infinite_integral(a, u)
    numerical_inf = gaussian_numerical_line_integral(-t_inf, t_inf, a, u, mu, Sigma)

    print("Infinite Integral Results:")
    print(f"Closed-form: {closed_inf:.9f}")
    print(f"Numerical:   {numerical_inf:.9f}")
    print(f"Difference:  {np.abs(closed_inf - numerical_inf):.2e}")

    # test double_tomographic_integral(self, x1, x2, s1_hat, s2_hat, s1m, s1p, s2m, s2p, resolution: int)

    x1 = np.array([0, 0, 0])
    x2 = np.array([1, 0, 0])
    s1_hat = np.array([0, -0.01, 0.99])
    s2_hat = np.array([0.001, 0.01, 0.99])
    s1m = 200
    s1p = 300
    s2m = 200
    s2p = 300
    resolution = 20
    integral = GaussianLineIntegral(None, Sigma).double_tomographic_integral(x1, x2, s1_hat, s2_hat, s1m, s1p, s2m, s2p,
                                                                             resolution)
    numerical_integral = numerical_double_integral(x1, x2, s1_hat, s2_hat, s1m, s1p, s2m, s2p, resolution, a, u, mu,
                                                   Sigma)
    print(f"Double Tomographic Integral: {integral:.9f}"
          f"\nNumerical Double Tomographic Integral: {numerical_integral:.9f}"
          f"\nDifference: {np.abs(integral - numerical_integral):.2e}")

def main():
    ionosphere_integrator = IonosphereLayerIntegral(length_scale=1.0)
    ionosphere_integrator.calibrate_resolution(max_sep=20., max_angle=15/57, bottom=100., width=100.)
    ionosphere_integrator.calibrate_resolution(max_sep=20., max_angle=15/57, bottom=200., width=100.)
    ionosphere_integrator.calibrate_resolution(max_sep=20., max_angle=15/57, bottom=400., width=100.)

    ionosphere_integrator = IonosphereLayerIntegral(length_scale=5.0)
    ionosphere_integrator.calibrate_resolution(max_sep=20., max_angle=15 / 57, bottom=100., width=100.)
    ionosphere_integrator.calibrate_resolution(max_sep=20., max_angle=15 / 57, bottom=200., width=100.)
    ionosphere_integrator.calibrate_resolution(max_sep=20., max_angle=15 / 57, bottom=400., width=100.)



if __name__ == '__main__':
    main()
