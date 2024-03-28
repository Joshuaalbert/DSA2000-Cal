import numpy as np
from jax import numpy as jnp

from dsa2000_cal.common.uniform_interp import optimized_interp_jax_safe, multilinear_interp_2d


def test_linear_interpolation():
    xp = jnp.linspace(0, 10, 11)
    yp = jnp.linspace(0, 10, 11)
    x = jnp.array([0.5, 1.5, 2.5, 7.5])

    expected = jnp.array([0.5, 1.5, 2.5, 7.5])
    np.testing.assert_allclose(optimized_interp_jax_safe(x, xp, yp), expected, rtol=1e-6)


def test_outside_bounds():
    xp = jnp.linspace(0, 10, 11)
    yp = jnp.linspace(0, 10, 11)
    x = jnp.array([-1, 11])

    expected = jnp.array([0, 10])  # Assuming linear extrapolation is not performed
    np.testing.assert_allclose(optimized_interp_jax_safe(x, xp, yp), expected, rtol=1e-6)


def test_uniform_spacing():
    xp = jnp.linspace(0, 10, 100)
    yp = jnp.sin(xp)
    x = jnp.array([0.1, 5.5, 9.9])

    # Use the sin function for expected values
    expected = jnp.sin(x)
    np.testing.assert_allclose(optimized_interp_jax_safe(x, xp, yp), expected, rtol=1e-2)


def test_within_bounds_2d():
    xp = jnp.linspace(0, 10, 11)
    yp = jnp.linspace(0, 10, 11)
    x, y = jnp.meshgrid(xp, yp, indexing='ij')
    z = jnp.sin(x) * jnp.cos(y)

    # Points for testing within bounds
    x_eval = jnp.array([1, 3, 5])
    y_eval = jnp.array([1, 3, 5])

    # Expected results computed manually or using a known good method
    expected = jnp.sin(x_eval) * jnp.cos(y_eval)

    # Perform interpolation and compare
    z_eval = multilinear_interp_2d(x_eval, y_eval, xp, yp, z)
    np.testing.assert_allclose(z_eval, expected, rtol=1e-5)


def test_edge_cases_2d():
    xp = jnp.linspace(0, 10, 11)
    yp = jnp.linspace(0, 10, 11)
    x, y = jnp.meshgrid(xp, yp, indexing='ij')
    z = jnp.sin(x) * jnp.cos(y)

    # Edge cases (literally, on the edges of the domain)
    x_eval = jnp.array([0, 10, 0])
    y_eval = jnp.array([0, 10, 10])

    # Expected results at the edges
    expected = jnp.sin(x_eval) * jnp.cos(y_eval)

    # Perform interpolation and compare
    z_eval = multilinear_interp_2d(x_eval, y_eval, xp, yp, z)
    np.testing.assert_allclose(z_eval, expected, rtol=1e-5)


def test_out_of_bounds_2d():
    xp = jnp.linspace(0, 10, 11)
    yp = jnp.linspace(0, 10, 11)
    x, y = jnp.meshgrid(xp, yp, indexing='ij')
    z = jnp.sin(x) * jnp.cos(y)

    # Out-of-bounds coordinates
    x_eval = jnp.array([-1, 11, 0])
    y_eval = jnp.array([-1, 11, 15])

    # Expected results should match the closest edge
    expected = jnp.sin(jnp.clip(x_eval, xp[0], xp[-1])) * jnp.cos(jnp.clip(y_eval, yp[0], yp[-1]))

    # Perform interpolation and compare
    z_eval = multilinear_interp_2d(x_eval, y_eval, xp, yp, z)
    np.testing.assert_allclose(z_eval, expected, rtol=1e-5)
