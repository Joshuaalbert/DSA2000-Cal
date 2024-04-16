import astropy.time as at
import astropy.units as au
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.interp_utils import optimized_interp_jax_safe, multilinear_interp_2d, \
    get_interp_indices_and_weights, _left_broadcast_multiply, convolved_interp, get_centred_insert_index


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


def test_get_interp_indices_and_weights():
    xp = [0, 1, 2, 3]
    x = 1.5
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 1
    assert alpha0 == 0.5
    assert i1 == 2
    assert alpha1 == 0.5

    x = 0
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1
    assert i1 == 1
    assert alpha1 == 0

    x = 3
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 2
    assert alpha0 == 0
    assert i1 == 3
    assert alpha1 == 1

    x = -1
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == -1

    x = 4
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == 4

    x = 5
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == 5

    xp = [0., 0.]
    x = 0.
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1
    assert i1 == 1
    assert alpha1 == 0.

    xp = [0., 0.]
    x = -1
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 2.
    assert i1 == 1
    assert alpha1 == -1.

    xp = [0.]
    x = 0.5
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1.
    assert i1 == 0
    assert alpha1 == 0.

    # Vector ops
    xp = [0, 1, 2, 3]
    x = [1.5, 1.5]
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    np.testing.assert_array_equal(i0, jnp.asarray([1, 1]))
    np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    np.testing.assert_array_equal(i1, jnp.asarray([2, 2]))
    np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))

    xp = [0, 1, 2, 3]
    x = [1.5, 2.5]
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    np.testing.assert_array_equal(i0, jnp.asarray([1, 2]))
    np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    np.testing.assert_array_equal(i1, jnp.asarray([2, 3]))
    np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))

    # xp = [0, 1, 2, 3]
    # x = [-0.5, 3.5]
    # (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp)
    # print(i0, alpha0, i1, alpha1)
    # np.testing.assert_array_equal(i0, jnp.asarray([1, 2]))
    # np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    # np.testing.assert_array_equal(i1, jnp.asarray([2, 3]))
    # np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))

def test_get_interp_indices_and_weights_astropy_time():
    # Test with at.Time objects
    xp = at.Time.now() + np.arange(10) * au.s
    x = xp
    x0 = x[0]
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights((x - x0).sec, (xp - x0).sec)
    assert np.allclose((xp - x0).sec, np.arange(10))
    print((i0, alpha0), (i1, alpha1))
    assert np.all(i0 == np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 8]))
    assert np.all(i1 == np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 9]))


def test__left_broadcast_multiply():
    assert np.all(_left_broadcast_multiply(np.ones((2, 3)), np.ones((2,))) == np.ones((2, 3)))
    assert np.all(_left_broadcast_multiply(np.ones((2,)), np.ones((2, 3))) == np.ones((2, 3)))
    assert np.all(_left_broadcast_multiply(np.ones((2, 3)), np.ones((2, 3))) == np.ones((2, 3)))


def test_convolved_interp():
    x = jnp.array([[0., 0.], [1., 1.]])
    y = jnp.array([[0., 0.], [1., 1.], [2., 2.]])
    z = jnp.array([0., 1., 2.])

    z_interp = convolved_interp(y, y, z, k=1, mode='euclidean')
    print(z_interp)
    np.testing.assert_array_equal(z_interp, z)

    z_interp = convolved_interp(x, y, z, k=1, mode='euclidean')
    print(z_interp)
    np.testing.assert_array_equal(z_interp, jnp.array([0., 1.]))

    x = jnp.array([[0.5, 0.5], [1.5, 1.5]])
    z_interp = convolved_interp(x, y, z, k=2, mode='euclidean')
    print(z_interp)
    np.testing.assert_allclose(z_interp, jnp.array([0.5, 1.5]), rtol=1e-6)


def test__get_centred_insert_index():
    time_centres = np.asarray([0.5, 1.5, 2.5])

    times_to_insert = np.asarray([0, 1, 2])
    expected_time_idx = np.asarray([0, 1, 2])
    time_idx = get_centred_insert_index(times_to_insert, time_centres)
    np.testing.assert_array_equal(time_idx, expected_time_idx)

    times_to_insert = np.asarray([1, 2, 3 - 1e-10])
    expected_time_idx = np.asarray([1, 2, 2])
    time_idx = get_centred_insert_index(times_to_insert, time_centres)
    np.testing.assert_array_equal(time_idx, expected_time_idx)

    with pytest.raises(ValueError):
        get_centred_insert_index(np.asarray([3]), time_centres)

    with pytest.raises(ValueError):
        get_centred_insert_index(np.asarray([0 - 1e-10]), time_centres)

    # Try with out of bounds
    times_to_insert = np.asarray([-10, 10])
    expected_time_idx = np.asarray([0, 2])
    time_idx = get_centred_insert_index(times_to_insert, time_centres, ignore_out_of_bounds=True)
    np.testing.assert_array_equal(time_idx, expected_time_idx)
