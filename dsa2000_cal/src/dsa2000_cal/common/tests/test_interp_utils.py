import astropy.time as at
import astropy.units as au
import jax.random
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.interp_utils import optimized_interp_jax_safe, multilinear_interp_2d, \
    get_interp_indices_and_weights, left_broadcast_multiply, convolved_interp, get_centred_insert_index, apply_interp, \
    InterpolatedArray


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


def test_multilinear_interp_2d():
    xp = jnp.asarray([0, 1, 2])
    yp = jnp.asarray([0, 1, 2])
    z = jnp.asarray([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    x, y = jnp.meshgrid(jnp.linspace(0., 1., 4),
                        jnp.linspace(0., 1., 4),
                        indexing='ij')

    expected = jnp.asarray(
        [
            [0., 0.3333333, 0.6666666, 1.],
            [1., 1.3333333, 1.6666666, 2.],
            [2., 2.3333333, 2.6666666, 3.],
            [3., 3.3333333, 3.6666666, 4.]
        ]
    )
    np.testing.assert_allclose(multilinear_interp_2d(x, y, xp, yp, z), expected, atol=1e-6)

    # within_bounds_2d
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

    # edge_cases_2d
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

    # out_of_bounds_2d
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
    xp = jnp.asarray([0, 1, 2, 3])
    x = 1.5
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 1
    assert alpha0 == 0.5
    assert i1 == 2
    assert alpha1 == 0.5

    x = 0
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1
    assert i1 == 1
    assert alpha1 == 0

    x = 3
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 2
    assert alpha0 == 0
    assert i1 == 3
    assert alpha1 == 1

    x = -1
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == -1

    x = 4
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == 4

    x = 5
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == 5

    xp = jnp.asarray([0., 0.])
    x = 0.

    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    with pytest.raises(AssertionError):
        assert i0 == 0
        assert alpha0 == 1
        assert i1 == 1
        assert alpha1 == 0.

    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp, check_spacing=True)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1
    assert i1 == 1
    assert alpha1 == 0.

    xp = jnp.asarray([0., 0.])
    x = -1
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    with pytest.raises(AssertionError):
        assert i0 == 0
        assert alpha0 == 2.
        assert i1 == 1
        assert alpha1 == -1.

    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp, check_spacing=True)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 2.
    assert i1 == 1
    assert alpha1 == -1.

    xp = jnp.asarray([0.])
    x = 0.5
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1.
    assert i1 == 0
    assert alpha1 == 0.

    # Vector ops
    xp = jnp.asarray([0, 1, 2, 3])
    x = jnp.asarray([1.5, 1.5])
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    np.testing.assert_array_equal(i0, jnp.asarray([1, 1]))
    np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    np.testing.assert_array_equal(i1, jnp.asarray([2, 2]))
    np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))

    xp = jnp.asarray([0, 1, 2, 3])
    x = jnp.asarray([1.5, 2.5])
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    print(i0, alpha0, i1, alpha1)
    np.testing.assert_array_equal(i0, jnp.asarray([1, 2]))
    np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    np.testing.assert_array_equal(i1, jnp.asarray([2, 3]))
    np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))

    # xp = [0, 1, 2, 3]
    # x = [-0.5, 3.5]
    # (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp)
    # print(i0, alpha0, i1, alpha1)
    # np.testing.assert_array_equal(i0, jnp.asarray([1, 2]))
    # np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    # np.testing.assert_array_equal(i1, jnp.asarray([2, 3]))
    # np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))


@pytest.mark.parametrize('regular_grid', [True, False])
def test_apply_interp(regular_grid):
    xp = jnp.linspace(0., 1., 10)
    x = jnp.linspace(-0.1, 1.1, 10)
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=regular_grid)
    np.testing.assert_allclose(apply_interp(xp, i0, alpha0, i1, alpha1), x, atol=1e-6)

    x = jnp.linspace(-0.1, 1.1, 15)
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=regular_grid)
    assert apply_interp(jnp.zeros((4, 5, 10, 6)), i0, alpha0, i1, alpha1, axis=2).shape == (4, 5, 15, 6)

    x = 0.
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=regular_grid)
    assert apply_interp(jnp.zeros((4, 5, 10, 6)), i0, alpha0, i1, alpha1, axis=2).shape == (4, 5, 6)

    print(
        jax.jit(lambda x: apply_interp(x, i0, alpha0, i1, alpha1, axis=2)).lower(
            jnp.zeros((4, 5, 10, 6))).compile().cost_analysis()
    )
    # [{'bytes accessed': 1440.0, 'utilization1{}': 2.0, 'bytes accessed0{}': 960.0, 'bytes accessedout{}': 480.0, 'bytes accessed1{}': 960.0}]
    # [{'bytes accessed1{}': 960.0,  'utilization1{}': 2.0, 'bytes accessedout{}': 480.0, 'bytes accessed0{}': 960.0, 'bytes accessed': 1440.0}]

    # Test with axis=-1
    x = jnp.linspace(-0.1, 1.1, 15)
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=regular_grid)
    assert apply_interp(jnp.zeros((4, 5, 6, 10)), i0, alpha0, i1, alpha1, axis=3).shape == (4, 5, 6, 15)
    assert apply_interp(jnp.zeros((4, 5, 6, 10)), i0, alpha0, i1, alpha1, axis=-1).shape == (4, 5, 6, 15)
    np.testing.assert_allclose(
        apply_interp(jnp.zeros((4, 5, 6, 10)), i0, alpha0, i1, alpha1, axis=-1),
        apply_interp(jnp.zeros((4, 5, 6, 10)), i0, alpha0, i1, alpha1, axis=3),
    )


def test_regular_grid():
    # Inside bounds
    xp = jnp.linspace(0., 1., 10)
    fp = jax.random.normal(jax.random.PRNGKey(0), (10, 15))
    x = jnp.linspace(0., 1., 100)
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=False)
    f_no = apply_interp(fp, i0, alpha0, i1, alpha1)

    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=True)
    f_yes = apply_interp(fp, i0, alpha0, i1, alpha1)
    np.testing.assert_allclose(
        f_yes, f_no,
        atol=1e-6
    )

    # Outside bounds
    x = jnp.linspace(-0.1, 1.1, 100)
    xp = jnp.linspace(0., 1., 10)
    fp = jax.random.normal(jax.random.PRNGKey(0), (10, 15))
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=False)
    f_no = apply_interp(fp, i0, alpha0, i1, alpha1)

    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=True)
    f_yes = apply_interp(fp, i0, alpha0, i1, alpha1)
    np.testing.assert_allclose(
        f_yes, f_no,
        atol=1e-6
    )


def test_get_interp_indices_and_weights_astropy_time():
    # Test with at.Time objects
    xp = at.Time.now() + np.arange(10) * au.s
    x = xp
    x0 = x[0]
    (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights((x - x0).sec, (xp - x0).sec)
    assert np.allclose((xp - x0).sec, np.arange(10))
    print((i0, alpha0, i1, alpha1))
    assert np.all(i0 == np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 8]))
    assert np.all(i1 == np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 9]))


def test_left_broadcast_multiply():
    assert np.all(left_broadcast_multiply(np.ones((2, 3)), np.ones((2,))) == np.ones((2, 3)))
    assert np.all(left_broadcast_multiply(np.ones((2, 3)), np.ones((2, 3))) == np.ones((2, 3)))
    assert np.all(
        left_broadcast_multiply(np.ones((1, 2, 3, 4, 5)), np.ones((3, 4)), axis=2) == np.ones((1, 2, 3, 4, 5)))
    assert np.all(
        left_broadcast_multiply(np.ones((1, 2, 3, 4, 5)), np.ones((3, 4)), axis=-3) == np.ones((1, 2, 3, 4, 5)))


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


@pytest.mark.parametrize('regular_grid', [True, False])
@pytest.mark.parametrize('normalise', [True, False])
def test_interpolated_array(regular_grid: bool, normalise: bool):
    # scalar time
    times = jnp.linspace(0, 10, 100)
    values = jnp.sin(times)
    interp = InterpolatedArray(times, values, regular_grid=regular_grid, normalise=normalise)
    assert interp(5.).shape == ()
    np.testing.assert_allclose(interp(5.), jnp.sin(5), atol=2e-3)

    # vector time
    assert interp(jnp.array([5., 6.])).shape == (2,)
    np.testing.assert_allclose(interp(jnp.array([5., 6.])), jnp.sin(jnp.array([5., 6.])), atol=2e-3)

    # Now with axis = 1
    times = jnp.linspace(0, 10, 100)
    values = jnp.stack([jnp.sin(times), jnp.cos(times)], axis=0)  # [2, 100]
    interp = InterpolatedArray(times, values, axis=1, regular_grid=regular_grid, normalise=normalise)
    assert interp(5.).shape == (2,)
    np.testing.assert_allclose(interp(5.), jnp.array([jnp.sin(5), jnp.cos(5)]), atol=2e-3)

    # Vector
    assert interp(jnp.array([5., 6., 7.])).shape == (2, 3)
    np.testing.assert_allclose(interp(jnp.array([5., 6., 7.])),
                               jnp.stack([jnp.sin(jnp.array([5., 6., 7.])), jnp.cos(jnp.array([5., 6., 7.]))],
                                         axis=0),
                               atol=2e-3)


def test_interpolated_array_normalise():
    # compare same with and without
    times = jnp.linspace(0, 10, 100)  # [100]
    values = jnp.sin(times)  # [100]
    interp = InterpolatedArray(times, values, normalise=False)
    interp_norm = InterpolatedArray(times, values, normalise=True)
    assert interp(5.).shape == ()
    assert interp_norm(5.).shape == ()
    np.testing.assert_allclose(interp(5.), interp_norm(5.), atol=1e-8)

    # Find case where normalisation is needed, i.e. different that tolerance
    times = jnp.linspace(0, 10, 100)  # [100]
    values = jnp.sin(times) * 1e30 + 0.00000001  # [100]
    interp = InterpolatedArray(times, values, normalise=False)
    interp_norm = InterpolatedArray(times, values, normalise=True)
    assert interp(5.).shape == ()
    assert interp_norm(5.).shape == ()
    print(interp(5.0001) - interp_norm(5.0001))
    # assert not np.allclose(interp(5.), interp_norm(5.), atol=1e-8)


def test_interoplated_array_getitem():
    x = np.linspace(0, 1, 10)
    shape = (10, 5, 6)
    values = np.random.rand(*shape)
    interp = InterpolatedArray(x, values, axis=0, auto_reorder=False)
    assert interp.shape == (5, 6)

    interp_slice = interp[0]
    assert interp_slice.shape == (6,)
    np.testing.assert_allclose(interp_slice.values, values[:, 0, :])

    # for axis = -1
    shape = (5, 6, 10)
    values = np.random.rand(*shape)
    interp = InterpolatedArray(x, values, axis=-1)
    assert interp.shape == (5, 6)

    interp_slice = interp[0]
    assert interp_slice.shape == (6,)
    np.testing.assert_allclose(interp_slice.values, values[0, :, :])

def test_interpolated_array_auto_reorder():
    x = np.linspace(0, 1, 10)
    values = np.random.rand(10, 5, 6)
    interp = InterpolatedArray(x, values, axis=0, auto_reorder=True)
    assert interp.shape == (5, 6)
    np.testing.assert_allclose(interp.values, np.moveaxis(values, 0, -1))
