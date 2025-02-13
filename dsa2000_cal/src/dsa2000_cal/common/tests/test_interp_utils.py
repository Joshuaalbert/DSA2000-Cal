import timeit

import astropy.time as at
import astropy.units as au
import jax
import jax.random
import numpy as np
import pylab as plt
import pytest
from jax import numpy as jnp

from dsa2000_cal.common.interp_utils import optimized_interp_jax_safe, multilinear_interp_2d, \
    get_interp_indices_and_weights, left_broadcast_multiply, convolved_interp, get_centred_insert_index, apply_interp, \
    InterpolatedArray, select_interpolation_points


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
@pytest.mark.parametrize('value_shape', [(10,), (10, 20), (1,)])
@pytest.mark.parametrize('axis', [0, -1])
@pytest.mark.parametrize('auto_reorder', [True, False])
@pytest.mark.parametrize('value_dtype', [np.float32, np.int32])
def test_interpolated_array(regular_grid: bool, normalise: bool, value_shape: tuple, axis: int, auto_reorder: bool,
                            value_dtype):
    print(f"Testing with regular_grid={regular_grid}, normalise={normalise}, value_shape={value_shape}, axis={axis}, "
          f"auto_reorder={auto_reorder}")
    # scalar time
    xp = jnp.linspace(0, 10, value_shape[axis])
    values = jnp.arange(np.prod(value_shape)).astype(value_dtype).reshape(value_shape)

    interp = InterpolatedArray(
        xp, values, regular_grid=regular_grid, normalise=normalise, axis=axis, auto_reorder=auto_reorder
    )

    if axis == 0:
        assert interp.shape == value_shape[1:]
    elif axis == -1:
        assert interp.shape == value_shape[:-1]
    else:
        raise ValueError('Invalid axis')

    x = 0.
    if axis == 0:
        y_expected = values[0, ...]
    elif axis == -1:
        y_expected = values[..., 0]
    else:
        raise ValueError('Invalid axis')
    assert interp(x).shape == interp.shape
    np.testing.assert_allclose(interp(x), y_expected, atol=1e-6)

    x = jnp.asarray([xp[0], xp[-1]])
    if axis == 0:
        y_expected = values[jnp.asarray([0, -1]), ...]
    elif axis == -1:
        y_expected = jnp.moveaxis(values[..., jnp.asarray([0, -1])], -1, 0)
    else:
        raise ValueError('Invalid axis')

    assert interp(x).shape == (2,) + interp.shape
    np.testing.assert_allclose(interp(x), y_expected, atol=1e-6)

    assert interp[None].shape == (1,) + interp.shape
    assert interp[..., None].shape == interp.shape + (1,)
    if interp.shape != ():
        assert interp[0].shape == interp.shape[1:]
        assert interp[..., 0].shape == interp.shape[:-1]
        assert interp[jnp.asarray([0, 0])].shape == (2,) + interp.shape[1:]
        assert interp[..., jnp.asarray([0, 0])].shape == interp.shape[:-1] + (2,)

    if auto_reorder:
        np.testing.assert_allclose(interp.values, jnp.moveaxis(values, axis, -1))

    leaves, treedef = jax.tree.flatten(interp)
    pytree = jax.tree.unflatten(treedef, leaves)
    np.testing.assert_allclose(pytree.values, interp.values)

    @jax.jit
    def f(ia: InterpolatedArray):
        return ia(ia.x)

    ia = InterpolatedArray(np.linspace(0, 1, 10), np.random.rand(10, 3), axis=0, regular_grid=regular_grid,
                           normalise=normalise, auto_reorder=auto_reorder)
    f_jit = f.lower(ia).compile()
    np.testing.assert_allclose(f_jit(ia), ia(ia.x))

    # test pytree values
    tuple_values = (values, values)

    interp = InterpolatedArray(
        xp, tuple_values, regular_grid=regular_grid, normalise=normalise, axis=axis, auto_reorder=auto_reorder
    )

    if axis == 0:
        assert interp.shape == (value_shape[1:], value_shape[1:])
    elif axis == -1:
        assert interp.shape == (value_shape[:-1], value_shape[:-1])
    else:
        raise ValueError('Invalid axis')

    x = 0.
    if axis == 0:
        y_expected = (values[0, ...], values[0, ...])
    elif axis == -1:
        y_expected = (values[..., 0], values[..., 0])
    else:
        raise ValueError('Invalid axis')
    np.testing.assert_allclose(interp(x), y_expected, atol=1e-6)


def test_interpolated_array_dunders():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    z = np.cos(x)
    ia1 = InterpolatedArray(x, y)
    ia2 = InterpolatedArray(x, z)
    # Test addition, subtraction, multiplication, and division on InterpolatedArray
    ia3 = ia1 + ia2
    np.testing.assert_allclose(ia3(x), y + z)
    ia3 = ia1 - ia2
    np.testing.assert_allclose(ia3(x), y - z)
    ia3 = ia1 * ia2
    np.testing.assert_allclose(ia3(x), y * z)
    ia3 = ia1 / ia2
    np.testing.assert_allclose(ia3(x), y / z)
    # Test addition, subtraction, multiplication, and division on InterpolatedArray with scalar
    ia3 = ia1 + 1
    np.testing.assert_allclose(ia3(x), y + 1)
    ia3 = ia1 - 1
    np.testing.assert_allclose(ia3(x), y - 1)
    ia3 = ia1 * 2
    np.testing.assert_allclose(ia3(x), y * 2)
    ia3 = ia1 / 2
    np.testing.assert_allclose(ia3(x), y / 2)


def test_compare_interpolation_methods():
    # They give the same performance, so choose eaiest to read
    batch_sizes, shape_sizes, num_runs = [1024], [32, 64], 5

    # Store the timing results
    option1_times = []
    option2_times = []
    option1_minus_option2_times = []

    for batch_size in batch_sizes:
        for shape_size in shape_sizes:
            # Generate random data
            shape = (shape_size, shape_size, shape_size)  # Example 3D shape
            x = jnp.array(np.random.rand(*shape))

            # Generate indices and interpolation weights
            i0 = jnp.array(np.random.randint(0, shape[-1] - 1, size=(batch_size,)))
            i1 = i0 + 1  # Ensure i1 is valid
            alpha0 = jnp.array(np.random.rand(batch_size))
            alpha1 = 1 - alpha0  # Ensure weights sum to 1

            # Option 1: Interpolation along last axis, then moveaxis
            def option1(x, i0, i1, alpha0, alpha1):
                result = jnp.moveaxis(alpha0 * x[..., i0] + alpha1 * x[..., i1], -1, 0)
                return result

            # Option 2: Use jax.vmap to vectorize interpolation
            def option2(x, i0, i1, alpha0, alpha1):
                interpolate_single = lambda i0, i1, alpha0, alpha1: alpha0 * x[..., i0] + alpha1 * x[..., i1]
                interpolate_vectorized = jax.vmap(interpolate_single)
                result = interpolate_vectorized(i0, i1, alpha0, alpha1)
                return result

            # JIT compile the functions
            option1_compiled = jax.jit(option1)
            option2_compiled = jax.jit(option2)

            # Warm-up runs
            option1_compiled(x, i0, i1, alpha0, alpha1).block_until_ready()
            option2_compiled(x, i0, i1, alpha0, alpha1).block_until_ready()

            # Measure execution time for Option 1
            time1 = timeit.timeit(lambda: option1_compiled(x, i0, i1, alpha0, alpha1).block_until_ready(),
                                  number=num_runs)
            option1_avg_time = time1 / num_runs
            option1_times.append((batch_size, shape_size, option1_avg_time))

            # Measure execution time for Option 2
            time2 = timeit.timeit(lambda: option2_compiled(x, i0, i1, alpha0, alpha1).block_until_ready(),
                                  number=num_runs)
            option2_avg_time = time2 / num_runs
            option2_times.append((batch_size, shape_size, option2_avg_time))

            option1_minus_option2_times.append((batch_size, shape_size, option1_avg_time - option2_avg_time))

            print(
                f"Batch size: {batch_size}, Shape size: {shape_size} | Option 1 Time: {option1_avg_time:.6f}s, Option 2 Time: {option2_avg_time:.6f}s")

    # Convert timing results to numpy arrays for plotting
    option1_times = np.array(option1_times)
    option2_times = np.array(option2_times)
    option1_minus_option2_times = np.array(option1_minus_option2_times)

    # Plotting the results
    plt.figure(figsize=(12, 6))

    # Plot for Option 1
    plt.subplot(1, 2, 1)
    plt.title('Option 1: Interpolation with moveaxis')
    for batch_size in batch_sizes:
        times = option1_times[option1_times[:, 0] == batch_size]
        plt.plot(times[:, 1], times[:, 2], label=f'Batch size {batch_size}')
    plt.xlabel('Shape size')
    plt.ylabel('Average Execution Time (s)')
    plt.legend()

    # Plot for Option 2
    plt.subplot(1, 2, 2)
    plt.title('Option 2: Interpolation with jax.vmap')
    for batch_size in batch_sizes:
        times = option2_times[option2_times[:, 0] == batch_size]
        plt.plot(times[:, 1], times[:, 2], label=f'Batch size {batch_size}')
    plt.xlabel('Shape size')
    plt.ylabel('Average Execution Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Comparison plot
    plt.figure(figsize=(8, 6))
    plt.title('Option 1 vs Option 2 Execution Time')
    for batch_size in batch_sizes:
        times1 = option1_times[option1_times[:, 0] == batch_size]
        times2 = option2_times[option2_times[:, 0] == batch_size]
        plt.plot(times1[:, 1], times1[:, 2], label=f'Option 1 - Batch {batch_size}')
        plt.plot(times2[:, 1], times2[:, 2], '--', label=f'Option 2 - Batch {batch_size}')
    plt.xlabel('Shape size')
    plt.ylabel('Average Execution Time (s)')
    plt.legend()
    plt.show()

    # Comparison plot
    plt.figure(figsize=(8, 6))
    plt.title('Option 1 - Option 2 Execution Time (>0 ==> Option 1 is slower)')
    for batch_size in batch_sizes:
        times = option1_minus_option2_times[option1_minus_option2_times[:, 0] == batch_size]
        plt.plot(times[:, 1], times[:, 2], label=f'Batch {batch_size}')
    plt.xlabel('Shape size')
    plt.ylabel('Average Execution Time (s)')
    plt.legend()
    plt.show()


def test_select_interpolation_points():
    desired_freqs = np.asarray([1.0])
    model_freqs = np.asarray([1.0, 2.0, 3.0])
    expected_select_idxs = np.asarray([0, 1])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)

    desired_freqs = np.asarray([1.0])
    model_freqs = np.asarray([1.0, 1.0, 2.0, 3.0])
    expected_select_idxs = np.asarray([1, 2])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)

    desired_freqs = np.asarray([3.0])
    model_freqs = np.asarray([1.0, 2.0, 3.0])
    expected_select_idxs = np.asarray([2])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)

    desired_freqs = np.asarray([3.0])
    model_freqs = np.asarray([1.0, 2.0, 3.0, 3.0])
    expected_select_idxs = np.asarray([3])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)

    desired_freqs = np.asarray([1.0, 2.0, 3.0])
    model_freqs = np.asarray([0.5, 1.5, 2.5, 3.5])
    expected_select_idxs = np.asarray([0, 1, 2, 3])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)

    desired_freqs = np.asarray([1.0, 2.0, 3.0])
    model_freqs = np.asarray([0.5, 1.5, 1.5, 1.5, 1.75, 2.5, 3.5])
    expected_select_idxs = np.asarray([0, 1, 4, 5, 6])
    select_idxs = select_interpolation_points(desired_freqs, model_freqs)
    np.testing.assert_allclose(select_idxs, expected_select_idxs)
