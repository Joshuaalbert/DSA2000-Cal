import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_cal.common.corr_translation import flatten_coherencies, unflatten_coherencies, stokes_to_linear, \
    linear_to_stokes, stokes_to_circular, circular_to_stokes, linear_to_circular, circular_to_linear


def test_flatten_coherencies():
    coherencies = jnp.asarray([[1, 2], [3, 4]])
    assert jnp.alltrue(flatten_coherencies(coherencies) == jnp.asarray([1, 2, 3, 4]))


def test_unflatten_coherencies():
    coherencies = jnp.asarray([1, 2, 3, 4])
    assert jnp.alltrue(unflatten_coherencies(coherencies) == jnp.asarray([[1, 2], [3, 4]]))


def test_linear_to_linear():
    stokes = jnp.asarray([1, 2, 3, 4])
    linear = stokes_to_linear(stokes, flat_output=True)
    stokes_output = linear_to_stokes(linear, flat_output=True)
    np.testing.assert_allclose(stokes, stokes_output, atol=1e-6)


def test_circular_to_circular():
    stokes = jnp.asarray([1, 2, 3, 4])
    circular = stokes_to_circular(stokes, flat_output=True)
    stokes_output = circular_to_stokes(circular, flat_output=True)
    np.testing.assert_allclose(stokes, stokes_output, atol=1e-6)


def test_linear_to_circular():
    linear = jnp.asarray([1, 2, 3, 4])
    circular = linear_to_circular(linear, flat_output=True)
    linear_output = circular_to_linear(circular, flat_output=True)
    np.testing.assert_allclose(linear, linear_output, atol=1e-6)


def test_circular_to_linear():
    circular = jnp.asarray([1, 2, 3, 4])
    linear = circular_to_linear(circular, flat_output=True)
    circular_output = linear_to_circular(linear, flat_output=True)
    np.testing.assert_allclose(circular, circular_output, atol=1e-6)


def test_linear_to_stokes():
    x = jnp.ones(4) + 0j
    T = jax.jacobian(lambda x: linear_to_stokes(x, flat_output=True), holomorphic=True)(x)
    T_correct = jnp.asarray([
        [1., 0., 0., 1.],
        [1., 0., 0., -1.],
        [0., 1., 1., 0.],
        [0., -1j, 1j, 0.]
    ])
    np.testing.assert_allclose(T, T_correct)
