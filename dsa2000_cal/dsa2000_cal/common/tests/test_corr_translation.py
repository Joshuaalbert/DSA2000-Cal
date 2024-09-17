import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_cal.common.corr_translation import flatten_coherencies, unflatten_coherencies, stokes_to_linear, \
    linear_to_stokes, stokes_to_circular, circular_to_stokes, linear_to_circular, circular_to_linear


def test_flatten_coherencies():
    coherencies = jnp.asarray([[1, 2], [3, 4]])
    assert jnp.all(flatten_coherencies(coherencies) == jnp.asarray([1, 2, 3, 4]))


def test_unflatten_coherencies():
    coherencies = jnp.asarray([1, 2, 3, 4])
    assert jnp.all(unflatten_coherencies(coherencies) == jnp.asarray([[1, 2], [3, 4]]))


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


def test_proofs():
    import sympy as sp

    def circular_to_stokes(circular_coherencies):
        RR, RL, LR, LL = circular_coherencies
        I = (RR + LL)
        Q = (RL + LR)
        U = sp.I * (LR - RL)
        V = (RR - LL)
        return sp.Matrix([I, Q, U, V])

    def stokes_to_circular(stokes_coherencies):
        I, Q, U, V = stokes_coherencies
        RR = I + V
        RL = Q + sp.I * U
        LR = Q - sp.I * U
        LL = I - V
        return sp.Matrix([RR, RL, LR, LL]) / sp.Rational(2)

    def linear_to_stokes(linear_coherencies):
        XX, XY, YX, YY = linear_coherencies
        I = (XX + YY)
        Q = (XX - YY)
        U = (XY + YX)
        V = sp.I * (YX - XY)
        return sp.Matrix([I, Q, U, V])

    def stokes_to_linear(stokes_coherencies):
        I, Q, U, V = stokes_coherencies
        XX = I + Q
        XY = U + sp.I * V
        YX = U - sp.I * V
        YY = I - Q
        return sp.Matrix([XX, XY, YX, YY]) / sp.Rational(2)

    RR, RL, LR, LL = sp.symbols('RR RL LR LL')
    XX, XY, YX, YY = sp.symbols('XX XY YX YY')
    I, Q, U, V = sp.symbols('I Q U V')

    # Linear stokes
    assert stokes_to_linear(linear_to_stokes([XX, XY, YX, YY])) == sp.Matrix([XX, XY, YX, YY])
    assert linear_to_stokes(stokes_to_linear([I, Q, U, V])) == sp.Matrix([I, Q, U, V])
    # Circular stokes
    assert stokes_to_circular(circular_to_stokes([RR, RL, LR, LL])) == sp.Matrix([RR, RL, LR, LL])
    assert circular_to_stokes(stokes_to_circular([I, Q, U, V])) == sp.Matrix([I, Q, U, V])
