import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_cal.common.vec_utils import unvec, vec


def flatten_coherencies(coherencies: jax.Array) -> jax.Array:
    """
    Flatten coherencies.

    Args:
        coherencies: [2, 2] array of coherencies in the order [[XX, XY], [YX, YY]]

    Returns:
        [4] array of coherencies in the order [XX, XY, YX, YY]
    """
    return vec(coherencies, transpose=True)


def unflatten_coherencies(coherencies: jax.Array) -> jax.Array:
    """
    Unflatten coherencies.

    Args:
        coherencies: [4] array of coherencies in the order [XX, XY, YX, YY]

    Returns:
        [2, 2] array of coherencies in the order [[XX, XY], [YX, YY]]
    """
    return unvec(coherencies, (2, 2), transpose=True)


def stokes_to_linear(stokes_coherencies: jax.Array, flat_output: bool = False) -> jax.Array:
    if np.size(stokes_coherencies) != 4:
        raise ValueError("Stokes coherencies must have 4 elements.")
    if np.shape(stokes_coherencies) == (2, 2):
        stokes_coherencies = flatten_coherencies(stokes_coherencies)  # [4]
    I, Q, U, V = stokes_coherencies
    XX = I + Q
    XY = U + 1j * V
    YX = U - 1j * V
    YY = I - Q
    output = 0.5 * jnp.stack([XX, XY, YX, YY], axis=-1)
    if flat_output:
        return output
    return unflatten_coherencies(output)

def stokes_I_to_linear(stokes_I: jax.Array, flat_output: bool = False) -> jax.Array:
    if np.size(stokes_I) != 1:
        raise ValueError("Stokes I must have 1 element.")
    I = stokes_I
    XX = I
    XY = 0
    YX = 0
    YY = I
    output = 0.5 * jnp.stack([XX, XY, YX, YY], axis=-1)
    if flat_output:
        return output
    return unflatten_coherencies(output)


def stokes_to_circular(stokes_coherencies: jax.Array, flat_output: bool = False) -> jax.Array:
    if np.size(stokes_coherencies) != 4:
        raise ValueError("Stokes coherencies must have 4 elements.")
    if np.shape(stokes_coherencies) == (2, 2):
        stokes_coherencies = flatten_coherencies(stokes_coherencies)  # [4]
    I, Q, U, V = stokes_coherencies
    RR = I + V
    RL = Q + 1j * U
    LR = Q - 1j * U
    LL = I - V
    output = 0.5 * jnp.stack([RR, RL, LR, LL], axis=-1)
    if flat_output:
        return output
    return unflatten_coherencies(output)


def linear_to_stokes(linear_coherencies: jax.Array, flat_output: bool = False) -> jax.Array:
    if np.size(linear_coherencies) != 4:
        raise ValueError("Linear coherencies must have 4 elements.")
    if np.shape(linear_coherencies) == (2, 2):
        linear_coherencies = flatten_coherencies(linear_coherencies)  # [4]
    XX, XY, YX, YY = linear_coherencies
    I = (XX + YY)
    Q = (XX - YY)
    U = (XY + YX)
    V = 1j * (YX - XY)
    output = jnp.stack([I, Q, U, V], axis=-1)
    if flat_output:
        return output
    return unflatten_coherencies(output)


def circular_to_stokes(circular_coherencies: jax.Array, flat_output: bool = False) -> jax.Array:
    if np.size(circular_coherencies) != 4:
        raise ValueError("Circular coherencies must have 4 elements.")
    if np.shape(circular_coherencies) == (2, 2):
        circular_coherencies = flatten_coherencies(circular_coherencies)  # [4]
    RR, RL, LR, LL = circular_coherencies
    I = (RR + LL)
    Q = (RL + LR)
    U = 1j * (LR - RL)
    V = (RR - LL)
    output = jnp.stack([I, Q, U, V], axis=-1)
    if flat_output:
        return output
    return unflatten_coherencies(output)


def linear_to_circular(linear_coherencies: jax.Array, flat_output: bool = False) -> jax.Array:
    return stokes_to_circular(linear_to_stokes(linear_coherencies, flat_output=True), flat_output=flat_output)


def circular_to_linear(circular_coherencies: jax.Array, flat_output: bool = False) -> jax.Array:
    return stokes_to_linear(circular_to_stokes(circular_coherencies, flat_output=True), flat_output=flat_output)


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
