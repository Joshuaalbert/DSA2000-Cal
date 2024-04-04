import numpy as np
from jax import numpy as jnp

from dsa2000_cal.predict.vec_utils import unvec, vec


def flatten_coherencies(coherencies):
    """
    Flatten coherencies.

    Args:
        coherencies: [2, 2] array of coherencies in the order [[XX, XY], [YX, YY]]

    Returns:
        [4] array of coherencies in the order [XX, XY, YX, YY]
    """
    return vec(coherencies, transpose=True)


def unflatten_coherencies(coherencies):
    """
    Unflatten coherencies.

    Args:
        coherencies: [4] array of coherencies in the order [XX, XY, YX, YY]

    Returns:
        [2, 2] array of coherencies in the order [[XX, XY], [YX, YY]]
    """
    return unvec(coherencies, (2, 2), transpose=True)


def stokes_to_linear(stokes_coherencies, flat_output: bool = False):
    if np.size(stokes_coherencies) != 4:
        raise ValueError("Stokes coherencies must have 4 elements.")
    if np.shape(stokes_coherencies) == (2, 2):
        stokes_coherencies = flatten_coherencies(stokes_coherencies)  # [4]
    I, Q, U, V = stokes_coherencies
    XX = I + Q
    XY = U + 1j * V
    YX = U - 1j * V
    YY = I - Q
    output = 0.5 * jnp.asarray([XX, XY, YX, YY])
    if flat_output:
        return output
    return unflatten_coherencies(output)


def stokes_to_circular(stokes_coherencies, flat_output: bool = False):
    if np.size(stokes_coherencies) != 4:
        raise ValueError("Stokes coherencies must have 4 elements.")
    if np.shape(stokes_coherencies) == (2, 2):
        stokes_coherencies = flatten_coherencies(stokes_coherencies)  # [4]
    I, Q, U, V = stokes_coherencies
    RR = I + V
    RL = Q + 1j * U
    LR = Q - 1j * U
    LL = I - V
    output = 0.5 * jnp.asarray([RR, RL, LR, LL])
    if flat_output:
        return output
    return unflatten_coherencies(output)


def linear_to_stokes(linear_coherencies, flat_output: bool = False):
    if np.size(linear_coherencies) != 4:
        raise ValueError("Linear coherencies must have 4 elements.")
    if np.shape(linear_coherencies) == (2, 2):
        linear_coherencies = flatten_coherencies(linear_coherencies)  # [4]
    XX, XY, YX, YY = linear_coherencies
    I = (XX + YY)
    Q = (XX - YY)
    U = (XY + YX)
    V = 1j * (YX - XY)
    output = jnp.asarray([I, Q, U, V])
    if flat_output:
        return output
    return unflatten_coherencies(output)


def circular_to_stokes(circular_coherencies, flat_output: bool = False):
    if np.size(circular_coherencies) != 4:
        raise ValueError("Circular coherencies must have 4 elements.")
    if np.shape(circular_coherencies) == (2, 2):
        circular_coherencies = flatten_coherencies(circular_coherencies)  # [4]
    RR, RL, LR, LL = circular_coherencies
    I = (RR + LL)
    Q = (RL + LR)
    U = 1j * (LR - RL)
    V = (RR - LL)
    output = jnp.asarray([I, Q, U, V])
    if flat_output:
        return output
    return unflatten_coherencies(output)


def linear_to_circular(linear_coherencies, flat_output: bool = False):
    return stokes_to_circular(linear_to_stokes(linear_coherencies, flat_output=True), flat_output=flat_output)


def circular_to_linear(circular_coherencies, flat_output: bool = False):
    return stokes_to_linear(circular_to_stokes(circular_coherencies, flat_output=True), flat_output=flat_output)
