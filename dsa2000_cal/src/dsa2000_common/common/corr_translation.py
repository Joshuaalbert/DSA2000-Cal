import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_common.common.vec_utils import unvec, vec


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
    """
    Convert stokes coherencies to linear coherencies.

    Args:
        stokes_coherencies: [4] array of stokes coherencies in the order [I, Q, U, V] or [[I, Q], [U, V]]
        flat_output: if True, return the output as a flat array.

    Returns:
        [4] array of linear coherencies in the order [XX, XY, YX, YY], or [[XX, XY], [YX, YY]]
    """
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
    """
    Convert stokes I to linear coherencies.

    Args:
        stokes_I: [1] array of stokes I.
        flat_output: if True, return the output as a flat array.

    Returns:
        [4] array of linear coherencies in the order [XX, XY, YX, YY], or [[XX, XY], [YX, YY]]
    """
    if np.shape(stokes_I) != ():
        raise ValueError(f"Stokes I must be a scalar, got shape {np.shape(stokes_I)}.")
    I = stokes_I
    XX = I
    XY = YX = jnp.zeros_like(I)
    YY = I
    output = 0.5 * jnp.stack([XX, XY, YX, YY], axis=-1)
    if flat_output:
        return output
    return unflatten_coherencies(output)


def stokes_to_circular(stokes_coherencies: jax.Array, flat_output: bool = False) -> jax.Array:
    """
    Convert stokes coherencies to circular coherencies.

    Args:
        stokes_coherencies: [4] array of stokes coherencies in the order [I, Q, U, V] or [[I, Q], [U, V]]
        flat_output: if True, return the output as a flat array.

    Returns:
        [4] array of circular coherencies in the order [RR, RL, LR, LL], or [[RR, RL], [LR, LL]]
    """
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
    """
    Convert linear coherencies to stokes coherencies.

    Args:
        linear_coherencies: [4] array of linear coherencies in the order [XX, XY, YX, YY] or [[XX, XY], [YX, YY]]
        flat_output: if True, return the output as a flat array.

    Returns:
        [4] array of stokes coherencies in the order [I, Q, U, V], or [[I, Q], [U, V]]
    """
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
    """
    Convert circular coherencies to stokes coherencies.

    Args:
        circular_coherencies: [4] array of circular coherencies in the order [RR, RL, LR, LL] or [[RR, RL], [LR, LL]]
        flat_output: if True, return the output as a flat array.

    Returns:
        [4] array of stokes coherencies in the order [I, Q, U, V], or [[I, Q], [U, V]]
    """
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
    """
    Convert linear coherencies to circular coherencies.

    Args:
        linear_coherencies: [4] array of linear coherencies in the order [XX, XY, YX, YY] or [[XX, XY], [YX, YY]]
        flat_output: if True, return the output as a flat array.

    Returns:
        [4] array of circular coherencies in the order [RR, RL, LR, LL], or [[RR, RL], [LR, LL]]
    """
    return stokes_to_circular(linear_to_stokes(linear_coherencies, flat_output=True), flat_output=flat_output)


def circular_to_linear(circular_coherencies: jax.Array, flat_output: bool = False) -> jax.Array:
    """
    Convert circular coherencies to linear coherencies.

    Args:
        circular_coherencies: [4] array of circular coherencies in the order [RR, RL, LR, LL] or [[RR, RL], [LR, LL]]
        flat_output: if True, return the output as a flat array.

    Returns:
        [4] array of linear coherencies in the order [XX, XY, YX, YY], or [[XX, XY], [YX, YY]]
    """
    return stokes_to_linear(circular_to_stokes(circular_coherencies, flat_output=True), flat_output=flat_output)
