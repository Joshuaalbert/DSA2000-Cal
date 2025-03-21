import jax.numpy as jnp
import numpy as np

from dsa2000_common.common.mixed_precision_utils import mp_policy


def average_rule(array, num_model_size: int, axis: int):
    """
    Block average array along axis.

    Args:
        array: [..., N, ...] on axis `axis`
        num_model_size: how many blocks to average
        axis: the axis

    Returns:
        [..., num_model_size, ...] on axis `axis`
    """
    axis_size = np.shape(array)[axis]
    if axis_size % num_model_size != 0:
        raise ValueError(f"Axis {axis} must be divisible by {num_model_size}.")
    if num_model_size > axis_size:
        raise ValueError(f"Axis {axis} smaller than {num_model_size}.")
    block_size = axis_size // num_model_size
    return array.reshape(np.shape(array)[:axis] + (num_model_size, block_size) + np.shape(array)[axis + 1:]).mean(
        axis=axis + 1)

def test_average_rule():
    assert np.shape(average_rule(np.ones((5,6,10)), 1, axis=1)) == (5, 1, 10)


def average_flags(flags, num_model_size: int, axis: int):
    # flags are same a zero weight
    weights = jnp.where(flags, 0., 1.).astype(mp_policy.weight_dtype)
    weights = average_weights(weights, num_model_size, axis)
    flags = (weights == 0).astype(mp_policy.flag_dtype)
    return flags


def test_average_flags():
    flags = jnp.asarray([False, True, False, False])
    np.testing.assert_allclose(average_flags(flags, 2, 0), [True, False])
    flags = jnp.asarray([False, True, True, False])
    np.testing.assert_allclose(average_flags(flags, 2, 0), [True, True])


def average_weights(weights, num_model_size: int, axis: int):
    # weights are same as inv variance
    inv_weights = jnp.reciprocal(weights)
    inv_weights = average_rule(inv_weights, num_model_size, axis)
    weights = jnp.reciprocal(inv_weights)
    return weights
