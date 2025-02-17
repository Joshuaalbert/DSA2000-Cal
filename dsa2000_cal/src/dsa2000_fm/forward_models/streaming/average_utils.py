import numpy as np


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
    block_size = axis_size // num_model_size
    return array.reshape(np.shape(array)[:axis] + (num_model_size, block_size) + np.shape(array)[axis + 1:]).mean(
        axis=axis + 1)
