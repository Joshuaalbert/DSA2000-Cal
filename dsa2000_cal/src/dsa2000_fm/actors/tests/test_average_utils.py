import numpy as np

from dsa2000_fm.actors.average_utils import average_rule


def test_average_rule():
    array = np.arange(9)
    num_model_times = 3
    axis = 0
    result = average_rule(array, num_model_times, axis)
    assert np.allclose(result, np.array([1., 4., 7.]))

    # Tile the array
    n = 5
    array = np.tile(array[None, :], (n, 1))
    result = average_rule(array, num_model_times, 1)
    assert np.allclose(result, np.array([1., 4, 7])[None, :])

    array = array.T
    result = average_rule(array, num_model_times, 0)
    assert np.allclose(result, np.array([1., 4, 7])[:, None])
