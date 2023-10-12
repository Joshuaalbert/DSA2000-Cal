import numpy as np

from dsa2000_cal.ionosphere.interpolate_h5parm import compute_mean_coordinates
from dsa2000_cal.ionosphere.ionosphere_simulation import grid_coordinates


def test_grid_coordiates():
    coords = np.random.uniform(low=0, high=[1, 1], size=(10, 2))
    coords_grid = grid_coordinates(coords, dx=0.15)
    dx = np.min(np.max(np.abs(coords_grid[:, None, :] - coords[None, :, :]), axis=-1), axis=-1)
    assert np.all(dx <= 0.15)
    #
    # import pylab as plt
    # plt.scatter(coords[:, 0], coords[:, 1])
    # plt.scatter(coords_grid[:, 0], coords_grid[:, 1])
    # plt.show()


def test_compute_mean_coordinates():
    # Example coordinates
    coordinates = np.array([[0, 0],
                            [0, np.pi / 2]])

    # Compute the mean coordinates
    mean_coords = compute_mean_coordinates(coordinates)

    assert np.allclose(mean_coords, np.asarray([0., np.pi / 4]), atol=1e-4)
