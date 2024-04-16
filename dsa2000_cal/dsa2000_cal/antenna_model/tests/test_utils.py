import astropy.units as au
import numpy as np

from dsa2000_cal.antenna_model.utils import bore_sight_coords_to_pixel_coords, pixel_coords_to_bore_sight_coords, \
    plot_circular_beam, get_dish_model_beam_widths
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry


def _test_bore_sight_coords_to_pixel_coords():
    # Broken
    def wrap(x):
        return np.arctan2(np.sin(x), np.cos(x))

    theta = wrap(np.linspace(0, np.pi, 3))
    phi = wrap(np.linspace(0, 2 * np.pi, 3))
    X, Y = np.meshgrid(phi, theta)
    x, y = bore_sight_coords_to_pixel_coords(X, Y)
    X2, Y2 = pixel_coords_to_bore_sight_coords(x, y)
    assert np.allclose(X, X2)

    assert np.allclose(Y, Y2)


def test_get_beam_width():
    fill_registries()
    antenna_beam = array_registry.get_instance(array_registry.get_match('dsa2000W')).get_antenna_beam()
    plot_circular_beam(antenna_beam.get_model(), theshold=0.5)
    freqs, beam_widths = get_dish_model_beam_widths(antenna_beam.get_model())
    assert np.all(beam_widths > 0. * au.deg) and np.all(beam_widths < 180. * au.deg)
