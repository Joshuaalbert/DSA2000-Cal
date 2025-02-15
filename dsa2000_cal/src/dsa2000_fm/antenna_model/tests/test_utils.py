import astropy.units as au
import numpy as np
import pytest

from dsa2000_fm.antenna_model.antenna_model_utils import bore_sight_coords_to_pixel_coords, pixel_coords_to_bore_sight_coords, \
    plot_beam_profile, get_dish_model_beam_widths
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry


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


# @pytest.mark.parametrize('array_name', ['lwa_mock', 'dsa2000W_small'])
@pytest.mark.parametrize('array_name', ['dsa2000_optimal_v1'])
def test_get_beam_width(array_name: str):
    fill_registries()
    antenna_model = array_registry.get_instance(array_registry.get_match(array_name)).get_antenna_model()
    # antenna_model.plot_polar_amplitude()
    # antenna_model.plot_polar_phase()
    plot_beam_profile(antenna_model, threshold=0.5)
    freqs, beam_widths = get_dish_model_beam_widths(antenna_model, threshold=0.5)
    print(freqs, beam_widths)
    assert np.all(beam_widths > 0. * au.deg) and np.all(beam_widths < 180. * au.deg)

    plot_beam_profile(antenna_model, threshold=0.1)
    freqs, beam_widths = get_dish_model_beam_widths(antenna_model, threshold=0.1)
    print(freqs, beam_widths)
    assert np.all(beam_widths > 0. * au.deg) and np.all(beam_widths < 180. * au.deg)

    plot_beam_profile(antenna_model, threshold=0.2)
    freqs, beam_widths = get_dish_model_beam_widths(antenna_model, threshold=0.2)
    print(freqs, beam_widths)
    assert np.all(beam_widths > 0. * au.deg) and np.all(beam_widths < 180. * au.deg)
