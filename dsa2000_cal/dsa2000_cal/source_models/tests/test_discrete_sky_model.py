import numpy as np
from astropy import coordinates as ac, units as au, time as at

from dsa2000_cal.common.astropy_utils import create_spherical_grid
from dsa2000_cal.source_models.discrete_sky_model import DiscreteSkyModel


def test_discrete_sky_model():
    coords_icrs = ac.ICRS(4 * au.deg, 2 * au.deg)
    freqs = au.Quantity([1, 2, 3], unit=au.Hz)
    brightness_I = au.Quantity([[1, 2, 3]], unit=au.Jy)

    discrete_sky_model = DiscreteSkyModel(coords_icrs=coords_icrs, freqs=freqs, brightness=brightness_I)
    assert discrete_sky_model.num_sources == 1
    assert discrete_sky_model.num_freqs == 3

    time = at.Time("2021-01-01T00:00:00", scale='utc')
    phase_tracking = ac.ICRS(0 * au.deg, 0 * au.deg)
    lmn = discrete_sky_model.compute_lmn(phase_tracking, time)
    assert lmn.shape == (1, 3)


def test_discrete_sky_model_bigger():
    sources = create_spherical_grid(
        pointing=ac.ICRS(4 * au.deg, 2 * au.deg),
        angular_width=au.Quantity(1, au.deg),
        dr=au.Quantity(0.1, au.deg)
    )
    freqs = au.Quantity([1, 2, 3], unit=au.Hz)
    brightness = np.ones((len(sources), len(freqs), 4)) * au.Jy
    discrete_sky_model = DiscreteSkyModel(coords_icrs=sources, freqs=freqs, brightness=brightness)
    diameter = discrete_sky_model.get_angular_diameter()
    print(diameter)
    assert np.isclose(diameter, 2 * au.deg, atol=1e-6 * au.deg)
