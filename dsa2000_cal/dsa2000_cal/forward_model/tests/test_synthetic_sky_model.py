import pytest
from astropy import units as au, coordinates as ac, time as at

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.common.astropy_utils import create_spherical_grid
from dsa2000_cal.forward_model.synthetic_sky_model import choose_dr, SyntheticSkyModelProducer


@pytest.mark.parametrize('total_n, expected_n', [
    (1, 1),
    (2, 7),
    (7, 7),
    (18, 7),
    (19, 19)
])
def test_choose_dr(total_n, expected_n):
    field_of_view = 4 * au.deg
    dr = choose_dr(field_of_view=field_of_view, total_n=total_n)
    sources = create_spherical_grid(
        pointing=ac.ICRS(15 * au.deg, 0 * au.deg),
        angular_radius=0.5 * field_of_view,
        dr=dr,
        sky_rotation=40 * au.deg
    )
    import pylab as plt
    plt.plot(sources.ra, sources.dec, 'o')
    plt.show()
    assert len(sources) == expected_n


def test_create_sky_model():
    fill_registries()
    sky_model_producer = SyntheticSkyModelProducer(
        phase_tracking=ac.ICRS(15 * au.deg, 0 * au.deg),
        obs_time=at.Time('2021-01-01T00:00:00'),
        freqs=au.Quantity([0.7, 1.4, 2], unit=au.GHz),
        num_bright_sources=7,
        num_faint_sources=7,
        field_of_view=au.Quantity(2, au.deg),
        mean_major=au.Quantity(1, au.arcmin),
        mean_minor=au.Quantity(1, au.arcmin),
        seed=42
    )
    sky_model = sky_model_producer.create_sky_model(
        include_bright=True,
        include_faint=True,
        include_bright_outside_fov=True,
        include_a_team=True,
        include_trecs=False,
        include_illustris=False
    )

    print(sky_model)
