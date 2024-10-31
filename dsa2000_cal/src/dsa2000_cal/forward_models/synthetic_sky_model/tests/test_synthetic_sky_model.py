import pytest
from astropy import units as au, coordinates as ac

from dsa2000_cal.common.astropy_utils import create_spherical_grid_old, choose_dr


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
    sources = create_spherical_grid_old(
        pointing=ac.ICRS(15 * au.deg, 0 * au.deg),
        angular_radius=0.5 * field_of_view,
        dr=dr,
        sky_rotation=40 * au.deg
    )
    import pylab as plt
    plt.plot(sources.ra, sources.dec, 'o')
    plt.show()
    assert len(sources) == expected_n
