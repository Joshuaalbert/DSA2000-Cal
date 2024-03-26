import numpy as np
from astropy import units as au, coordinates as ac, time as at

from dsa2000_cal.source_models.faint_sky_model import get_centre_ref_pixel, FaintSkyModel


def test_get_centre_ref_pixel():
    # [0, 1, 2] -> [1]
    assert get_centre_ref_pixel(3, 3).tolist() == [1, 1]
    # [0, 1, 2, 3] -> [1.5]
    assert get_centre_ref_pixel(4, 4).tolist() == [1.5, 1.5]


def test_faint_sky_model():
    freqs = au.Quantity([1, 2, 3, 4], unit=au.Hz)
    image = au.Quantity(np.zeros((13, 13, 4, 1)), unit=au.Jy)
    cell_size = au.Quantity([0.1, 0.1], unit=au.deg)

    faint_sky_model = FaintSkyModel(
        image=image,
        cell_size=cell_size,
        freqs=freqs,
        stokes=['I']
    )

    array_location = ac.EarthLocation.from_geodetic(0, 0, 0)
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    pointing = ac.ICRS(0 * au.deg, 0 * au.deg)
    lmn = faint_sky_model.compute_lmn(pointing, array_location, time)
    assert lmn.shape == (13, 13, 3)

    # Check that the lmn coordinates are correct
    assert np.isfinite(lmn).all()

    sources = faint_sky_model.compute_icrs(pointing, array_location, time)
    # ref_pixel is float make int
    ref_pixel = (int(faint_sky_model.ref_pixel[0]), int(faint_sky_model.ref_pixel[1]))

    # Check at ref pixel the cell size is correct
    next_source: ac.ICRS = sources[ref_pixel[0] + 1, ref_pixel[1]]
    sep_ra = next_source.separation(sources[ref_pixel[0], ref_pixel[1]])
    np.testing.assert_allclose(sep_ra, cell_size[0])

    next_source: ac.ICRS = sources[ref_pixel[0], ref_pixel[1] + 1]
    sep_dec = next_source.separation(sources[ref_pixel[0], ref_pixel[1]])
    np.testing.assert_allclose(sep_dec, cell_size[1])
