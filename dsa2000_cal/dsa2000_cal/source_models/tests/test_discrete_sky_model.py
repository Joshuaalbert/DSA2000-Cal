from astropy import coordinates as ac, units as au, time as at

from dsa2000_cal.source_models.discrete_sky_model import DiscreteSkyModel


def test_discrete_sky_model():
    coords_icrs = ac.ICRS(4 * au.deg, 2 * au.deg)
    freqs = au.Quantity([1, 2, 3], unit=au.Hz)
    brightness = au.Quantity([[1, 2, 3]], unit=au.Jy)

    discrete_sky_model = DiscreteSkyModel(coords_icrs=coords_icrs, freqs=freqs, brightness=brightness)
    assert discrete_sky_model.num_sources == 1
    assert discrete_sky_model.num_freqs == 3

    array_location = ac.EarthLocation.from_geodetic(0, 0, 0)
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    pointing = ac.ICRS(0 * au.deg, 0 * au.deg)
    lmn = discrete_sky_model.compute_lmn(pointing, array_location, time)
    assert lmn.shape == (1, 3)
