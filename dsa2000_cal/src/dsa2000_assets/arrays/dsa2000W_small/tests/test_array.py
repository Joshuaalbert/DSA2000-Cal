import astropy.coordinates as ac

from dsa2000_assets.arrays.dsa2000W_small.array import DSA2000WSmallArray


def test_beam():
    model = DSA2000WSmallArray(seed='test').get_antenna_model()
    assert model.get_amplitude().shape == (len(model.get_theta()), len(model.get_phi()), len(model.get_freqs()), 2, 2)
    assert model.get_phase().shape == (len(model.get_theta()), len(model.get_phi()), len(model.get_freqs()), 2, 2)
    model.plot_polar_amplitude()
    model.plot_polar_phase()


def test_antennas():
    array = DSA2000WSmallArray(seed='test')
    antennas = array.get_antennas()
    assert isinstance(antennas, ac.EarthLocation)
