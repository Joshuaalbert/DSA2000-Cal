from dsa2000_cal.assets import LWAArray


def test_lwa_beam():
    model = LWAArray(seed='test').get_antenna_model()
    assert model.get_amplitude().shape == (len(model.get_theta()), len(model.get_phi()), len(model.get_freqs()), 2, 2)
    assert model.get_phase().shape == (len(model.get_theta()), len(model.get_phi()), len(model.get_freqs()), 2, 2)
    model.plot_polar_amplitude()
    model.plot_polar_phase()
