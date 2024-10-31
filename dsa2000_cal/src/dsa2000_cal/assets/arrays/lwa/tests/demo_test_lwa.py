from dsa2000_cal.assets import LWAArray


def test_lwa_beam():
    array = LWAArray(seed='test')
    print(len(array.get_antennas()), array.get_system_equivalent_flux_density())
    model = array.get_antenna_model()
    assert model.get_amplitude().shape == (len(model.get_theta()), len(model.get_phi()), len(model.get_freqs()), 2, 2)
    assert model.get_phase().shape == (len(model.get_theta()), len(model.get_phi()), len(model.get_freqs()), 2, 2)

    model.plot_polar_amplitude(p=0, q=0)
    model.plot_polar_amplitude(p=0, q=1)
    model.plot_polar_amplitude(p=1, q=0)
    model.plot_polar_amplitude(p=1, q=1)

    model.plot_polar_phase(p=0, q=0)
    model.plot_polar_phase(p=0, q=1)
    model.plot_polar_phase(p=1, q=0)
    model.plot_polar_phase(p=1, q=1)
