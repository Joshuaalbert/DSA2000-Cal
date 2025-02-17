from dsa2000_assets.arrays.dsa2000W.array import DSA2000WArray



def test_dsa_beam():
    model = DSA2000WArray(seed='test').get_antenna_model()
    assert model.get_amplitude().shape == (len(model.get_theta()), len(model.get_phi()), len(model.get_freqs()), 2, 2)
    assert model.get_phase().shape == (len(model.get_theta()), len(model.get_phi()), len(model.get_freqs()), 2, 2)
    model.plot_polar_amplitude()
    model.plot_polar_phase()
