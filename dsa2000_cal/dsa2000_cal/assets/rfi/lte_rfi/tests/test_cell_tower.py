import numpy as np
from astropy import units as au

from dsa2000_cal.assets.rfi.lte_rfi.lwa_cell_tower import LWACellTower
from dsa2000_cal.assets.rfi.lte_rfi.mock_cell_tower import MockCellTower


def test_lte_rfi_source_factory():
    model = MockCellTower(seed='test')
    import pylab as plt
    source_params = model.make_source_params(freqs=np.linspace(700, 800, 50) * au.MHz)
    plt.plot(source_params.freqs, source_params.spectral_flux_density[0])
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Luminosity [W/Hz]')
    plt.show()
    plt.plot(source_params.delay_acf.x, source_params.delay_acf.values[:, 0])
    plt.xlabel('Delay [s]')
    plt.ylabel('Auto-correlation function')
    plt.show()
    assert source_params.delay_acf.regular_grid