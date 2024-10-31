import jax
import numpy as np
from astropy import units as au

from dsa2000_cal.assets import MockCellTower


def test_lte_rfi_source_factory():
    model = MockCellTower(seed='test')
    import pylab as plt
    source_params = model.make_source_params(freqs=np.linspace(700, 800, 50) * au.MHz)
    delays = np.linspace(-1e7, 1e7, 1000)
    print(source_params.delay_acf)
    plt.plot(delays, jax.vmap(source_params.delay_acf)(delays)[:, 0, 0])
    plt.xlabel('Delay [s]')
    plt.ylabel('Auto-correlation function')
    plt.show()