from dsa2000_cal.assets.rfi.lte_rfi.lwa_cell_tower import LWACellTower


def test_lwa_cell_tower():
    lwa_cell_tower = LWACellTower(seed='abc')
    lwa_cell_tower.plot_acf()
