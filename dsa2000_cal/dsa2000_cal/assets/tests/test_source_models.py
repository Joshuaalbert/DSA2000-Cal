import os

from astropy import units as au

from dsa2000_cal.assets.source_models.cyg_a.source_model import CygASourceModel


def test_model():
    for freq, file in CygASourceModel(seed='abc').get_wsclean_fits_files():
        assert freq.unit.is_equivalent(au.Hz)
        assert os.path.isfile(file)
