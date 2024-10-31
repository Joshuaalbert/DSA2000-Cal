from src.dsa2000_cal.antenna_model.h5_efield_model import H5AntennaModelV1
from src.dsa2000_cal.assets import DSA200031b


def test_beam_model():
    array = DSA200031b(seed='test')
    antenna_model: H5AntennaModelV1 = array.get_antenna_model()
    antenna_model.plot_e_field(nu=1)