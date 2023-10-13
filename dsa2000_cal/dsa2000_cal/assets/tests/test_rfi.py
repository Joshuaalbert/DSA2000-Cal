from scipy.io import loadmat

from dsa2000_cal.assets.rfi.rfi_data import RFIData


def test_antenna_pattern():
    ant_model = loadmat(RFIData().dsa2000_antenna_model())
    print(ant_model['ThetaDeg'].shape)
    print(ant_model['PhiDeg'].shape)
    print(ant_model['coPolPattern_dBi_Freqs_15DegConicalShield'].shape)
    print(ant_model['coPolPattern_dBi_Freqs_CylindricalShield'].shape)
    print(ant_model['coPolPattern_dBi_Freqs_NoShield'].shape)
    print(ant_model['freqListGHz'].shape)
    print(ant_model.keys())