import os

import astropy.units as au

from dsa2000_cal.antenna_model.abc import AbstractAntennaModel
from dsa2000_cal.antenna_model.h5_efield_model import H5AntennaModelV1
from dsa2000_cal.antenna_model.matlab_amplitude_only_model import MatlabAntennaModelV1
from dsa2000_cal.assets.beam_models.beam_model import AbstractBeamModel
from dsa2000_cal.assets.registries import beam_model_registry


@beam_model_registry(template='dsa_original')
class DSAOriginalBeamModel(AbstractBeamModel):
    def get_antenna_model(self) -> AbstractAntennaModel:
        return MatlabAntennaModelV1(
            antenna_model_file=os.path.join(*self.content_path, 'dsa2000_antenna_model.mat'),
            model_name='coPolPattern_dBi_Freqs_15DegConicalShield'
        )


@beam_model_registry(template='dsa_prototype')
class DSAPrototypeBeamModel(AbstractBeamModel):
    def get_antenna_model(self) -> AbstractAntennaModel:
        return H5AntennaModelV1(
            angular_units=au.deg,
            freq_units=au.GHz,
            beam_file=os.path.join(*self.content_path,
                                   'DSA2000-beam-wShieldSolidCylinder600mm.h5')
        )
