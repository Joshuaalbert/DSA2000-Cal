import os

from dsa2000_assets.base_content import BaseContent
from dsa2000_assets.registries import beam_model_registry
from dsa2000_common.abc import AbstractBeamModel
from dsa2000_fm.antenna_model.abc import AbstractAntennaModel
from dsa2000_fm.antenna_model.h5_efield_model import H5AntennaModelV1


@beam_model_registry(template='lwa_highres')
class LWAHighResBeamModel(BaseContent, AbstractBeamModel):
    def get_antenna_model(self) -> AbstractAntennaModel:
        return H5AntennaModelV1(
            beam_file=os.path.join(*self.content_path, 'OVRO-LWA_soil_pt.h5')
        )
