import glob
import os
from typing import List

from src.dsa2000_cal.assets import source_model_registry
from src.dsa2000_cal.assets.source_models.source_model import AbstractWSCleanSourceModel


@source_model_registry(template='vir_a')
class TauASourceModel(AbstractWSCleanSourceModel):

    def get_wsclean_clean_component_file(self) -> str:
        return os.path.join(*self.content_path, 'Vir-sources.txt')

    def get_wsclean_fits_files(self) -> List[str]:
        fits_files = glob.glob(os.path.join(*self.content_path, 'fits_models', 'Vir-*-model.fits'))
        return fits_files
