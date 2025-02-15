import glob
import os
from typing import List

from dsa2000_assets.registries import source_model_registry
from dsa2000_common.abc import AbstractWSCleanSourceModel


@source_model_registry(template='tau_a')
class TauASourceModel(AbstractWSCleanSourceModel):

    def get_wsclean_clean_component_file(self) -> str:
        return os.path.join(*self.content_path, 'Tau-sources.txt')

    def get_wsclean_fits_files(self) -> List[str]:
        fits_files = glob.glob(os.path.join(*self.content_path, 'fits_models', 'Tau-*-model.fits'))
        return fits_files
