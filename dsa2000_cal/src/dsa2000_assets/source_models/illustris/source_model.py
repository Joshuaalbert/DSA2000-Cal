import glob
import os
from typing import List

from dsa2000_assets.base_content import BaseContent
from dsa2000_assets.registries import source_model_registry
from dsa2000_common.abc import AbstractWSCleanSourceModel


@source_model_registry(template='illustris')
class IllustrisSourceModel(BaseContent, AbstractWSCleanSourceModel):

    def get_wsclean_clean_component_file(self) -> str:
        return os.path.join(*self.content_path, 'Illustris-sources.txt')

    def get_wsclean_fits_files(self) -> List[str]:
        fits_files = glob.glob(os.path.join(*self.content_path, 'fits_models', '449659_2.0_60.0_4096_128_1.0_5.0_*.fits'))
        return fits_files
