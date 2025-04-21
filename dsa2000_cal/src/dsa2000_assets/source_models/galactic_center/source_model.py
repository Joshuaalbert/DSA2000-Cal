import glob
import os
from typing import List

from dsa2000_assets.base_content import BaseContent
from dsa2000_assets.registries import source_model_registry
from dsa2000_common.abc import AbstractWSCleanSourceModel


@source_model_registry(template='meerkat_gc')
class MeerKatCGSourceModel(BaseContent, AbstractWSCleanSourceModel):

    def get_wsclean_clean_component_file(self) -> str:
        raise NotImplementedError()

    def get_wsclean_fits_files(self) -> List[str]:
        fits_files = glob.glob(os.path.join(*self.content_path, 'KATCG-model.fits'))
        return fits_files
