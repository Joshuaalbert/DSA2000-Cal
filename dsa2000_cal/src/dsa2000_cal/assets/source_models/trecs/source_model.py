import glob
import os
from typing import List

from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.assets.source_models.source_model import AbstractWSCleanSourceModel


@source_model_registry(template='trecs')
class TrecsSourceModel(AbstractWSCleanSourceModel):

    def get_wsclean_clean_component_file(self) -> str:
        raise NotImplementedError('Not implemented')

    def get_wsclean_fits_files(self) -> List[str]:
        fits_files = glob.glob(os.path.join(*self.content_path, 'fits_models', '*.fits'))
        return fits_files
