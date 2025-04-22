import glob
import os
from typing import List

from dsa2000_assets.base_content import BaseContent
from dsa2000_assets.registries import source_model_registry
from dsa2000_common.abc import AbstractWSCleanSourceModel


@source_model_registry(template='point_sources')
class NCG5194SourceModel(BaseContent, AbstractWSCleanSourceModel):

    def get_wsclean_clean_component_file(self) -> str:
        raise NotImplementedError()

    def get_wsclean_fits_files(self) -> List[str]:
        fits_files = glob.glob(os.path.join(*self.content_path, 'point_sources.fits'))
        return fits_files

# if __name__ == '__main__':
#     with standardize_fits('NGC_5194_RO_MOM0_THINGS.FITS', 'NGC_5194_RO_MOM0_THINGS_STD.FITS', overwrite=True):
#         ...
#     # standardize_fits('KATGC-model.fits', 'KATGC-model_STD.fits', overwrite=True)
