import glob
import os
from typing import List

from dsa2000_assets.base_content import BaseContent
from dsa2000_assets.registries import source_model_registry
from dsa2000_common.abc import AbstractWSCleanSourceModel


@source_model_registry(template='nvss_calibrators')
class NVSSCalibratorsSourceModel(BaseContent, AbstractWSCleanSourceModel):

    def get_wsclean_clean_component_file(self) -> str:
        return os.path.join(*self.content_path, 'nvss_calibrators_gt_117mJy.txt')

    def get_wsclean_fits_files(self) -> List[str]:
        fits_files = glob.glob(os.path.join(*self.content_path, 'fits_models', '*.fits'))
        return fits_files


def test_fix():
    cal = NVSSCalibratorsSourceModel(seed='abc')
    # fixed = []
    # with (open(cal.get_wsclean_clean_component_file()) as f):
    #     fixed.append(next(f))
    #     for line in f:
    #         line = line.split(',')
    #         ra_str = line[2]
    #         dec_str = line[3]
    #         if not ra_str.startswith('-'):
    #             ra_str = '+' + ra_str
    #         if not dec_str.startswith('-'):
    #             dec_str = '+' + dec_str
    #         line[2] = ra_str
    #         line[3] = dec_str
    #         line[5] = "[-0.7]"
    #         line[6] = "true"
    #         fixed.append(','.join(line))
    # print(fixed[:5])
    # with open(cal.get_wsclean_clean_component_file(), 'w') as f:
    #     f.write(''.join(fixed))
