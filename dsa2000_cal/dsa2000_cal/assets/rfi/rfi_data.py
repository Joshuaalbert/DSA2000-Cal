import os

from dsa2000_cal.assets.base_content import BaseContent


class RFIData(BaseContent):
    def __init__(self):
        super().__init__(seed='rfi data')

    def rfi_injection_model(self) -> str:
        return os.path.join(*self.content_path, '../../../large_files/rfi_injection_model.mat')
