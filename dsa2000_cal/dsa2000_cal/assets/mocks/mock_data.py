import os

from dsa2000_cal.assets.base_content import BaseContent


class MockData(BaseContent):
    def __init__(self):
        super().__init__(seed='mock')

    def faint_sky_model(self) -> str:
        return os.path.join(*self.content_path, "mock_faint_sky_model.fits")
