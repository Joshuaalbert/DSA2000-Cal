import os

from dsa2000_cal.assets.base_content import BaseContent


class Templates(BaseContent):
    def __init__(self):
        super().__init__(seed='Templates')

    def template_antenna_table(self) -> str:
        return os.path.join(*self.content_path, 'TEMPLATE_ANTENNA')
