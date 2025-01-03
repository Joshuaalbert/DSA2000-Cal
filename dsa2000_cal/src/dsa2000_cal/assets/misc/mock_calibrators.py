import os

from dsa2000_cal.assets.base_content import BaseContent
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import misc_registry


@misc_registry(template='mock_calibrators')
class MockCalibrators(BaseContent):
    """
    Abstract array class.
    """

    def __init__(self, *args, **kwargs):
        BaseContent.__init__(self, seed='survey_pointings')

    def wsclean_component_file(self) -> str:
        return str(os.path.join(*self.content_path, '../source_models/mock_calibrators/mock_calibrators.txt'))


def test_mock_calibrators():
    fill_registries()
    survey_pointings = misc_registry.get_instance(misc_registry.get_match('mock_calibrators'))
    print(survey_pointings.wsclean_component_file())
