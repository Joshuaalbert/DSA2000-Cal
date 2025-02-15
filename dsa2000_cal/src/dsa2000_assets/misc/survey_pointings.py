import os

import astropy.coordinates as ac

from dsa2000_assets.base_content import BaseContent
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import misc_registry
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel


@misc_registry(template='survey_pointings')
class SurveyPointings(BaseContent):
    """
    Abstract array class.
    """

    def __init__(self, *args, **kwargs):
        BaseContent.__init__(self, seed='survey_pointings')

    def survey_pointings_v1(self) -> ac.ICRS:
        class SurveyPointingsV1(SerialisableBaseModel):
            pointings: ac.ICRS

        file = str(os.path.join(*self.content_path, 'survey_pointings_v1.json'))
        survey_pointings = SurveyPointingsV1.parse_file(file)
        return survey_pointings.pointings


def test_survey_pointings():
    fill_registries()
    survey_pointings = misc_registry.get_instance(misc_registry.get_match('survey_pointings'))
    print(survey_pointings.survey_pointings_v1())
    # <ICRS Coordinate: (ra, dec) in deg
    #     [(328.88708087,  89.85172121), (135.64056827,  88.75188608),
    #      (275.8150158 ,  88.32370549), ..., (127.25737576, -30.97488149),
    #      (264.76513561, -30.98653843), ( 42.27289489, -30.99819841)]>