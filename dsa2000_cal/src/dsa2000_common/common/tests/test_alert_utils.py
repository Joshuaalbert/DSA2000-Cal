from datetime import datetime, timedelta

from dsa2000_common.common.alert_utils import post_completed_forward_modelling_run


def test_post_completed_forward_modelling_run():
    post_completed_forward_modelling_run(
        "/test/some/run/dir",
        datetime.now(),
        timedelta(seconds=10)
    )
    assert True
