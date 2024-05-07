import json
import os
import socket
import warnings
from datetime import datetime, timedelta

import requests


def post_completed_forward_modelling_run(run_dir: str, start_time: datetime, duration: timedelta,
                                         hook_url: str | None = None):
    """
    Post a message to slack that a forward modelling run has completed.

    Args:
        run_dir: the directory of the run
        start_time: the start time of the run
        duration: the duration of the run
    """
    if hook_url is None:
        hook_url = os.environ.get('SLACK_FINISHED_RUNS_HOOK_URL', None)

    if hook_url is None:
        warnings.warn("No SLACK_FINISHED_RUNS_HOOK_URL set. Not posting to slack.")

    hostname = socket.gethostname()

    data = {
        "run_duration": f"{duration.total_seconds()} seconds",
        "run_dir": f"{hostname}:{run_dir}",
        "run_start_time": f"{start_time.isoformat()}"
    }
    requests.post(
        url=hook_url,
        data=json.dumps(data),
        headers={'Content-type': 'application/json'}
    )
