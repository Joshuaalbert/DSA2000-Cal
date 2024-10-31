import inspect
import json
import os
import warnings
from datetime import datetime, timedelta

import requests


def get_grandparent_info(relative_depth: int = 7):
    """
    Get the file, line number and function name of the caller of the caller of this function.

    Args:
        relative_depth: the number of frames to go back from the caller of this function. Default is 6. Should be
        enough to get out of a jax.tree.map call.

    Returns:
        str: a string with the file, line number and function name of the caller of the caller of this function.
    """
    # Get the grandparent frame (caller of the caller)
    s = []
    for depth in range(1, min(1 + relative_depth, len(inspect.stack()) - 1) + 1):
        caller_frame = inspect.stack()[depth]
        caller_file = caller_frame.filename
        caller_line = caller_frame.lineno
        caller_func = caller_frame.function
        s.append(f"{os.path.basename(caller_file)}:{caller_line} in {caller_func}")
    s = s[::-1]
    s = f"at {' -> '.join(s)}"
    return s


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
        return

    hostname = os.uname().nodename

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
