from datetime import datetime, tzinfo
from typing import Union, Tuple


def set_datetime_timezone(dt: datetime, offset: Union[str, tzinfo]) -> datetime:
    """
    Replaces the datetime object's timezone with one from an offset.

    Args:
        dt: datetime, with out without a timezone set. If set, will be replaced.
        offset: tzinfo, or str offset like '-04:00' (which means EST)

    Returns:
        datetime with timezone set
    """
    if isinstance(offset, str):
        dt = dt.replace(tzinfo=None)
        return datetime.fromisoformat(f"{dt.isoformat()}{offset}")
    if isinstance(offset, tzinfo):
        return dt.replace(tzinfo=offset)
    raise ValueError(f"offset {offset} not understood.")


def current_utc() -> datetime:
    """
    Get the current datetime in UTC, with timezone set to UTC.

    Returns:
        datetime
    """
    return set_datetime_timezone(datetime.utcnow(), '+00:00')


def get_microtimestamp_range(microtimestamp: int, block_size_hours: float = 1.) -> Tuple[int, int]:
    """
    Get the microtimestamp range for a microtimestamp.

    Args:
        microtimestamp: microtimestamp
        block_size_hours: block size in hours

    Returns:
        microtimestamp range
    """
    dt = int(3600 * block_size_hours * 1e6)
    start_microtimestamp = microtimestamp - (microtimestamp % dt)
    end_microtimestamp = start_microtimestamp + dt
    return start_microtimestamp, end_microtimestamp
