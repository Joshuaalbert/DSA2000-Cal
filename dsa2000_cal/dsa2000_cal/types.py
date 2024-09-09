from typing import List

import numpy as np
from astropy import coordinates as ac, time as at, units as au

from dsa2000_cal.common.serialise_utils import SerialisableBaseModel


class CalibrationSolutions(SerialisableBaseModel):
    """
    Calibration solutions, stored in a serialisable format.
    """
    times: at.Time  # [time]
    pointings: ac.ICRS | None  # [[ant]]
    antennas: ac.EarthLocation  # [ant]
    antenna_labels: List[str]  # [ant]
    freqs: au.Quantity  # [chan]
    gains: np.ndarray  # [facet, time, ant, chan[, 2, 2]]


class SystemGains(SerialisableBaseModel):
    """
    Simulated system gains, stored in a serialisable format.
    """
    directions: ac.ICRS  # [source]
    times: at.Time  # [time]
    antennas: ac.EarthLocation  # [ant]
    antenna_labels: List[str]  # [ant]
    freqs: au.Quantity  # [chan]
    gains: np.ndarray  # [source, time, ant, chan, 2, 2]
