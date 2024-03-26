import dataclasses

import numpy as np
from astropy import units as au, coordinates as ac, time as at

from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.gain_models.gain_model import GainModel, get_interp_indices_and_weights


@dataclasses.dataclass(eq=False)
class IonosphereGainModel(GainModel):
    """
    Uses nearest neighbour interpolation to compute the gain model.
    """
    freqs: au.Quantity  # [num_freqs]
    directions: ac.ICRS  # [num_dir]
    times: at.Time  # [num_time]
    dtec: au.Quantity  # [num_time, num_dir, num_ant]

    dtype: np.dtype = np.complex64

    TEC_CONV: float = -8.4479745  # MHz/mTECU

    def __post_init__(self):
        # make sure all 1D
        if self.freqs.isscalar:
            self.freqs = self.freqs.reshape((1,))
        if self.directions.isscalar:
            self.directions = self.directions.reshape((1,))
        if self.times.isscalar:
            self.times = self.times.reshape((1,))
        # Check shapes
        if len(self.dtec.shape) != 3:
            raise ValueError(f"Expected dtec to have 3 dimensions but got {len(self.dtec.shape)}")

        if self.dtec.shape[:2] != (self.times.shape[0], self.directions.shape[0]):
            raise ValueError(
                f"dtec shape {self.dtec.shape} does not match times shape {self.times.shape} "
                f"and directions shape {self.directions.shape}."
            )

    def compute_beam(self, sources: ac.ICRS, phase_tracking: ac.ICRS, array_location: ac.EarthLocation, time: at.Time):
        shape = sources.shape
        sources = sources.reshape((-1,))
        lmn_data = icrs_to_lmn(
            sources=self.directions,
            array_location=array_location,
            time=time,
            phase_tracking=phase_tracking
        )  # [num_dir, 3]
        lmn_sources = icrs_to_lmn(
            sources=sources,
            array_location=array_location,
            time=time,
            phase_tracking=phase_tracking
        )  # [source_shape, 3]
        # Find the nearest neighbour of each source to that in data

        dist_sq = np.sum(np.square(lmn_sources[:, None, :] - lmn_data[None, :, :]), axis=-1)  # [num_sources, num_dir]
        closest = np.argmin(dist_sq, axis=-1)  # [num_sources]

        # Interpolate in time
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(time.jd, self.times.jd)
        dtec = self.dtec[i0, closest, :] * alpha0 + self.dtec[i1, closest, :] * alpha1
        phase = dtec[..., None] * (self.TEC_CONV / self.freqs.to('MHz').value)  # [num_sources, num_ant, num_freq]
        phase = phase.reshape(shape + phase.shape[1:])
        scalar_gain = np.exp(np.asarray(1j * 2 * np.pi * phase, self.dtype))
        # set diagonal
        gains = np.zeros(phase.shape + (2, 2), self.dtype)
        gains[..., 0, 0] = scalar_gain
        gains[..., 1, 1] = scalar_gain

        return gains
