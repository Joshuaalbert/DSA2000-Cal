import dataclasses

from astropy import coordinates as ac
from astropy import time as at
from astropy import units as au

from dsa2000_cal.coord_utils import icrs_to_lmn


@dataclasses.dataclass(eq=False)
class DiscreteSkyModel:
    coords_icrs: ac.ICRS  # [num_sources]
    freqs: au.Quantity  # [num_freqs]
    brightness: au.Quantity  # [num_sources, num_freqs]

    def __post_init__(self):
        if self.coords_icrs.isscalar:
            # Reshape to 1D array
            self.coords_icrs = self.coords_icrs.reshape((1,))

        # Check shapes
        if self.brightness.shape != (self.coords_icrs.shape[0], self.freqs.shape[0]):
            raise ValueError(
                f"brightness shape {self.brightness.shape} does not match coords_icrs shape {self.coords_icrs.shape} "
                f"and freqs shape {self.freqs.shape}."
            )
        self.num_sources = self.coords_icrs.shape[0]
        self.num_freqs = self.freqs.shape[0]

        # Assert freqs units are congruent with Hz
        if not self.freqs.unit.is_equivalent(au.Hz):
            raise ValueError(f"Expected freqs to be in Hz, got {self.freqs.unit}")

        # Assert brightness units are congruent with Jy
        if not self.brightness.unit.is_equivalent(au.Jy):
            raise ValueError(f"Expected brightness to be in Jy, got {self.brightness.unit}")

    def compute_lmn(self, pointing: ac.ICRS, array_location: ac.EarthLocation, time: at.Time):
        """
        Compute the l and m coordinates of the sources.

        Args:
            pointing: the pointing direction
            array_location: the location of the array reference location
            time: the time of the observation
        """
        return icrs_to_lmn(self.coords_icrs, array_location, time, pointing)

