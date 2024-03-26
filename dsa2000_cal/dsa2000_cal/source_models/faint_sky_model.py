import dataclasses
from typing import List

import numpy as np
from astropy import coordinates as ac
from astropy import time as at
from astropy import units as au

from dsa2000_cal.common.coord_utils import icrs_to_lmn, lmn_to_icrs


def get_centre_ref_pixel(num_l, num_m) -> np.ndarray:
    # [0, 1, 2] -> [1]
    # [0, 1, 2, 3] -> [1.5]
    return np.array([num_l / 2, num_m / 2]) - 0.5


@dataclasses.dataclass(eq=False)
class FaintSkyModel:
    image: au.Quantity  # [num_l, num_m, num_freqs, num_stokes]
    cell_size: au.Quantity  # [2]
    freqs: au.Quantity  # [num_freqs]
    stokes: List[str]  # [num_stokes]
    ref_pixel: np.ndarray | None = None  # [2], if None then center of image

    def __post_init__(self):

        # Ensure shapes
        if self.image.ndim != 4:
            raise ValueError(f"Expected image to have 4 dimensions, got {self.image.ndim}")
        if self.ref_pixel is None:
            self.ref_pixel = get_centre_ref_pixel(*self.image.shape[:2])
        if self.ref_pixel.shape != (2,):
            raise ValueError(f"Expected ref_pixel to have shape (2,), got {self.ref_pixel.shape}")
        if self.cell_size.shape != (2,):
            raise ValueError(f"Expected cell_size to have shape (2,), got {self.cell_size.shape}")
        if self.image.shape[2] != len(self.freqs):
            raise ValueError(
                f"Expected image to have same number of frequencies as freqs, "
                f"got {self.image.shape[2]} and {len(self.freqs)}"
            )
        if self.image.shape[3] != len(self.stokes):
            raise ValueError(
                f"Expected image to have same number of stokes as stokes, "
                f"got {self.image.shape[3]} and {len(self.stokes)}"
            )

        # Assert freqs units are congruent with Hz
        if not self.freqs.unit.is_equivalent(au.Hz):
            raise ValueError(f"Expected freqs to be in Hz, got {self.freqs.unit}")

        # Assert brightness units are congruent with Jy
        if not self.image.unit.is_equivalent(au.Jy):
            raise ValueError(f"Expected brightness to be in Jy, got {self.image.unit}")

        # Assert cell_size units are congruent with deg
        if not self.cell_size.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected cell_size to be in deg, got {self.cell_size.unit}")

    def compute_icrs(self, pointing: ac.ICRS, array_location: ac.EarthLocation, time: at.Time):
        """
        Compute the ICRS coordinates of the sources.

        Args:
            pointing: the pointing direction
            array_location: the location of the array reference location
            time: the time of the observation
        """
        lmn = self.compute_lmn(pointing, array_location, time)
        shape = lmn.shape
        lmn = au.Quantity(lmn.reshape((-1, 3)), unit=au.dimensionless_unscaled)
        sources = lmn_to_icrs(lmn=lmn, array_location=array_location, time=time, phase_tracking=pointing)
        return sources.reshape(shape[:-1])

    def compute_lmn(self, pointing: ac.ICRS, array_location: ac.EarthLocation, time: at.Time):
        """
        Compute the l, m, n coordinates of the sources.

        Args:
            pointing: the pointing direction
            array_location: the location of the array reference location
            time: the time of the observation
        """
        # The image is given in a uniform grid over l,m (i.e. the sky tangent at the pointing direction)
        # dlm are computed at the reference pixel and then the l,m coordinates are computed for each pixel
        # relative to the reference pixel
        lmn_pointing = icrs_to_lmn(sources=pointing, phase_tracking=pointing, array_location=array_location, time=time)
        pixel_source_ra = ac.ICRS(pointing.ra + self.cell_size[0], pointing.dec)
        pixel_source_dec = ac.ICRS(pointing.ra, pointing.dec + self.cell_size[1])
        lmn_ra = icrs_to_lmn(sources=pixel_source_ra, phase_tracking=pointing, array_location=array_location, time=time)
        lmn_dec = icrs_to_lmn(sources=pixel_source_dec, phase_tracking=pointing, array_location=array_location, time=time)
        dl = lmn_ra[0] - lmn_pointing[0]
        dm = lmn_dec[1] - lmn_pointing[1]
        L, M = np.meshgrid(
            (np.arange(self.image.shape[0]) - self.ref_pixel[0]) * dl,
            (np.arange(self.image.shape[1]) - self.ref_pixel[1]) * dm,
            indexing='ij'
        )
        N = np.sqrt(1 - L ** 2 - M ** 2)
        # [0, 1, 2] -> [-1, 0, 1] * dl
        lmn = np.stack([L, M, N], axis=-1)  # [num_l, num_m, 3]
        return lmn


