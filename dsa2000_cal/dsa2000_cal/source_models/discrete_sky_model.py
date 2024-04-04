import dataclasses

import jax
import jax.numpy as jnp
from astropy import coordinates as ac
from astropy import time as at
from astropy import units as au

from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.source_models.corr_translation import stokes_to_linear


@dataclasses.dataclass(eq=False)
class DiscreteSkyModel:
    coords_icrs: ac.ICRS  # [num_sources]
    freqs: au.Quantity  # [num_freqs]
    brightness: au.Quantity  # [num_sources, num_freqs, 4] Stokes: I, Q, U, V

    def __post_init__(self):

        # Assert freqs units are congruent with Hz
        if not self.freqs.unit.is_equivalent(au.Hz):
            raise ValueError(f"Expected freqs to be in Hz, got {self.freqs.unit}")
        # Assert brightness units are congruent with Jy
        if not self.brightness.unit.is_equivalent(au.Jy):
            raise ValueError(f"Expected brightness to be in Jy, got {self.brightness.unit}")

        if self.coords_icrs.isscalar:
            # Reshape to 1D array
            self.coords_icrs = self.coords_icrs.reshape((1,))
        if self.freqs.isscalar:
            # Reshape to 1D array
            self.freqs = self.freqs.reshape((1,))

        self.num_sources = self.coords_icrs.shape[0]
        self.num_freqs = self.freqs.shape[0]

        # Check shapes
        if len(self.brightness.shape) == 2:
            self.brightness = self.brightness.reshape(self.brightness.shape + (1,))
        if self.brightness.shape[:2] != (self.coords_icrs.shape[0], self.freqs.shape[0]):
            raise ValueError(
                f"brightness shape {self.brightness.shape} does not match coords_icrs shape {self.coords_icrs.shape} "
                f"and freqs shape {self.freqs.shape}."
            )
        if self.brightness.shape[2] == 1:
            # Represnts I, rest are 0
            self.brightness = jnp.concatenate(
                [self.brightness,
                 jnp.zeros_like(self.brightness),  # Q
                 jnp.zeros_like(self.brightness),  # U
                 jnp.zeros_like(self.brightness)  # V
                 ],
                axis=-1
            ) * self.brightness.unit

        if self.brightness.shape[2] != 4:
            raise ValueError(f"Expected brightness to have 4 stokes, got {self.brightness.shape[2]}")

    def get_angular_diameter(self) -> au.Quantity:
        if self.coords_icrs.isscalar or self.coords_icrs.shape[0] == 1:
            return au.Quantity(0, 'rad')

        # Get maximal separation
        return max(coord.separation(self.coords_icrs).max() for coord in self.coords_icrs).to('deg')

    def get_source_model_linear(self, freqs: jax.Array) -> jnp.ndarray:

        # get freq interp indices
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(freqs, self.freqs)
        # Convert stokes to XX XY YX YY

        brightness = quantity_to_jnp(self.brightness, 'Jy')
        brightness = brightness[:, i0] * alpha0[:, None] + brightness[:, i1] * alpha1[:, None]

        I = brightness[:, :, 0]
        Q = brightness[:, :, 1]
        U = brightness[:, :, 2]
        V = brightness[:, :, 3]

        return stokes_to_linear(I, Q, U, V).reshape((self.num_sources, len(freqs), 2, 2))

    def compute_lmn(self, phase_tracking: ac.ICRS, array_location: ac.EarthLocation, time: at.Time):
        """
        Compute the l and m coordinates of the sources.

        Args:
            phase_tracking: the pointing direction
            array_location: the location of the array reference location
            time: the time of the observation
        """
        return icrs_to_lmn(self.coords_icrs, array_location, time, phase_tracking)
