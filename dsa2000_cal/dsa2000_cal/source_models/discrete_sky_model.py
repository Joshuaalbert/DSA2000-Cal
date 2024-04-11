import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from astropy import coordinates as ac
from astropy import time as at
from astropy import units as au
from jax import lax

from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.source_models.corr_translation import stokes_to_linear


@dataclasses.dataclass(eq=False)
class DiscreteSkyModel:
    coords_icrs: ac.ICRS
    freqs: au.Quantity
    brightness: au.Quantity

    def __post_init__(self):

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
        if self.brightness.shape[:2] != (self.num_sources, self.num_freqs):
            raise ValueError(
                f"brightness shape {self.brightness.shape} does not match coords_icrs shape {self.coords_icrs.shape} "
                f"and freqs shape {self.freqs.shape}."
            )

        if len(self.brightness.shape) == 2:
            # Represnts I, rest are 0
            self.brightness = jnp.stack(
                [self.brightness.value,
                 jnp.zeros_like(self.brightness.value),  # Q
                 jnp.zeros_like(self.brightness.value),  # U
                 jnp.zeros_like(self.brightness.value)  # V
                 ],
                axis=-1
            ) * self.brightness.unit

        if len(self.brightness.shape) != 3:
            raise ValueError(f"Expected brightness to have 3 dimensions, got {self.brightness.shape}")

        if self.brightness.shape[2] != 4:
            raise ValueError(f"Expected brightness to have 4 stokes, got {self.brightness.shape[2]}")

    def get_angular_diameter(self) -> au.Quantity:
        if self.coords_icrs.isscalar or self.coords_icrs.shape[0] == 1:
            return au.Quantity(0, 'rad')

        # Get maximal separation
        return max(coord.separation(self.coords_icrs).max() for coord in self.coords_icrs).to('deg')

    def get_source_model_linear(self, freqs: jax.Array) -> jax.Array:

        # get freq interp indices
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(freqs, self.freqs)
        # Convert stokes to XX XY YX YY

        brightness = quantity_to_jnp(self.brightness, 'Jy')
        brightness = brightness[:, i0] * alpha0[:, None] + brightness[:, i1] * alpha1[:, None]

        shape = np.shape(brightness)[:-1]  # Shape without the coherency dimension
        brightness = brightness.reshape((-1, 4))  # [num_sources * num_freqs, 4]

        linear_coherencies = jax.vmap(partial(stokes_to_linear, flat_output=False))(
            brightness)  # [num_sources * num_freqs, 2, 2]

        linear_coherencies = lax.reshape(linear_coherencies, shape + (2, 2))
        return linear_coherencies

    def compute_lmn(self, phase_tracking: ac.ICRS, time: at.Time):
        """
        Compute the l and m coordinates of the sources.

        Args:
            phase_tracking: the pointing direction
            time: the time of the observation
        """
        return icrs_to_lmn(self.coords_icrs, time, phase_tracking)
