from dataclasses import dataclass
from typing import Literal

from astropy import constants

import jax.numpy as jnp
from jax._src.typing import SupportsDType

from dsa2000_cal.common.quantity_utils import quantity_to_jnp


@dataclass(eq=False)
class WProjKernel:
    """
    Class for constructing the W-projection kernel
    """
    convention: Literal['physical', 'casa'] = 'physical'
    dtype: SupportsDType = jnp.complex64

    def kernel(self, l: jnp.ndarray, m: jnp.ndarray, w: jnp.ndarray, freq: jnp.ndarray) -> jnp.ndarray:
        if self.convention == 'physical':
            constant = -2. * jnp.pi / quantity_to_jnp(constants.c)
        elif self.convention == 'casa':
            constant = 2. * jnp.pi / quantity_to_jnp(constants.c)
        else:
            raise ValueError("convention not in ('physical', 'casa')")
        n = jnp.sqrt(1. - l ** 2 - m ** 2)
        phase = constant * freq * w * (n - 1.)
        return jnp.exp(jnp.asarray(1.0j * phase, self.dtype))
