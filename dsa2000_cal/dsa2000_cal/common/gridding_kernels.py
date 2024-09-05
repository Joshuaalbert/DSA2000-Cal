from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from astropy import constants
from jax._src.typing import SupportsDType

from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.types import complex_type


@dataclass(eq=False)
class WProjKernel:
    """
    Class for constructing the W-projection kernel
    """
    convention: Literal['physical', 'casa'] = 'physical'
    dtype: SupportsDType = complex_type

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
