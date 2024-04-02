from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax._src.typing import SupportsDType


@dataclass(eq=False)
class WProjKernel:
    """
    Class for constructing the W-projection kernel
    """
    convention: Literal['fourier', 'casa'] = 'fourier'
    dtype: SupportsDType = jnp.complex64

    def kernel(self, l: jnp.ndarray, m: jnp.ndarray, w: jnp.ndarray, freq: jnp.ndarray) -> jnp.ndarray:
        if self.convention == 'fourier':
            constant = minus_two_pi_over_c
        elif self.convention == 'casa':
            constant = two_pi_over_c
        else:
            raise ValueError("convention not in ('fourier', 'casa')")
        n = jnp.sqrt(1. - l ** 2 - m ** 2)
        phase = constant * freq * w * (n - 1.)
        return jnp.exp(jnp.asarray(1.0j * phase, self.dtype))
