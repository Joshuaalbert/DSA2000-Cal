import dataclasses
from functools import partial
from typing import Tuple, NamedTuple

import jax
import numpy as np
from astropy import constants
from jax import numpy as jnp, lax
from jax._src.typing import SupportsDType

from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords


class DFTModelData(NamedTuple):
    """
    Data for predict.
    """
    image: jax.Array  # [source, chan, 2, 2] in [[xx, xy], [yx, yy]] format
    gains: jax.Array  # [source, time, ant, chan, 2, 2]
    lmn: jax.Array  # [source, 3]


@dataclasses.dataclass(eq=False)
class DFTPredict:
    convention: str = 'fourier'
    dtype: SupportsDType = jnp.complex64

    def __post_init__(self):
        if self.convention not in ('fourier', 'casa'):
            raise ValueError("convention not in ('fourier', 'casa')")

    def predict(self, freqs: jax.Array, dft_model_data: DFTModelData, visibility_coords: VisibilityCoords) -> jax.Array:

        g1 = dft_model_data.gains[
             :, visibility_coords.time_idx, visibility_coords.antenna_1, :, :, :
             ]  # [source, row, chan, 2, 2]
        g2 = dft_model_data.gains[
             :, visibility_coords.time_idx, visibility_coords.antenna_2, :, :, :
             ]  # [source, row, chan, 2, 2]

        # Data will be sharded over frequency so don't reduce over these dimensions, or else communication happens.
        # We want the outer broadcast to be over chan, so we'll do this order:
        # chan -> source -> row
        # TODO: explore other orders

        # lmn: [source, 3]
        # uvw: [rows, 3]
        # g1, g2: [source, row, chan, 2, 2]
        # freq: [chan]
        # image: [source, chan, 2, 2]
        @partial(jax.vmap, in_axes=[None, None, 2, 2, 0, 1])
        # lmn: [source, 3]
        # uvw: [rows, 3]
        # g1, g2: [source, row, 2, 2]
        # freq: []
        # image: [source, 2, 2]
        @partial(jax.vmap, in_axes=[0, None, 0, 0, None, 0])
        # lmn: [3]
        # uvw: [rows, 3]
        # g1, g2: [row, 2, 2]
        # freq: []
        # image: [2, 2]
        @partial(jax.vmap, in_axes=[None, 0, 0, 0, None, None])
        # lmn: [3]
        # uvw: [3]
        # g1, g2: [2, 2]
        # freq: []
        # image: [2, 2]
        def compute_visibility(*args):
            return self._single_compute_visibilty(*args)

        visibilities = compute_visibility(
            dft_model_data.lmn,
            visibility_coords.uvw,
            g1,
            g2,
            freqs,
            dft_model_data.image
        )  # [chan, source, row, 2, 2]
        visibilities = jnp.sum(visibilities, axis=1)  # [chan, row, 2, 2]
        # make sure the output is [row, chan, 2, 2]
        return lax.transpose(visibilities, (1, 0, 2, 3))

    def _single_compute_visibilty(self, lmn, uvw, g1, g2, freq, image):
        """
        Compute the visibility from a single direction for a single baseline.

        Args:
            lmn: [3]
            uvw: [3]
            g1: [2, 2]
            g2: [2, 2]
            freq: []
            image: [2, 2]

        Returns:
            [2, 2] visibility in given direction for given baseline.
        """
        wavelength = freq / quantity_to_jnp(constants.c)
        if self.convention == 'fourier':
            constant = -2. * np.pi
        elif self.convention == 'casa':
            constant = 2. * np.pi
        else:
            raise ValueError("convention not in ('fourier', 'casa')")

        u, v, w = uvw  # scalar

        l, m, n = lmn  # scalar

        # -2*pi*freq/c*(l*u + m*v + (n-1)*w)
        delay = l * u + m * v + (n - 1.) * w  # scalar

        phi = jnp.asarray(
            1j * (delay * constant * wavelength),
            dtype=self.dtype
        )  # scalar
        fringe = (jnp.exp(phi) / n)
        # TODO: Might need to add complex conjugate here, uvw is hermitean
        return fringe * kron_product(g1, image, g2.T.conj())


def vec(a: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorize a matrix.

    Args:
        a: [n, m] array

    Returns:
        [n*m] array
    """
    if len(a.shape) != 2:
        raise ValueError(f"a should be a matrix, got shape {a.shape}")
    n, m = a.shape
    # a.T.ravel()
    return lax.reshape(a, (n * m,), (1, 0))


def unvec(a: jnp.ndarray, shape: Tuple[int, ...] | None = None) -> jnp.ndarray:
    """
    Unvectorize a matrix.

    Args:
        a: [n*m] array
        shape: shape of the unvectorized array

    Returns:
        [n, m] array
    """
    if shape is None:
        # assume square
        n = int(np.sqrt(a.shape[-1]))
        if n * n != a.shape[-1]:
            raise ValueError(f"a is not square. Can't infer unvec shape.")
        shape = (n, n)
    if len(shape) != 2:
        raise ValueError(f"shape should be length 2, got {len(shape)}")
    # jnp.reshape(a, shape).T
    return lax.transpose(lax.reshape(a, shape), (1, 0))


def kron(a, b):
    """
    Compute the Kronecker product of two arrays.

    Args:
        a: [n, m]
        b: [p, q]

    Returns:
        [n*p, m*q]
    """
    if len(np.shape(a)) < len(np.shape(b)):
        a = lax.expand_dims(a, range(np.ndim(b) - np.ndim(a)))
    elif np.ndim(b) < np.ndim(a):
        b = lax.expand_dims(b, range(np.ndim(a) - np.ndim(b)))
    a_reshaped = lax.expand_dims(a, range(1, 2 * np.ndim(a), 2))
    b_reshaped = lax.expand_dims(b, range(0, 2 * np.ndim(b), 2))
    out_shape = tuple(np.multiply(np.shape(a), np.shape(b)))
    return lax.reshape(lax.mul(a_reshaped, b_reshaped), out_shape)


def kron_product(a, b, c):
    # return unvec(kron(c.T, a) @ vec(b), (a.shape[0], c.shape[1]))
    # Fewer bytes accessed, better utilisation (2x as many flops though -- which is better than memory access)
    return unvec(jnp.sum(kron(c.T, a) * vec(b), axis=-1), (a.shape[0], c.shape[1]))
