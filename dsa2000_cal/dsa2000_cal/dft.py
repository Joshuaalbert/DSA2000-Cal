from typing import Literal

from africanus.dft import im_to_vis
from jax import numpy as jnp, vmap

from dsa2000_cal.jax_utils import chunked_pmap

# Lightspeed
c = 2.99792458e8

two_pi_over_c = 2 * jnp.pi / c
minus_two_pi_over_c = -two_pi_over_c

def im_to_vis(
        image: jnp.ndarray,
        uvw: jnp.ndarray,
        lm: jnp.ndarray,
        frequency: jnp.ndarray,
        convention: Literal['fourier', 'casa'] = 'fourier',
        dtype=jnp.complex64,
        chunksize: int = 1
) -> jnp.ndarray:
    """
    Convert an image to visibilities.

    Args:
        image: [source, chan, corr]
        uvw: [row, 3]
        lm: [source, 2]
        frequency: [chan]
        convention: 'fourier' or 'casa'
        dtype: dtype of output
        chunksize: chunksize for pmap

    Returns:
        visibilities: [row, chan, corr]
    """
    if convention == 'fourier':
        constant = minus_two_pi_over_c
    elif convention == 'casa':
        constant = two_pi_over_c
    else:
        raise ValueError("convention not in ('fourier', 'casa')")

    # We scan over UVW coordinates, and then compute output
    def body(uvw):
        u, v, w = uvw

        @vmap
        def sum_over_source(lm, im_chan_corr):
            l, m = lm
            n = jnp.sqrt(1. - l ** 2 - m ** 2)
            # -2*pi*(l*u + m*v + n*w)/c
            real_phase = constant * (l * u + m * v + (n - 1.) * w)  # [scalar]

            # Multiple in frequency for each channel
            @vmap
            def sum_over_channel(f, im_corr):
                p = jnp.asarray(real_phase * f * 1.0j, dtype=dtype)
                return im_corr * (jnp.exp(p) / n)  # [corr]

            return sum_over_channel(frequency, im_chan_corr)  # [chan, corr]

        return jnp.sum(sum_over_source(lm, image), axis=0)  # [chan, corr]

    vis = chunked_pmap(
        f=body,
        chunksize=chunksize,
        batch_size=uvw.shape[0]
    )(uvw)  # [row, chan, corr]

    return vis


def im_to_vis_with_gains(
        image: jnp.ndarray,
        gains: jnp.ndarray,
        antenna_1: jnp.ndarray,
        antenna_2: jnp.ndarray,
        uvw: jnp.ndarray,
        lm: jnp.ndarray,
        frequency: jnp.ndarray,
        convention: Literal['fourier', 'casa'] = 'fourier',
        dtype=jnp.complex64,
        chunksize: int = 1
) -> jnp.ndarray:
    """
    Convert an image to visibilities.

    Args:
        image: [source, chan, 2, 2]
        gains: [ant, source, chan, 2, 2]
        antenna_1: [row]
        antenna_2: [row]
        uvw: [row, 3]
        lm: [source, 2]
        frequency: [chan]
        convention: 'fourier' or 'casa'
        dtype: dtype of output
        chunksize: chunksize for pmap

    Returns:
        visibilities: [row, chan, 2, 2]
    """
    if convention == 'fourier':
        constant = minus_two_pi_over_c
    elif convention == 'casa':
        constant = two_pi_over_c
    else:
        raise ValueError("convention not in ('fourier', 'casa')")

    # We scan over UVW coordinates, and then compute output
    def body(uvw, antenna_1, antenna_2):
        u, v, w = uvw

        g1 = gains[antenna_1, :, :, :, :]  # [source, chan, 2, 2]
        g2 = gains[antenna_2, :, :, :, :]  # [source, chan, 2, 2]

        @vmap
        def sum_over_source(lm, im_chan_corr, g1_chan_corr, g2_chan_corr):
            l, m = lm
            n = jnp.sqrt(1. - l ** 2 - m ** 2)
            # -2*pi*(l*u + m*v + n*w)/c
            real_phase = constant * (l * u + m * v + (n - 1.) * w)  # [scalar]

            # Multiple in frequency for each channel
            @vmap
            def sum_over_channel(freq, im_corr, g1_corr, g2_corr):
                p = jnp.asarray((real_phase * freq) * 1.0j, dtype=dtype)
                vis_comp = jnp.einsum('ab,bc,dc->ad', g1_corr, im_corr, jnp.conj(g2_corr))  # [2,2]
                return vis_comp * (jnp.exp(p) / n)  # [2, 2]

            return sum_over_channel(frequency, im_chan_corr, g1_chan_corr, g2_chan_corr)  # [chan, 2, 2]

        return jnp.sum(sum_over_source(lm, image, g1, g2), axis=0)  # [chan, 2, 2]

    vis = chunked_pmap(
        f=body,
        chunksize=chunksize,
        batch_size=uvw.shape[0]
    )(uvw, antenna_1, antenna_2)  # [row, chan, 2, 2]

    return vis
