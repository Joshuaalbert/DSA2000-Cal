from typing import Literal

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
    Convert an image to visibilities for ideal interferometer. Assumes no instrumental effects.

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
        time_idx: jnp.ndarray,
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
        gains: [time, ant, source, chan, 2, 2]
        antenna_1: [row]
        antenna_2: [row]
        time_idx: [row]
        uvw: [row, 3]
        lm: [source, 2]
        frequency: [chan]
        convention: 'fourier' or 'casa'
        dtype: dtype of output
        chunksize: chunksize for pmap

    Returns:
        visibilities: [row, chan, 2, 2]
    """
    # Do shape checks
    check_shapes(antenna_1, antenna_2, frequency, gains, image, lm, time_idx, uvw)
    if convention == 'fourier':
        constant = minus_two_pi_over_c
    elif convention == 'casa':
        constant = two_pi_over_c
    else:
        raise ValueError("convention not in ('fourier', 'casa')")

    # We scan over UVW coordinates, and then compute output
    def body(uvw, antenna_1, antenna_2, time_idx):
        u, v, w = uvw

        g1 = gains[time_idx, antenna_1, :, :, :, :]  # [source, chan, 2, 2]
        g2 = gains[time_idx, antenna_2, :, :, :, :]  # [source, chan, 2, 2]

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
    )(uvw, antenna_1, antenna_2, time_idx)  # [row, chan, 2, 2]

    return vis


def check_shapes(antenna_1: jnp.ndarray, antenna_2: jnp.ndarray, frequency: jnp.ndarray, gains: jnp.ndarray,
                 image: jnp.ndarray, lm: jnp.ndarray, time_idx: jnp.ndarray, uvw: jnp.ndarray):
    """
    Check that the shapes of the inputs are correct.

    Args:
        antenna_1: [row]
        antenna_2: [row]
        frequency: [chan]
        gains: [time, ant, source, chan, 2, 2]
        image: [source, chan, 2, 2]
        lm: [source, 2]
        time_idx: [row]
        uvw: [row, 3]

    Raises:
        ValueError: if any of the shapes are incorrect.
    """
    # Check num dims
    if len(image.shape) != 4:
        raise ValueError("image must have shape [source, chan, 2, 2]")
    if len(gains.shape) != 6:
        raise ValueError("gains must have shape [time, ant, source, chan, 2, 2]")
    if len(antenna_1.shape) != 1:
        raise ValueError("antenna_1 must have shape [row]")
    if len(antenna_2.shape) != 1:
        raise ValueError("antenna_2 must have shape [row]")
    if len(time_idx.shape) != 1:
        raise ValueError("time_idx must have shape [row]")
    if len(uvw.shape) != 2:
        raise ValueError("uvw must have shape [row, 3]")
    if len(lm.shape) != 2:
        raise ValueError("lm must have shape [source, 2]")
    if len(frequency.shape) != 1:
        raise ValueError("frequency must have shape [chan]")
    # Check shapes
    if image.shape[2] != 2 or image.shape[3] != 2:
        raise ValueError(f"image must have shape [source, chan, 2, 2], got {image.shape}")
    if gains.shape[4] != 2 or gains.shape[5] != 2:
        raise ValueError(f"gains must have shape [time, ant, source, chan, 2, 2], got {gains.shape}")
    if antenna_1.shape[0] != antenna_2.shape[0]:
        raise ValueError(f"antenna_1 and antenna_2 must have same length, got {antenna_1.shape} and {antenna_2.shape}")
    if antenna_1.shape[0] != time_idx.shape[0]:
        raise ValueError(f"antenna_1 and time_idx must have same length, got {antenna_1.shape} and {time_idx.shape}")
    if uvw.shape[0] != antenna_1.shape[0]:
        raise ValueError(f"uvw and antenna_1 must have same length, got {uvw.shape} and {antenna_1.shape}")
    if lm.shape[0] != image.shape[0]:
        raise ValueError(f"lm and image must have same length, got {lm.shape} and {image.shape}")
