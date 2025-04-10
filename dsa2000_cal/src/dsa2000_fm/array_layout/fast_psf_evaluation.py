import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.mixed_precision_utils import mp_policy
from dsa2000_common.common.sum_utils import scan_sum

tfpd = tfp.distributions


# Compose PSF
def rotation_matrix_change_dec(delta_dec: FloatArray):
    """
    Get rotation matrix for changing DEC by delta_dec.

    Args:
        delta_dec: the change in DEC in radians

    Returns:
        R: the rotation matrix
    """
    # Rotate up or down changing DEC, but keeping RA constant.
    # Used for projecting ENU system
    c, s = jnp.cos(delta_dec), jnp.sin(delta_dec)
    R = jnp.asarray(
        [
            [1., 0., 0.],
            [0., c, -s],
            [0., s, c]
        ]
    )
    return R


def rotate_coords(antennas: FloatArray, dec_from: FloatArray, dec_to: FloatArray) -> FloatArray:
    """
    Rotate the antennas from one DEC to another DEC.

    Args:
        antennas: [..., 3]
        dec_from: the DEC to rotate from
        dec_to: the DEC to rotate to

    Returns:
        [..., 3] the rotated antennas
    """
    # East to east
    delta_dec = dec_to - dec_from
    east, north, up = antennas[..., 0], antennas[..., 1], antennas[..., 2]
    east_prime = east
    north_prime = jnp.cos(delta_dec) * north - jnp.sin(delta_dec) * up
    up_prime = jnp.sin(delta_dec) * north + jnp.cos(delta_dec) * up
    return jnp.stack([east_prime, north_prime, up_prime], axis=-1)


def deproject_antennas(antennas_projected: FloatArray, latitude: FloatArray, transit_dec: FloatArray) -> FloatArray:
    """
    Deproject the antennas from the projected coordinates.

    Args:
        antennas_projected: [..., 3]
        latitude: the latitude of the array
        transit_dec: the transit DEC of the array

    Returns:
        [..., 3] the deprojected antennas
    """
    antennas = rotate_coords(antennas_projected, transit_dec, latitude)
    # antennas = antennas.at[..., 2].set(0.)
    return antennas


def project_antennas(antennas: FloatArray, latitude: FloatArray, transit_dec: FloatArray) -> FloatArray:
    """
    Project the antennas to the projected coordinates.

    Args:
        antennas: [..., 3] the antennas in ENU at the latitude
        latitude: the latitude of the array
        transit_dec: the transit DEC of the array

    Returns:
        [..., 3] the projected antennas
    """
    antennas_projected = rotate_coords(antennas, latitude, transit_dec)
    # antennas_projected = antennas_projected.at[..., 2].set(0.)
    return antennas_projected


def compute_psf(antennas: FloatArray, lmn: FloatArray, freqs: FloatArray, latitude: FloatArray,
                transit_dec: FloatArray, with_autocorr: bool = True, accumulate_dtype=jnp.float32) -> FloatArray:
    """
    Compute the point spread function of the array. Uses short cut,

    B(l,m) = (sum_i e^(-i2pi (u_i l + v_i m)))^2/N^2

    To remove auto-correlations, there are N values of 1 to subtract from N^2 values, then divide by (N-1)N
    PSF(l,m) = (N^2 B(l,m) - N)/(N-1)/N = (N B(l,m) - 1)/(N-1) where B(l,m) in [0, 1].
    Thus the amount of negative is (-1/(N-1))

    Args:
        antennas: [N, 3] the antennas in ENU
        lmn: [..., 3] the lmn coordinates to evaluate the PSF
        freqs: [C] the frequency in Hz of each channel
        latitude: the latitude of the array in radians
        transit_dec: the transit DEC of the array in radians
        with_autocorr: whether to include the autocorrelation in the PSF
        accumulate_dtype: the dtype to use for accumulation

    Returns:
        psf: [...] the point spread function
    """

    def compute_psf_delta(freq):
        antennas_proj = project_antennas(antennas, latitude, transit_dec)
        wavelength = mp_policy.cast_to_length(299792458. / freq)
        r = antennas_proj / wavelength
        delay = (2 * jnp.pi) * (jnp.sum(r * lmn[..., None, :], axis=-1) - r[..., 2])  # [..., N]
        delay = delay.astype(accumulate_dtype)
        voltage_beam_real = jnp.cos(delay).mean(axis=-1)  # [...]
        voltage_beam_imag = jnp.sin(delay).mean(axis=-1)  # [...]
        # voltage_beam = jax.lax.complex(jnp.cos(delay), jnp.sin(delay))  # [..., N]
        # voltage_beam = jnp.mean(voltage_beam, axis=-1)  # [...]
        power_beam = jnp.square(voltage_beam_real) + jnp.square(voltage_beam_imag)  # [...]
        power_beam *= lmn[..., 2].astype(accumulate_dtype)
        if with_autocorr:
            delta = power_beam
        else:
            N = antennas_proj.shape[-2]
            N1 = jnp.asarray(N / (N - 1), accumulate_dtype)
            N2 = jnp.asarray(-1 / (N - 1), accumulate_dtype)
            # (N * power_beam - 1)/(N-1)
            delta = N1 * power_beam + N2
        return delta

    psf_accum = jnp.zeros(np.shape(lmn)[:-1], dtype=accumulate_dtype)
    psf = scan_sum(compute_psf_delta, psf_accum, freqs)
    return psf / np.shape(freqs)[0]  # average over frequencies
