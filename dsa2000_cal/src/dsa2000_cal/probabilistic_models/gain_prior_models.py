from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, NamedTuple

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jaxctx import transform
from jaxctx.priors.prior import Prior

from dsa2000_common.common.array_types import FloatArray, ComplexArray, Array

tfpd = tfp.distributions

GainType = TypeVar('GainType')


class AbstractGainPriorModel(ABC, Generic[GainType]):

    def compute_gains(self, key, params: Any, ra: FloatArray,
                      dec: FloatArray,
                      antennas_gcrs: FloatArray,
                      freqs: FloatArray,
                      time: FloatArray) -> GainType:
        prior_model = transform(self.build_prior_model)
        apply_result = prior_model.apply({'params': key}, collections=params, ra=ra, dec=dec,
                                         antennas_gcrs=antennas_gcrs, freqs=freqs, time=time)
        return apply_result.fn_val

    def get_init_params(self, key, ra: FloatArray,
                        dec: FloatArray,
                        antennas_gcrs: FloatArray,
                        freqs: FloatArray,
                        time: FloatArray) -> Any:
        prior_model = transform(self.build_prior_model)
        init_result = prior_model.init(key, None, ra=ra, dec=dec,
                                       antennas_gcrs=antennas_gcrs, freqs=freqs, time=time)
        return init_result.collections

    @abstractmethod
    def build_prior_model(
            self,
            ra: FloatArray,
            dec: FloatArray,
            antennas_gcrs: FloatArray,
            freqs: FloatArray,
            time: FloatArray
    ) -> GainType:
        """
        Define the prior model for the gains.

        Args:
            ra: [D] Right Ascension of the sources in radians.
            dec: [D] Declination of the sources in radians.
            antennas_gcrs: [A, 3] Antenna positions in GCRS
            freqs: [C] Frequencies in Hz.
            time: time in TT since start of observation.

        Returns:
            some gain type which can be used to apply the gains to the data.
        """
        ...


class LNAGains(NamedTuple):
    # G, LNA + electronics gains
    lna_amp: FloatArray  # [A] # scalar, LNA + electronics, slow amplitude and slow phase
    lna_phase: FloatArray  # [A] # scalar, LNA + electronics, slow amplitude and slow phase


class ParallacticAngleGains(NamedTuple):
    # P, Parallactic angle gains
    parallactic_angle: FloatArray  # [A] # scalar, parallactic angle, slow over time


class LeakageGains(NamedTuple):
    # D, Polarisation leakage gains
    model_freqs: FloatArray  # [Cm] # frequencies at which the leakage is defined
    leakage_amp: FloatArray  # [A, Cm, 2] # cross-diagonal, freq poly, amplitude + delay
    leakage_phase: FloatArray  # [A, Cm, 2] # cross-diagonal, freq poly, amplitude + delay


class BandpassGains(NamedTuple):
    # B, Bandpass gains
    model_freqs: FloatArray  # [Cm] # frequencies at which the bandpass is defined
    bandpass_amp: FloatArray  # [A, Cm, 2] # diagonal, freq poly, amplitude + delay
    bandpass_delay: FloatArray  # [A, Cm, 2] # diagonal, freq poly, amplitude + delay


class RotationMeasureGains(NamedTuple):
    # R, DD Stokes I gains
    rotation_measure: FloatArray  # [D, A] # scalar, freq model, rotation measure


class TECGains(NamedTuple):
    # Z, DD Stokes V gains
    tec: ComplexArray  # [D, A] # scalar, freq model, TEC


class Gains(NamedTuple):
    # Model: G @ B @ D @ E @ P @ R @ Z @ K = G * Z (B @ D @ E @ P @ R @ K)
    lna_gains: LNAGains | None
    parallactic_angle_gains: ParallacticAngleGains | None
    leakage_gains: LeakageGains | None
    bandpass_gains: BandpassGains | None
    rotation_measure_gains: RotationMeasureGains | None
    tec_gains: TECGains | None


class PriorHyperParameters(NamedTuple):
    lna_amp_stddev: FloatArray = 0.1
    lna_phase_stddev_rad: FloatArray = 1.
    parallactic_offset_stddev_rad: FloatArray = 1e-2
    leakage_phase_stddev_rad: FloatArray = 0.1
    leakage_amp_max: FloatArray = 0.2
    rotation_measure_stddev_rad_m2: FloatArray = 0.2
    bandpass_amplitude_stddev: FloatArray = 0.5
    feed_delay_stddev_ns: FloatArray = 1.
    tec_stddev_mtecu: FloatArray = 200.0


def _quadratic_interpolation(x, xp, yp):
    """
    Perform quadratic interpolation to find the value at x given points xp and yp.

    Args:
        x: The point at which to evaluate the interpolation.
        xp: [3] The x-coordinates of the known points.
        yp: [3] The y-coordinates of the known points.

    Returns:
        The interpolated value at x.
    """
    if np.shape(xp) != (3,):
        raise ValueError(f"Expected xp to be of shape (3,), got {np.shape(xp)}")
    if np.shape(yp) != (3,):
        raise ValueError(f"Expected yp to be of shape (3,), got {np.shape(yp)}")

    x1, x2, x3 = xp[0], xp[1], xp[2]
    y1, y2, y3 = yp[0], yp[1], yp[2]

    # Pre-compute the constant denominators of the Lagrange basis
    d1 = (x1 - x2) * (x1 - x3)
    d2 = (x2 - x1) * (x2 - x3)
    d3 = (x3 - x1) * (x3 - x2)

    # Lagrange basis polynomials L0, L1, L2 evaluated at x
    L0 = ((x - x2) * (x - x3)) / d1
    L1 = ((x - x1) * (x - x3)) / d2
    L2 = ((x - x1) * (x - x2)) / d3

    return y1 * L0 + y2 * L1 + y3 * L2


def quadratic_interpolation(x, xp, yp, axis):
    """
    Perform quadratic interpolation to find the value at x given points xp and yp.

    Args:
        x: [N] The point(s) at which to evaluate the interpolation.
        xp: [3] The x-coordinates of the known points.
        yp: [..., 3, ...] The y-coordinates of the known points, where `axis` axis is the values to interpolate.
        axis: the axis along which to interpolate.

    Returns:
        [..., N, ...] The interpolated values at x.
    """
    move_axis = False
    if axis != -1:
        yp = jnp.moveaxis(yp, axis, -1)
        move_axis = True

    # flatten to (-1, 3)
    shape = np.shape(yp)
    yp = jnp.reshape(yp, (-1, 3))  # [-1, 3]
    # two vmap's one over N and one over [-1]
    y = jax.vmap(jax.vmap(_quadratic_interpolation, in_axes=(0, None, None)), in_axes=(None, None, 0))(x, xp,
                                                                                                       yp)  # [-1, N
    y = jnp.reshape(y, shape[:-1] + (-1,))  # [..., N]
    if move_axis:
        y = jnp.moveaxis(y, -1, axis)
    return y


def test_quadratic_interpolation():
    x = jnp.array([0, 1, 2])
    xp = jnp.array([0.0, 1.0, 2.0])
    yp = jnp.array([1.0, 2.0, 3.0])

    result = quadratic_interpolation(x, xp, yp, axis=-1)
    print("Interpolated values:", result)
    np.testing.assert_allclose(result, yp)

    # test with batch
    yp_batch = jnp.arange(5 * 4 * 3).reshape((5, 3, 4))
    result_batch = quadratic_interpolation(x, xp, yp_batch, axis=1)
    print("Batch interpolated values:", result_batch)
    np.testing.assert_allclose(result_batch, yp_batch)


def set_diagonal_scalar(values: Array) -> Array:
    """
    Sets the values on diagonal.

    Args:
        values: [...]

    Returns:
        [..., 2, 2] The values on the diagonal.
    """
    eye = jnp.eye(2, dtype=values.dtype)
    return values[..., None, None] * eye


def set_diagonal(values: Array) -> Array:
    """
    Sets the values on diagonal.

    Args:
        values: [..., 2]

    Returns:
        [..., 2, 2] The values on the diagonal.
    """
    if np.shape(values)[-1:] != (2,):
        raise ValueError(f"Expected last dimension of values to be 2, got {np.shape(values)}")
    eye = jnp.eye(2, dtype=values.dtype)
    return values[..., :, None] * eye


def set_cross(values: Array) -> Array:
    """
    Sets the values on cross.

    Args:
        values: [..., 2]

    Returns:
        [..., 2, 2] The values on the cross.
    """
    if np.shape(values)[-1:] != (2,):
        raise ValueError(f"Expected last dimension of values to be 2, got {np.shape(values)}")
    # Off-diagonal mask [[0,1],[1,0]] – broadcast over leading dims
    cross_mask = jnp.array([[0, 1],
                            [1, 0]], dtype=values.dtype)

    return values[..., :, None] * cross_mask


def test_set_diagonal_and_cross():
    a = jnp.array([1, 2])
    np.testing.assert_allclose(set_diagonal(a), jnp.array([[1, 0], [0, 2]]))
    np.testing.assert_allclose(set_cross(a), jnp.array([[0, 1], [2, 0]]))


def compute_parallactic_angle(hour_angle_rad, latitude_rad, declination_rad) -> FloatArray:
    """
    Compute the parallactic angle.

    Args:
        hour_angle_rad: [T] Hour angle in radians.
        latitude_rad: [A] Latitude in radians.
        declination_rad: [D] Declination in radians.

    Returns:
        [T, A, D] Parallactic angle in radians.
    """
    return jnp.arctan2(
        jnp.sin(hour_angle_rad),
        jnp.cos(declination_rad) * jnp.tan(latitude_rad) - jnp.sin(declination_rad) * jnp.cos(hour_angle_rad)
    )


def build_faraday_model(num_source, num_ant, freqs, times, rotation_measure_stddev_rad_m2):
    # Ionosphere TEC and faraday rotation measure, slow over time and frequency
    T = len(times)
    C = len(freqs)
    model_times = times[jnp.array([0, T // 2, -1])]  # [Tm]
    Tm = len(model_times)
    rotation_measure_ones = jnp.ones((num_source, Tm, num_ant, 1), dtype=jnp.float32)  # [D, Tm, A, 1]
    rotation_measure_model = Prior(
        tfpd.Normal(
            loc=jnp.zeros_like(rotation_measure_ones),
            scale=rotation_measure_ones * rotation_measure_stddev_rad_m2,
        ),
        name='rotation_measure_model'
    ).parameter(random_init=True)
    rotation_measure = quadratic_interpolation(
        x=times,
        xp=model_times,
        yp=rotation_measure_model,
        axis=1
    )  # [D, T, A, 1]
    c = 299792458.  # speed of light in m/s
    wavelengths = c / freqs  # [C]
    faraday_phase = rotation_measure * wavelengths ** 2  # [D, T, A, C]
    faraday_phase = jnp.stack([
        jnp.cos(faraday_phase), jnp.sin(faraday_phase),
        -jnp.sin(faraday_phase), jnp.cos(faraday_phase)
    ], axis=-1).reshape((num_source, T, num_ant, C, 2, 2))  # [D, T, A, C, 2, 2]
    faraday_gains = jax.lax.complex(
        jnp.cos(faraday_phase), jnp.sin(faraday_phase)
    )
    return faraday_gains


def build_tec_model(num_source, num_ant, times, tec_stddev_mtecu):
    # Ionosphere TEC and faraday rotation measure, slow over time and frequency
    tec_ones = jnp.ones((num_source, num_ant), dtype=jnp.float32)  # [D, Tm, A, 1]
    tec_model = Prior(
        tfpd.Normal(
            loc=jnp.zeros_like(tec_ones),
            scale=tec_ones * tec_stddev_mtecu,
        ),
        name='tec_model'
    ).parameter(random_init=True)

    return tec_model  # [D, T, A, C, 2, 2]


def build_leakage_model(num_ant, freqs, leakage_phase_stddev_rad, leakage_amp_max):
    C = len(freqs)
    # Polarisation leakage gains
    model_freqs = freqs[jnp.array([0, C // 2, -1])]  # [Cm]
    Cm = len(model_freqs)
    leakage_ones = jnp.ones((num_ant, Cm, 2), dtype=jnp.float32)  # [1, 1, A, Cm, 2]
    leakage_amp_model = Prior(
        tfpd.Uniform(
            low=jnp.zeros_like(leakage_ones),
            high=leakage_amp_max * leakage_ones
        ),
        name='leakage_amplitude_model'
    ).parameter(random_init=True)
    leakage_amp = jnp.exp(quadratic_interpolation(
        x=freqs,
        xp=model_freqs,
        yp=jnp.log(leakage_amp_model),
        axis=-2
    ))  # [1, 1, A, C, 2]
    leakage_phase_model = Prior(
        tfpd.Normal(
            loc=jnp.zeros_like(leakage_ones),
            scale=leakage_phase_stddev_rad * leakage_ones
        ),
        name='leakage_phase_model'
    ).parameter(random_init=True)
    leakage_phase = quadratic_interpolation(
        x=freqs,
        xp=model_freqs,
        yp=leakage_phase_model,
        axis=-2
    )  # [1, 1, A, C, 2]
    return LeakageGains(
        model_freqs=model_freqs,  # [Cm]
        leakage_amp=leakage_amp,
        leakage_phase=leakage_phase
    )


def build_parallactic_model(A, declination_rad, hour_angle_rad, latitude_rad, parallactic_offset_stddev_rad):
    # parallactic angle
    parallactic_angle = compute_parallactic_angle(
        hour_angle_rad=hour_angle_rad,
        latitude_rad=latitude_rad,
        declination_rad=declination_rad
    )
    parallactic_ones = jnp.ones((A,), dtype=jnp.float32)  # [A]
    parallactic_offset = Prior(
        tfpd.Normal(
            loc=jnp.zeros_like(parallactic_ones),
            scale=parallactic_ones * parallactic_offset_stddev_rad,
        ),
        name='parallactic_offset'
    ).parameter(random_init=True)
    parallactic_angle = parallactic_angle + parallactic_offset  # [A]
    # parallactic_gain = jnp.stack(
    #     [
    #         jnp.cos(parallactic_angle), jnp.sin(parallactic_angle),
    #         -jnp.sin(parallactic_angle), jnp.cos(parallactic_angle)
    #     ],
    #     axis=-1).reshape((A, 2, 2))  # [A, 2, 2]
    return ParallacticAngleGains(
        parallactic_angle=parallactic_angle
    )


def build_bandpass_model(freqs, num_ant, bandpass_amplitude_mean, bandpass_amplitude_stddev,
                         feed_delay_mean_ns, feed_delay_stddev_ns):
    C = np.shape(freqs)[0]
    if C <= 4:
        model_freqs = freqs[[0, -1]]
    else:
        model_freqs = freqs[jnp.array([0, C // 2, -1])]
    Cm = len(model_freqs)
    # Bandpass [T, A, C, 2] smooth over freqs, and time.
    bandpass_ones = jnp.ones((num_ant, Cm, 2), dtype=jnp.float32)
    bandpass_amplitude_model = Prior(
        tfpd.LogNormal.experimental_from_mean_variance(
            mean=bandpass_ones * bandpass_amplitude_mean,
            variance=bandpass_ones * bandpass_amplitude_stddev ** 2
        ),
        name='bandpass_amplitude_model'
    ).parameter(random_init=True)
    bandpass_amplitude = quadratic_interpolation(
        x=freqs,
        xp=model_freqs,
        yp=bandpass_amplitude_model,
        axis=1
    )

    # Add per feed delay
    feed_delay_ones = jnp.ones((num_ant, 2), dtype=jnp.float32)  # [A, 1, 2]
    feed_delay_ns = Prior(
        tfpd.Normal(
            loc=feed_delay_ones * feed_delay_mean_ns,
            scale=feed_delay_ones * feed_delay_stddev_ns
        ),
        name='feed_delay_ns'
    ).parameter(random_init=True)

    return BandpassGains(
        model_freqs=model_freqs,  # [Cm]
        bandpass_amp=bandpass_amplitude,  # [A, Cm, 2]
        bandpass_delay=feed_delay_ns  # [A, Cm, 2]
    )


def build_lna_model(num_ant, lna_amp_mean, lna_amp_stddev, lna_phase_mean_rad, lna_phase_stddev_rad):
    g_ones = jnp.ones((num_ant,))
    G_amp = Prior(
        tfpd.LogNormal.experimental_from_mean_variance(
            mean=g_ones * lna_amp_mean,
            variance=g_ones * lna_amp_stddev ** 2
        ),
        name="G_amp"
    ).parameter(random_init=True)

    G_phase = Prior(
        tfpd.Normal(
            loc=g_ones * lna_phase_mean_rad,
            scale=g_ones * lna_phase_stddev_rad
        ),
        name="G_phase"
    ).parameter(random_init=True)

    return LNAGains(
        lna_amp=G_amp,  # [A]
        lna_phase=G_phase  # [A]
    )
