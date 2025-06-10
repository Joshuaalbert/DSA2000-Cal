import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Generic, TypeVar, NamedTuple

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jaxctx import transform
from jaxctx.priors.prior import Prior

from dsa2000_common.common.array_types import FloatArray, ComplexArray, Array
from dsa2000_common.common.pytree import Pytree

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


class DDStokesIGains(NamedTuple):
    G_amp: FloatArray  # [A] # scalar, LNA + electronics, slow amplitude and slow phase
    G_phase: FloatArray  # [A] # scalar, LNA + electronics, slow amplitude and slow phase
    B_amp: FloatArray  # [A, Cm, 2] # diagonal, freq poly, amplitude + delay
    B_delay: FloatArray  # [A, 2] # diagonal, freq poly, amplitude + delay
    tec: ComplexArray  # [D, A] # scalar, freq model, TEC
    gains: ComplexArray  # [D, A, C, 2, 2] fully constructed gains


@dataclasses.dataclass(eq=False)
class DDStokesIGainPriorModel(Pytree, AbstractGainPriorModel[DDStokesIGains]):
    """
    Solves for Stokes-I direction dependent effects.

    Assumes the model visibilities include all these terms: GBDEPK,

    where:

        - G: scalar (LNA + electronics: slow amplitude and slow phase)
        - B: smooth amplitude + per feed delay
        - D: leakage and cross-polarisation terms
        - E: PB model
        - P: parallactic angle + offsets
        - K: geometric delay

    That is all these things have been calibrated on data and included in the model  visibilities.

    The next term that is solved is TEC (Z), which is scalar and direction dependent.

    Z(GBDEPK)
    """
    hour_angle_rad: FloatArray
    latitude_rad: FloatArray
    declination_rad: FloatArray
    lna_amp_stddev: FloatArray
    lna_phase_stddev_rad: FloatArray
    parallactic_offset_stddev_rad: FloatArray
    leakage_phase_stddev_rad: FloatArray
    leakage_amp_max: FloatArray
    rotation_measure_stddev_rad_m2: FloatArray
    bandpass_amplitude_stddev: FloatArray
    feed_delay_stddev_ns: FloatArray
    tec_stddev_mtecu: FloatArray
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return

    @classmethod
    def flatten(cls, this: 'DDStokesIGainPriorModel') -> Tuple[List[Any], Tuple[Any, ...]]:
        return (
            [

            ],
            (


            )
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> 'DDStokesIGainPriorModel':
        [] = children
        () = aux_data
        return DDStokesIGainPriorModel(
            skip_post_init=True
        )

    def build_prior_model(
            self,
            ra: FloatArray,
            dec: FloatArray,
            antennas_gcrs: FloatArray,
            freqs: FloatArray,
            time: FloatArray
    ) -> DDStokesIGains:
        terms = []

        num_ant = np.shape(antennas_gcrs)[0]  # number of antennas

        # G - scalar (LNA + electronics: slow amplitude and slow phase)
        lna_gains = build_lna_model(num_ant, self.lna_amp_stddev, self.lna_phase_stddev_rad, freqs,
                                    time)  # [1, 1, A, C, 2, 2]
        terms.append(lna_gains)

        # B - smooth amplitude + per feed delay
        bandpass_gains = build_bandpass_model(freqs, num_ant, bandpass_amplitude_stddev=self.bandpass_amplitude_stddev,
                                              feed_delay_stddev_ns=self.feed_delay_stddev_ns)  # [1, 1, A, C, 2, 2]
        terms.append(bandpass_gains)

        # D - leakage and Faraday rotation
        leakage_gain = build_leakage_model(
            num_ant, freqs,
            leakage_phase_stddev_rad=self.leakage_phase_stddev_rad,
            leakage_amp_max=self.leakage_amp_max
        )  # [1, 1, A, C, 2, 2]
        terms.append(leakage_gain)

        tec_gains = build_tec_model(num_source, num_ant, freqs, times, self.rotation_measure_stddev_rad_m2,
                                    tec_stddev_mtecu)
        terms.append(tec_gains)

        # Combine all terms with matrix multiplication
        output = terms[0]
        for term in terms[1:]:
            output = output @ term  # Matrix multiplication
        return output


DDStokesIGainPriorModel.register_pytree()


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


def build_tec_model(num_source, num_ant, freqs, times, rotation_measure_stddev_rad_m2, tec_stddev_mtecu):
    # Ionosphere TEC and faraday rotation measure, slow over time and frequency
    T = len(times)
    C = len(freqs)
    model_times = times[jnp.array([0, T // 2, -1])]  # [Tm]
    Tm = len(model_times)

    tec_ones = jnp.ones((num_source, Tm, num_ant, 1), dtype=jnp.float32)  # [D, Tm, A, 1]
    tec_model = Prior(
        tfpd.Normal(
            loc=jnp.zeros_like(tec_ones),
            scale=tec_ones * tec_stddev_mtecu,
        ),
        name='tec_model'
    ).parameter(random_init=True)

    tec = quadratic_interpolation(
        x=times,
        xp=model_times,
        yp=tec_model,
        axis=1
    )  # [D, T, A, 1]
    tec_conv = -8.4479745e6 / freqs  # rad / mTECU
    tec_phase = tec_conv * tec  # [D, T, A, C]
    tec_gains = set_diagonal_scalar(jax.lax.complex(jnp.cos(tec_phase), jnp.sin(tec_phase)))  # [D, T, A, C, 2, 2]
    return tec_gains  # [D, T, A, C, 2, 2]


def build_leakage_model(num_ant, freqs, leakage_phase_stddev_rad, leakage_amp_max):
    C = len(freqs)
    # Polarisation leakage gains
    model_freqs = freqs[jnp.array([0, C // 2, -1])]  # [Cm]
    Cm = len(model_freqs)
    leakage_ones = jnp.ones((1, 1, num_ant, Cm, 2), dtype=jnp.float32)  # [1, 1, A, Cm, 2]
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
    leakage_gain = leakage_amp * jax.lax.complex(jnp.cos(leakage_phase), jnp.sin(leakage_phase))  # [1, 1, A, C, 2]
    leakage_gain = jnp.eye(2) + set_cross(leakage_gain)  # [1, 1, A, C, 2, 2]
    return leakage_gain


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
    parallactic_gain = jnp.stack(
        [
            jnp.cos(parallactic_angle), jnp.sin(parallactic_angle),
            -jnp.sin(parallactic_angle), jnp.cos(parallactic_angle)
        ],
        axis=-1).reshape((A, 2, 2))  # [A, 2, 2]
    parallactic_gain = parallactic_gain[None, None, :, None, :, :]  # [1, 1, A, 1, 2, 2]
    return parallactic_gain


def build_bandpass_model(freqs, num_ant, bandpass_amplitude_mean, bandpass_amplitude_stddev, feed_delay_mean_ns,
                         feed_delay_stddev_ns):
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
    phase_conv = (2 * jnp.pi * 1e-9) * freqs  # rad / ns # [C]
    feed_delay_ones = jnp.ones((num_ant, 1, 2), dtype=jnp.float32)  # [A, 1, 2]
    feed_delay_ns = Prior(
        tfpd.Normal(
            loc=feed_delay_ones * feed_delay_mean_ns,
            scale=feed_delay_ones * feed_delay_stddev_ns
        ),
        name='feed_delay_ns'
    ).parameter(random_init=True)
    feed_phase = phase_conv[:, None] * feed_delay_ns  # [A, C, 2]
    feed_gains = jax.lax.complex(jnp.cos(feed_phase), jnp.sin(feed_phase))  # [1, 1, A, C, 2]
    feed_gains = set_diagonal(feed_gains)  # [1, 1, A, C, 2, 2]
    gains = bandpass_amplitude * feed_gains  # [1, 1, A, C, 2, 2]
    return gains


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

    G = G_amp * jax.lax.complex(
        jnp.cos(G_phase), jnp.sin(G_phase)
    )

    return G


def test_gain_prior_model():
    num_source = 2
    num_ant = 10
    freqs = jnp.linspace(700e6, 800e6, 5)
    times = jnp.linspace(0., 6, 3)

    transformed = transform(gain_prior_model)
    kwargs = dict(
        num_source=num_source,
        num_ant=num_ant,
        freqs=freqs,
        times=times,
        hour_angle_rad=jnp.asarray(0.),  # [1]
        latitude_rad=jnp.array(0.5),  # [A]
        declination_rad=jnp.array(0.1),  # [D]
        lna_amp_stddev=0.1,
        lna_phase_stddev_rad=0.1,
        parallactic_offset_stddev_rad=1e-2,
        leakage_phase_stddev_rad=0.1,
        leakage_amp_max=0.2,
        rotation_measure_stddev_rad_m2=0.2,
        tec_stddev_mtecu=200.0,
        bandpass_amplitude_stddev=0.5,
        feed_delay_stddev_ns=0.1,
    )
    params = transformed.init({'params': jax.random.PRNGKey(0)}, None, **kwargs).collections
    num_params = sum(jax.tree.map(np.size, jax.tree.leaves(params)))
    print(f"Number of parameters: {num_params}")

    gains = transformed.apply({'params': jax.random.PRNGKey(0)}, params, **kwargs).fn_val
