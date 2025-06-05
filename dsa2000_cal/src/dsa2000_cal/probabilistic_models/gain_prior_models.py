import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jaxctx import transform
from jaxctx.priors.prior import Prior

from dsa2000_common.common.array_types import FloatArray, ComplexArray, Array

tfpd = tfp.distributions


#
# class AbstractGainPriorModel(Pytree, ABC):
#
#     @abstractmethod
#     def gain_shape(self):
#         ...
#
#     @abstractmethod
#     def get_spec(self, freq_spec, time_spec) -> 'GainPriorModel':
#         ...
#
#     @abstractmethod
#     def compute_gains(self, params: Any):
#         ...
#
#     @abstractmethod
#     def get_init_params(self, key) -> Any:
#         ...
#
#     @abstractmethod
#     def build_prior_model(self, num_source: int, num_ant: int, freqs: jax.Array, times: jax.Array) -> PriorModelType:
#         """
#         Define the prior model for the gains.
#
#         Args:
#             num_source: the number of sources
#             num_ant: the number of antennas
#             freqs: [num_chan] the frequencies
#             times: [num_time] the times to compute the model data, in TT since start of observation
#
#         Returns:
#             gains: [num_source, num_time, num_ant, num_chan, [, 2, 2]].
#         """
#         ...
#
#
# @dataclasses.dataclass(eq=False)
# class GainPriorModel(AbstractGainPriorModel):
#     """
#     A gain model with unconstrained complex Gaussian priors with zero mean.
#     """
#     num_source: int
#     num_ant: int
#     freqs: FloatArray  # [Cs]
#     times: FloatArray  # [Ts]
#
#     full_stokes: bool = True
#
#     gain_stddev: float = 2.
#     max_clock_ns: float = 2
#     max_dtec_mtecu: float = 200
#
#     dd_type: str = 'unconstrained'  # Combine components with +
#     dd_dof: int = 4
#     double_differential: bool = True
#     di_dof: int = 4
#     di_type: str = 'unconstrained'  # Combine components with +
#
#     skip_post_init: bool = False
#
#     def __post_init__(self):
#         if self.skip_post_init:
#             return
#
#         if not self.full_stokes:
#             if self.di_dof != 1:
#                 raise ValueError('di_dof must be 1 if full stokes')
#             if self.dd_dof != 1:
#                 raise ValueError('dd_dof must be 1 if full stokes')
#         if len(np.shape(self.freqs)) != 1:
#             raise ValueError('freqs must be a 1D array')
#         if len(np.shape(self.times)) != 1:
#             raise ValueError('times must be a 1D array')
#
#     @classmethod
#     def flatten(cls, this: 'GainPriorModel') -> Tuple[List[Any], Tuple[Any, ...]]:
#         return (
#             [
#                 this.freqs, this.times
#             ],
#             (
#                 this.num_source, this.num_ant, this.full_stokes,
#                 this.gain_stddev, this.max_dtec_mtecu, this.max_clock_ns,
#                 this.dd_type, this.dd_dof, this.double_differential, this.di_dof, this.di_type,
#
#             )
#         )
#
#     @classmethod
#     def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> 'GainPriorModel':
#         [freqs, times] = children
#         (num_source, num_ant, full_stokes,
#          gain_stddev, max_dtec_mtecu, max_clock_ns,
#          dd_type, dd_dof, double_differential, di_dof, di_type) = aux_data
#         return GainPriorModel(
#             num_source=num_source,
#             num_ant=num_ant,
#             freqs=freqs,
#             times=times,
#             full_stokes=full_stokes,
#             gain_stddev=gain_stddev,
#             max_clock_ns=max_clock_ns,
#             max_dtec_mtecu=max_dtec_mtecu,
#             dd_type=dd_type,
#             dd_dof=dd_dof,
#             double_differential=double_differential,
#             di_dof=di_dof,
#             di_type=di_type,
#             skip_post_init=True
#         )
#
#     def get_spec(self, freq_spec, time_spec) -> 'GainPriorModel':
#         [freqs, times], aux_data = GainPriorModel.flatten(self)
#         return GainPriorModel.unflatten(aux_data, [freq_spec, time_spec])
#
#     def gain_shape(self):
#         if self.full_stokes:
#             return (self.num_source, len(self.times), self.num_ant, len(self.freqs), 2, 2)
#         else:
#             return (self.num_source, len(self.times), self.num_ant, len(self.freqs))
#
#     def compute_gains(self, params: Any):
#         def transform():
#             prior_model = self.build_prior_model(
#                 num_source=self.num_source,
#                 num_ant=self.num_ant,
#                 freqs=self.freqs,
#                 times=self.times
#             )
#             (gains,), _ = simulate_prior_model(jax.random.PRNGKey(0), prior_model)  # [D, Tm, A, Cm[,2,2]]
#             return gains
#
#         get_gains_transformed = ctx.transform(transform)
#         gains = get_gains_transformed.apply(params, jax.random.PRNGKey(0)).fn_val
#         return gains
#
#     def get_init_params(self, key) -> Any:
#         def transform():
#             prior_model = self.build_prior_model(
#                 num_source=self.num_source,
#                 num_ant=self.num_ant,
#                 freqs=self.freqs,
#                 times=self.times
#             )
#             (gains,), _ = simulate_prior_model(jax.random.PRNGKey(0), prior_model)  # [D, Tm, A, Cm[,2,2]]
#             return gains
#
#         get_gains_transformed = ctx.transform(transform)
#         return get_gains_transformed.init(key).params
#
#     def _make_gains_model_unconstrained(self, ones, name: str):
#         scale = self.gain_stddev * ones
#         gains_real = yield Prior(
#             tfpd.Normal(loc=ones,
#                         scale=scale
#                         ),
#             name=f'{name}_real'
#         ).parametrised()
#         gains_imag = yield Prior(
#             tfpd.Normal(loc=jnp.zeros_like(ones),
#                         scale=scale
#                         ),
#             name=f'{name}_imag'
#         ).parametrised()
#         gains = jax.lax.complex(gains_real, gains_imag)
#         return gains
#
#     def _make_gains_model_phase(self, ones, name: str):
#         gains_phase = yield Prior(
#             tfpd.Uniform(
#                 low=-jnp.pi * ones,
#                 high=jnp.pi * ones
#             ),
#             name=f'{name}_phase'
#         ).parametrised()
#         return gains_phase
#
#     def _broadcasted_freqs(self, dof: int):
#         if dof == 1:
#             return self.freqs
#         elif dof == 2:
#             return self.freqs[..., None]
#         elif dof == 4:
#             return self.freqs[..., None, None]
#         else:
#             raise ValueError('Invalid dof')
#
#     def _make_gains_model_clock(self, ones, name: str, dof: int):
#         # clock is in ns
#         clock = yield Prior(
#             tfpd.Uniform(
#                 low=-self.max_clock_ns * ones,
#                 high=self.max_clock_ns * ones
#             ),
#             name=f'{name}_clock'
#         ).parametrised()
#         phase_conv = (2 * jnp.pi * 1e-9) * self._broadcasted_freqs(dof=dof)
#         phase = phase_conv * clock
#         return phase
#
#     def _make_gains_model_dtec(self, ones, name: str, dof: int):
#         # dtec is in mtecu
#         dtec = yield Prior(
#             tfpd.Uniform(
#                 low=-self.max_dtec_mtecu * ones,
#                 high=self.max_dtec_mtecu * ones
#             ),
#             name=f'{name}_dtec'
#         ).parametrised()
#         # TEC_CONV = -8.4479745 * au.rad * au.MHz  # rad * MHz / mTECU
#         dtec_conv = (-8.4479745 * 1e6) / self._broadcasted_freqs(dof=dof)
#         phase = dtec_conv * dtec
#         return phase
#
#     def _make_gains_model_amplitude(self, ones, name: str):
#         # Rice distribution for X ~ N[1, sigma^2], Y ~ U[0, sigma^2] then R^2 = X^2 + Y^2 ~ Rice(1, sigma^2)
#         # We use noncentral chi^2 distribution to generate the squared amplitude.
#         gains_amplitude_2 = yield Prior(
#             tfpd.NoncentralChi2(
#                 noncentrality=ones / self.gain_stddev ** 2,
#                 df=2,
#             ),
#             name=f'{name}_amplitude_squared'
#         ).parametrised()
#         gains_amplitude = self.gain_stddev * jnp.sqrt(gains_amplitude_2)
#         gains = gains_amplitude  # [num_source, num_ant]
#         return gains
#
#     def _phase_to_gain(self, phase):
#         return jax.lax.complex(jnp.cos(phase), jnp.sin(phase))
#
#     def _build_ones(self, *args):
#         args = list(filter(lambda a: a is not None, args))
#
#         def to_ones(a, dtype):
#             ones = (a == a).astype(dtype)
#             return ones
#
#         arrays = []
#         for idx, arg in enumerate(args):
#             # reshape to [1,...,N,...1] i.e. N in [idx] dim
#             # [1, 1, N, 1, 1]
#             for _ in range(idx):
#                 arg = arg[None]
#             for _ in range(len(args) - idx - 1):
#                 arg = arg[..., None]
#             arrays.append(to_ones(arg, dtype=jnp.float64))
#         ones = arrays[0]
#         for arg in arrays[1:]:
#             ones *= arg
#         return ones
#
#     def _get_term(self, gain_type, direction_idxs, time_idxs, antenna_idxs, freq_idxs, name, dof):
#         if dof == 1:
#             suffix_idxs = ()
#         elif dof == 2:
#             suffix_idxs = (jnp.arange(2),)
#         elif dof == 4:
#             suffix_idxs = (jnp.arange(2), jnp.arange(2))
#         else:
#             raise ValueError('Invalid dof')
#
#         gain_components = gain_type.split('+')
#         phase_components = []  # add
#         amplitude_components = []  # multiply
#         for component in gain_components:
#             if component == 'phase':
#                 ones = self._build_ones(direction_idxs, time_idxs, antenna_idxs, freq_idxs, *suffix_idxs)
#                 term = (yield from self._make_gains_model_phase(ones, name))
#                 phase_components.append(term)
#             elif component == 'amplitude':
#                 ones = self._build_ones(direction_idxs, time_idxs, antenna_idxs, freq_idxs, *suffix_idxs)
#                 term = (yield from self._make_gains_model_amplitude(ones, name))
#                 amplitude_components.append(term)
#             elif component == 'unconstrained':
#                 ones = self._build_ones(direction_idxs, time_idxs, antenna_idxs, freq_idxs, *suffix_idxs)
#                 term = (yield from self._make_gains_model_unconstrained(ones, name))
#                 amplitude_components.append(term)
#             elif component == 'clock':
#                 # Always DI
#                 ones = self._build_ones(time_idxs, antenna_idxs, jnp.arange(1), *suffix_idxs)
#                 term = (yield from self._make_gains_model_clock(ones, name, dof))
#                 phase_components.append(term)
#             elif component == 'dtec':
#                 # DD if possible
#                 ones = self._build_ones(direction_idxs, time_idxs, antenna_idxs, jnp.arange(1), *suffix_idxs)
#                 term = (yield from self._make_gains_model_dtec(ones, name, dof))
#                 phase_components.append(term)
#             else:
#                 raise ValueError(f'Got unsupported component {component}.')
#
#         if len(phase_components) > 0:
#             phase = sum(phase_components[1:], phase_components[0])
#             amplitude_components.append(self._phase_to_gain(phase))
#
#         if len(amplitude_components) == 0:
#             raise ValueError(f"Not enough components provided {gain_type}.")
#
#         # Sort so complex objs last
#         amplitude_components = sorted(amplitude_components, key=lambda c: jnp.iscomplexobj(c))
#         gain = amplitude_components[0]
#         for component in amplitude_components[1:]:
#             gain = gain * component
#         if self.full_stokes:
#             # Broadcast to 2x2
#             if dof == 1:
#                 # Set diag
#                 fn = lambda g: jnp.full((2, 2), g)
#                 leading_dims = len(np.shape(gain))
#                 gain = simple_broadcast(fn, leading_dims=leading_dims)(gain)  # [D,T,A,F,2,2]
#             elif dof == 2:
#                 # Set diag
#                 leading_dims = len(np.shape(gain)) - 1
#                 gain = simple_broadcast(jnp.diag, leading_dims=leading_dims)(gain)  # [D,T,A,F,2,2]
#             elif dof == 4:
#                 # leading_dims = len(np.shape(ones)) - 2
#                 # gain = gain  # [D,T,A,F,2,2]
#                 pass
#             else:
#                 raise ValueError(f"Unsupported dof, {dof}")
#         else:
#             if dof != 1:
#                 raise ValueError(f"Unsupported dof, {dof} for full stokes.")
#         return gain
#
#     def build_prior_model(self, num_source: int, num_ant: int, freqs: FloatArray, times: FloatArray) -> PriorModelType:
#         D = num_source
#         T = len(times)
#         F = len(freqs)
#         A = num_ant
#
#         direction_idxs = jnp.arange(D)
#         time_idxs = self.times
#         antenna_idx = jnp.arange(A)
#         freq_idxs = self.freqs
#
#         def prior_model():
#             gains = yield from self._get_term(self.dd_type,
#                                               direction_idxs=direction_idxs, time_idxs=time_idxs,
#                                               antenna_idxs=antenna_idx, freq_idxs=freq_idxs,
#                                               name='dd', dof=self.dd_dof)
#             if self.double_differential:
#                 gains_di = yield from self._get_term(self.di_type,
#                                                      direction_idxs=None, time_idxs=time_idxs,
#                                                      antenna_idxs=antenna_idx, freq_idxs=freq_idxs,
#                                                      name='di', dof=self.di_dof)
#                 if self.full_stokes:
#                     gains = gains @ gains_di
#                 else:
#                     gains = gains * gains_di
#             return mp_policy.cast_to_gain(gains)
#
#         return prior_model
#
#
# GainPriorModel.register_pytree()

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


def gain_prior_model(
        num_source: int, num_ant: int,
        freqs: FloatArray, times: FloatArray,
        hour_angle_rad: FloatArray, latitude_rad: FloatArray,
        declination_rad: FloatArray,
        lna_amp_stddev: FloatArray, lna_phase_stddev_rad: FloatArray,
        parallactic_offset_stddev_rad: FloatArray,
        leakage_phase_stddev_rad: FloatArray,
        leakage_amp_max: FloatArray,
        rotation_measure_stddev_rad_m2: FloatArray,
        bandpass_amplitude_stddev: FloatArray,
        feed_delay_stddev_ns: FloatArray,
        tec_stddev_mtecu: FloatArray

) -> ComplexArray:
    """
    Create gains.

    Args:
        num_source: how many sources to model
        num_ant: how many antennas to model
        freqs: [Cs] the frequencies to compute the model data, in Hz
        times: [Ts] the times to compute the model data, in TT since start of observation

    Returns:
        [D, Ts, A, Cs, 2, 2]
    """
    terms = []

    # G - scalar (LNA + electronics: slow amplitude and slow phase)
    lna_gains = build_lna_gains(num_ant, lna_amp_stddev, lna_phase_stddev_rad, freqs, times)  # [1, 1, A, C, 2, 2]
    terms.append(lna_gains)

    # B - smooth amplitude + per feed delay
    bandpass_gains = build_bandpass_gains(freqs, num_ant, bandpass_amplitude_stddev=bandpass_amplitude_stddev,
                                          feed_delay_stddev_ns=feed_delay_stddev_ns)  # [1, 1, A, C, 2, 2]
    terms.append(bandpass_gains)

    # D - leakage and Faraday rotation
    leakage_gain = build_leakage_model(
        num_ant, freqs,
        leakage_phase_stddev_rad=leakage_phase_stddev_rad,
        leakage_amp_max=leakage_amp_max
    )  # [1, 1, A, C, 2, 2]
    terms.append(leakage_gain)

    # E - PB model
    # TODO: add PB model here

    # P - parallactic angle
    parallactic_gains = build_parallactic_gain(num_ant, declination_rad, hour_angle_rad, latitude_rad,
                                               parallactic_offset_stddev_rad)
    terms.append(parallactic_gains)  # [1, 1, A, 1, 2, 2]

    faraday_gains = build_ionosphere_gains(num_source, num_ant, freqs, times, rotation_measure_stddev_rad_m2,
                                           tec_stddev_mtecu)
    terms.append(faraday_gains)

    # Combine all terms with matrix multiplication
    output = terms[0]
    for term in terms[1:]:
        output = output @ term  # Matrix multiplication
    return output


def build_ionosphere_gains(num_source, num_ant, freqs, times, rotation_measure_stddev_rad_m2, tec_stddev_mtecu):
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
    return tec_gains @ faraday_gains  # [D, T, A, C, 2, 2]


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


def build_parallactic_gain(A, declination_rad, hour_angle_rad, latitude_rad, parallactic_offset_stddev_rad):
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


def build_lna_gains(num_ant, lna_amp_stddev, lna_phase_stddev_rad, freqs, times):
    # G - scalar (LNA + electronics: slow amplitude and slow phase)
    T = len(times)
    model_times = times[jnp.array([0, T // 2, -1])]
    Tm = len(model_times)

    g_ones = jnp.ones((1, Tm, num_ant, 1))
    G_amp_model = Prior(
        tfpd.LogNormal.experimental_from_mean_variance(
            mean=g_ones,
            variance=lna_amp_stddev ** 2
        ),
        name="G_amp_model"
    ).parameter(random_init=True)
    G_amp = quadratic_interpolation(
        x=times,
        xp=model_times,
        yp=G_amp_model,
        axis=1
    )  # [1, T, A, 1]

    G_phase_model = Prior(
        tfpd.Normal(
            loc=jnp.zeros_like(g_ones),
            scale=lna_phase_stddev_rad
        ),
        name="G_phase_model"
    ).parameter(random_init=True)
    G_phase = quadratic_interpolation(
        x=times,
        xp=model_times,
        yp=G_phase_model,
        axis=1
    )  # [1, T, A, 1]

    G = G_amp * jax.lax.complex(
        jnp.cos(G_phase), jnp.sin(G_phase)
    )

    G = set_diagonal_scalar(G)  # [1, T, A, 1, 2, 2]

    return G


def build_bandpass_gains(freqs, num_ant, bandpass_amplitude_stddev, feed_delay_stddev_ns):
    A = num_ant
    C = np.shape(freqs)[0]
    model_freqs = freqs[jnp.array([0, C // 2, -1])]
    Cm = len(model_freqs)
    # Bandpass [T, A, C, 2] smooth over freqs, and time.
    bandpass_ones = jnp.ones((1, 1, A, Cm, 2), dtype=jnp.float32)
    bandpass_amplitude_model = Prior(
        tfpd.LogNormal.experimental_from_mean_variance(
            mean=bandpass_ones,
            variance=bandpass_amplitude_stddev ** 2
        ),
        name='bandpass_amplitude_model'
    ).parameter(random_init=True)
    bandpass_amplitude = quadratic_interpolation(
        x=freqs,
        xp=model_freqs,
        yp=bandpass_amplitude_model,
        axis=-2
    )  # interp freq
    bandpass_amplitude = set_diagonal(bandpass_amplitude)  # [1, 1, A, C, 2, 2]

    # Add per feed delay
    phase_conv = (2 * jnp.pi * 1e-9) * freqs  # rad / ns # [C]
    feed_delay_ones = jnp.ones((1, 1, num_ant, 1, 2), dtype=jnp.float32)  # [1, 1, A, 1, 2]
    feed_delay_ns = Prior(
        tfpd.Normal(
            loc=jnp.zeros_like(feed_delay_ones),
            scale=feed_delay_stddev_ns * feed_delay_ones
        ),
        name='feed_delay_ns'
    ).parameter(random_init=True)
    feed_phase = phase_conv[:, None] * feed_delay_ns  # [1, 1, A, C, 2]
    feed_gains = jax.lax.complex(jnp.cos(feed_phase), jnp.sin(feed_phase))  # [1, 1, A, C, 2]
    feed_gains = set_diagonal(feed_gains)  # [1, 1, A, C, 2, 2]
    gains = bandpass_amplitude * feed_gains  # [1, 1, A, C, 2, 2]
    return gains


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
