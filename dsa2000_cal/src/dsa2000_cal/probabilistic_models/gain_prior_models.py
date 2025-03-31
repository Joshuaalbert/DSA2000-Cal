import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Tuple, List

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jaxns import PriorModelType, Prior
from jaxns.framework import context as ctx
from jaxns.framework.ops import simulate_prior_model

from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.jax_utils import simple_broadcast
from dsa2000_common.common.mixed_precision_utils import mp_policy
from dsa2000_common.common.pytree import Pytree

tfpd = tfp.distributions


class AbstractGainPriorModel(Pytree, ABC):

    @abstractmethod
    def gain_shape(self):
        ...

    @abstractmethod
    def get_spec(self, freq_spec, time_spec) -> 'GainPriorModel':
        ...

    @abstractmethod
    def compute_gains(self, params: Any):
        ...

    @abstractmethod
    def get_init_params(self, key) -> Any:
        ...

    @abstractmethod
    def build_prior_model(self, num_source: int, num_ant: int, freqs: jax.Array, times: jax.Array) -> PriorModelType:
        """
        Define the prior model for the gains.

        Args:
            num_source: the number of sources
            num_ant: the number of antennas
            freqs: [num_chan] the frequencies
            times: [num_time] the times to compute the model data, in TT since start of observation

        Returns:
            gains: [num_source, num_time, num_ant, num_chan, [, 2, 2]].
        """
        ...


@dataclasses.dataclass(eq=False)
class GainPriorModel(AbstractGainPriorModel):
    """
    A gain model with unconstrained complex Gaussian priors with zero mean.
    """
    num_source: int
    num_ant: int
    freqs: FloatArray # [Cs]
    times: FloatArray # [Ts]

    full_stokes: bool = True

    gain_stddev: float = 2.
    max_clock_ns: float = 2
    max_dtec_mtecu: float = 200

    dd_type: str = 'unconstrained'  # Combine components with +
    dd_dof: int = 4
    double_differential: bool = True
    di_dof: int = 4
    di_type: str = 'unconstrained'  # Combine components with +

    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return

        if not self.full_stokes:
            if self.di_dof != 1:
                raise ValueError('di_dof must be 1 if full stokes')
            if self.dd_dof != 1:
                raise ValueError('dd_dof must be 1 if full stokes')
        if len(np.shape(self.freqs)) != 1:
            raise ValueError('freqs must be a 1D array')
        if len(np.shape(self.times)) != 1:
            raise ValueError('times must be a 1D array')

    @classmethod
    def flatten(cls, this: 'GainPriorModel') -> Tuple[List[Any], Tuple[Any, ...]]:
        return (
            [
                this.freqs, this.times
            ],
            (
                this.num_source, this.num_ant, this.full_stokes,
                this.gain_stddev, this.max_dtec_mtecu, this.max_clock_ns,
                this.dd_type, this.dd_dof, this.double_differential, this.di_dof, this.di_type,

            )
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> 'GainPriorModel':
        [freqs, times] = children
        (num_source, num_ant, full_stokes,
         gain_stddev, max_dtec_mtecu, max_clock_ns,
         dd_type, dd_dof, double_differential, di_dof, di_type) = aux_data
        return GainPriorModel(
            num_source=num_source,
            num_ant=num_ant,
            freqs=freqs,
            times=times,
            full_stokes=full_stokes,
            gain_stddev=gain_stddev,
            max_clock_ns=max_clock_ns,
            max_dtec_mtecu=max_dtec_mtecu,
            dd_type=dd_type,
            dd_dof=dd_dof,
            double_differential=double_differential,
            di_dof=di_dof,
            di_type=di_type,
            skip_post_init=True
        )

    def get_spec(self, freq_spec, time_spec) -> 'GainPriorModel':
        [freqs, times], aux_data = GainPriorModel.flatten(self)
        return GainPriorModel.unflatten(aux_data, [freq_spec, time_spec])

    def gain_shape(self):
        if self.full_stokes:
            return (self.num_source, len(self.times), self.num_ant, len(self.freqs), 2, 2)
        else:
            return (self.num_source, len(self.times), self.num_ant, len(self.freqs), 2, 2)

    def compute_gains(self, params: Any):
        def transform():
            prior_model = self.build_prior_model(
                num_source=self.num_source,
                num_ant=self.num_ant,
                freqs=self.freqs,
                times=self.times
            )
            (gains,), _ = simulate_prior_model(jax.random.PRNGKey(0), prior_model)  # [D, Tm, A, Cm[,2,2]]
            return gains

        get_gains_transformed = ctx.transform(transform)
        gains = get_gains_transformed.apply(params, jax.random.PRNGKey(0)).fn_val
        return gains

    def get_init_params(self, key) -> Any:
        def transform():
            prior_model = self.build_prior_model(
                num_source=self.num_source,
                num_ant=self.num_ant,
                freqs=self.freqs,
                times=self.times
            )
            (gains,), _ = simulate_prior_model(jax.random.PRNGKey(0), prior_model)  # [D, Tm, A, Cm[,2,2]]
            return gains

        get_gains_transformed = ctx.transform(transform)
        return get_gains_transformed.init(key).params

    def _make_gains_model_unconstrained(self, ones, name: str):
        scale = self.gain_stddev * ones
        gains_real = yield Prior(
            tfpd.Normal(loc=ones,
                        scale=scale
                        ),
            name=f'{name}_real'
        ).parametrised()
        gains_imag = yield Prior(
            tfpd.Normal(loc=jnp.zeros_like(ones),
                        scale=scale
                        ),
            name=f'{name}_imag'
        ).parametrised()
        gains = jax.lax.complex(gains_real, gains_imag)
        return gains

    def _make_gains_model_phase(self, ones, name: str):
        gains_phase = yield Prior(
            tfpd.Uniform(
                low=-jnp.pi * ones,
                high=jnp.pi * ones
            ),
            name=f'{name}_phase'
        ).parametrised()
        return gains_phase

    def _broadcasted_freqs(self, dof: int):
        if dof == 1:
            return self.freqs
        elif dof == 2:
            return self.freqs[..., None]
        elif dof == 4:
            return self.freqs[..., None, None]
        else:
            raise ValueError('Invalid dof')

    def _make_gains_model_clock(self, ones, name: str, dof: int):
        # clock is in ns
        clock = yield Prior(
            tfpd.Uniform(
                low=-self.max_clock_ns * ones,
                high=self.max_clock_ns * ones
            ),
            name=f'{name}_clock'
        ).parametrised()
        phase_conv = (2 * jnp.pi * 1e-9) * self._broadcasted_freqs(dof=dof)
        phase = phase_conv * clock
        return phase

    def _make_gains_model_dtec(self, ones, name: str, dof: int):
        # dtec is in mtecu
        dtec = yield Prior(
            tfpd.Uniform(
                low=-self.max_dtec_mtecu * ones,
                high=self.max_dtec_mtecu * ones
            ),
            name=f'{name}_dtec'
        ).parametrised()
        # TEC_CONV = -8.4479745 * au.rad * au.MHz  # rad * MHz / mTECU
        dtec_conv = (-8.4479745 * 1e6) / self._broadcasted_freqs(dof=dof)
        phase = dtec_conv * dtec
        return phase

    def _make_gains_model_amplitude(self, ones, name: str):
        # Rice distribution for X ~ N[1, sigma^2], Y ~ U[0, sigma^2] then R^2 = X^2 + Y^2 ~ Rice(1, sigma^2)
        # We use noncentral chi^2 distribution to generate the squared amplitude.
        gains_amplitude_2 = yield Prior(
            tfpd.NoncentralChi2(
                noncentrality=ones / self.gain_stddev ** 2,
                df=2,
            ),
            name=f'{name}_amplitude_squared'
        ).parametrised()
        gains_amplitude = self.gain_stddev * jnp.sqrt(gains_amplitude_2)
        gains = gains_amplitude  # [num_source, num_ant]
        return gains

    def _phase_to_gain(self, phase):
        return jax.lax.complex(jnp.cos(phase), jnp.sin(phase))

    def _build_ones(self, *args):
        args = list(filter(lambda a: a is not None, args))

        def to_ones(a, dtype):
            ones = (a == a).astype(jnp.float32)
            return ones

        arrays = []
        for idx, arg in enumerate(args):
            # reshape to [1,...,N,...1] i.e. N in [idx] dim
            # [1, 1, N, 1, 1]
            for _ in range(idx):
                arg = arg[None]
            for _ in range(len(args) - idx - 1):
                arg = arg[..., None]
            arrays.append(to_ones(arg, dtype=jnp.float32))
        ones = arrays[0]
        for arg in arrays[1:]:
            ones *= arg
        return ones

    def _get_term(self, gain_type, direction_idxs, time_idxs, antenna_idxs, freq_idxs, name, dof):
        if dof == 1:
            suffix_idxs = ()
        elif dof == 2:
            suffix_idxs = (jnp.arange(2),)
        elif dof == 4:
            suffix_idxs = (jnp.arange(2), jnp.arange(2))
        else:
            raise ValueError('Invalid dof')

        gain_components = gain_type.split('+')
        phase_components = []  # add
        amplitude_components = []  # multiply
        for component in gain_components:
            if component == 'phase':
                ones = self._build_ones(direction_idxs, time_idxs, antenna_idxs, freq_idxs, *suffix_idxs)
                term = (yield from self._make_gains_model_phase(ones, name))
                phase_components.append(term)
            elif component == 'amplitude':
                ones = self._build_ones(direction_idxs, time_idxs, antenna_idxs, freq_idxs, *suffix_idxs)
                term = (yield from self._make_gains_model_amplitude(ones, name))
                amplitude_components.append(term)
            elif component == 'unconstrained':
                ones = self._build_ones(direction_idxs, time_idxs, antenna_idxs, freq_idxs, *suffix_idxs)
                term = (yield from self._make_gains_model_unconstrained(ones, name))
                amplitude_components.append(term)
            elif component == 'clock':
                # Always DI
                ones = self._build_ones(time_idxs, antenna_idxs, jnp.arange(1), *suffix_idxs)
                term = (yield from self._make_gains_model_clock(ones, name, dof))
                phase_components.append(term)
            elif component == 'dtec':
                # DD if possible
                ones = self._build_ones(direction_idxs, time_idxs, antenna_idxs, jnp.arange(1), *suffix_idxs)
                term = (yield from self._make_gains_model_dtec(ones, name, dof))
                phase_components.append(term)
            else:
                raise ValueError(f'Got unsupported component {component}.')

        if len(phase_components) > 0:
            phase = sum(phase_components[1:], phase_components[0])
            amplitude_components.append(self._phase_to_gain(phase))

        if len(amplitude_components) == 0:
            raise ValueError(f"Not enough components provided {gain_type}.")

        # Sort so complex objs last
        amplitude_components = sorted(amplitude_components, key=lambda c: jnp.iscomplexobj(c))
        gain = amplitude_components[0]
        for component in amplitude_components[1:]:
            gain = gain * component
        if self.full_stokes:
            # Broadcast to 2x2
            if dof == 1:
                # Set diag
                fn = lambda g: jnp.full((2, 2), g)
                leading_dims = len(np.shape(gain))
                gain = simple_broadcast(fn, leading_dims=leading_dims)(gain)  # [D,T,A,F,2,2]
            elif dof == 2:
                # Set diag
                leading_dims = len(np.shape(gain)) - 1
                gain = simple_broadcast(jnp.diag, leading_dims=leading_dims)(gain)  # [D,T,A,F,2,2]
            elif dof == 4:
                # leading_dims = len(np.shape(ones)) - 2
                # gain = gain  # [D,T,A,F,2,2]
                pass
            else:
                raise ValueError(f"Unsupported dof, {dof}")
        else:
            if dof != 1:
                raise ValueError(f"Unsupported dof, {dof} for full stokes.")
        return gain

    def build_prior_model(self, num_source: int, num_ant: int, freqs: FloatArray, times: FloatArray) -> PriorModelType:
        D = num_source
        T = len(times)
        F = len(freqs)
        A = num_ant

        direction_idxs = jnp.arange(D)
        time_idxs = self.times
        antenna_idx = jnp.arange(A)
        freq_idxs = self.freqs

        def prior_model():
            gains = yield from self._get_term(self.dd_type,
                                              direction_idxs=direction_idxs, time_idxs=time_idxs,
                                              antenna_idxs=antenna_idx, freq_idxs=freq_idxs,
                                              name='dd', dof=self.dd_dof)
            if self.double_differential:
                gains_di = yield from self._get_term(self.di_type,
                                                     direction_idxs=None, time_idxs=time_idxs,
                                                     antenna_idxs=antenna_idx, freq_idxs=freq_idxs,
                                                     name='di', dof=self.di_dof)
                if self.full_stokes:
                    gains = gains @ gains_di
                else:
                    gains = gains * gains_di
            return mp_policy.cast_to_gain(gains)

        return prior_model


GainPriorModel.register_pytree()
