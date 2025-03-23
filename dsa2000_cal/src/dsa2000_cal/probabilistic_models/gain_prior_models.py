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
    freqs: FloatArray
    times: FloatArray

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

    def _make_gains_model_unconstrained(self, shape, name: str):
        ones = jnp.ones(shape)
        scale = jnp.full(shape, self.gain_stddev)
        gains_real = yield Prior(
            tfpd.Normal(loc=ones,
                        scale=scale
                        ),
            name=f'{name}_real'
        ).parametrised()
        gains_imag = yield Prior(
            tfpd.Normal(loc=ones,
                        scale=scale
                        ),
            name=f'{name}_imag'
        ).parametrised()
        gains = jax.lax.complex(gains_real, gains_imag)
        return gains

    def _make_gains_model_phase(self, shape, name: str):
        ones = jnp.ones(shape)
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

    def _make_gains_model_clock(self, shape, name: str, dof: int):
        ones = jnp.ones(shape)
        # clock is in ns normally, and is unbounded
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

    def _make_gains_model_dtec(self, shape, name: str, dof: int):
        ones = jnp.ones(shape)
        # clock is in ns normally, and is unbounded
        dtec = yield Prior(
            tfpd.Uniform(
                low=-self.max_dtec_mtecu * ones,
                high=self.max_dtec_mtecu * ones
            ),
            name=f'{name}_dtec'
        ).parametrised()
        # TEC_CONV = -8.4479745 * au.rad * au.MHz  # rad * MHz / mTECU
        dtec_conv = -8.4479745 * 1e6 / self._broadcasted_freqs(dof=dof)
        phase = dtec_conv * dtec
        return phase

    def _make_gains_model_amplitude(self, shape, name: str):
        ones = jnp.ones(shape)
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

    def _get_term(self, gain_type, leading_shape, name, dof):
        leading_dims = len(leading_shape)

        if dof == 1:
            shape = leading_shape
        elif dof == 2:
            shape = leading_shape + (2,)
        elif dof == 4:
            shape = leading_shape + (2, 2)
        else:
            raise ValueError('Invalid dof')

        gain_components = gain_type.split('+')
        phase_components = []  # add
        amplitude_components = []  # mulitply
        for component in gain_components:
            if component == 'phase':
                term = (yield from self._make_gains_model_phase(shape, name))
                phase_components.append(term)
            elif component == 'amplitude':
                term = (yield from self._make_gains_model_amplitude(shape, name))
                amplitude_components.append(term)
            elif component == 'unconstrained':
                term = (yield from self._make_gains_model_unconstrained(shape, name))
                amplitude_components.append(term)
            elif component == 'clock':
                term = (yield from self._make_gains_model_clock(shape, name, dof))
                phase_components.append(term)
            elif component == 'dtec':
                term = (yield from self._make_gains_model_dtec(shape, name, dof))
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

        if dof == 1:
            # Set diag
            fn = lambda g: jnp.full((2, 2), g)
            gain = simple_broadcast(fn, leading_dims=leading_dims)(gain)  # [D,T,A,F,2,2]
        elif dof == 2:
            # Set diag
            gain = simple_broadcast(jnp.diag, leading_dims=leading_dims)(gain)  # [D,T,A,F,2,2]
        elif dof == 4:
            pass
            # gain = gain  # [D,T,A,F,2,2]
        else:
            raise ValueError(f"Unsupported dof, {dof}")
        return gain

    def build_prior_model(self, num_source: int, num_ant: int, freqs: FloatArray, times: FloatArray) -> PriorModelType:
        D = num_source
        T = len(times)
        F = len(freqs)
        A = num_ant

        def prior_model():
            gains = yield from self._get_term(self.dd_type, leading_shape=(D, T, A, F), name='dd', dof=self.dd_dof)
            if self.double_differential:
                gains_di = yield from self._get_term(self.di_type, leading_shape=(T, A, F), name='di', dof=self.di_dof)
                if self.full_stokes:
                    gains = gains @ gains_di
                else:
                    gains = gains * gains_di
            return mp_policy.cast_to_gain(gains)

        return prior_model


GainPriorModel.register_pytree()
