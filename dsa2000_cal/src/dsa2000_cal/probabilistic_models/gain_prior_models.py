import dataclasses
from abc import ABC, abstractmethod
from typing import Literal, Any, Tuple, List

from jaxns.framework import context as ctx

import jax
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jaxns import PriorModelType, Prior
from jaxns.framework.ops import simulate_prior_model

from dsa2000_common.common.array_types import FloatArray
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

    gain_stddev: float = 2.
    full_stokes: bool = True

    dd_type: Literal['unconstrained', 'rice', 'phase_only', 'amplitude_only'] = 'unconstrained'
    dd_dof: int = 4

    double_differential: bool = True
    di_dof: int = 4
    di_type: Literal['unconstrained', 'rice', 'phase_only', 'amplitude_only'] = 'unconstrained'

    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return

    @classmethod
    def flatten(cls, this: 'GainPriorModel') -> Tuple[List[Any], Tuple[Any, ...]]:
        return (
            [
            this.freqs, this.times
            ],
            (
                this.num_source, this.num_ant, this.gain_stddev, this.full_stokes,
                this.dd_type, this.dd_dof, this.di_dof, this.di_type
            )
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> 'GainPriorModel':
        [freqs, times] = children
        (num_source, num_ant, gain_stddev, full_stokes,
        dd_type, dd_dof, di_dof, di_type) = aux_data
        return GainPriorModel(
            num_source=num_source,
            num_ant=num_ant,
            freqs=freqs,
            times=times,
            gain_stddev=gain_stddev,
            full_stokes=full_stokes,
            dd_type=dd_type,
            dd_dof=dd_dof,
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
            name=f'{name}_gains_real'
        ).parametrised()
        gains_imag = yield Prior(
            tfpd.Normal(loc=ones,
                        scale=scale
                        ),
            name=f'{name}_gains_imag'
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
            name=f'{name}_gains_phase'
        ).parametrised()
        gains = jax.lax.complex(jnp.cos(gains_phase), jnp.sin(gains_phase))  # [num_source, num_ant]
        return gains

    def _make_gains_model_amplitude(self, shape, name: str):
        ones = jnp.ones(shape)
        # Rice distribution for X ~ N[1, sigma^2], Y ~ U[0, sigma^2] then R^2 = X^2 + Y^2 ~ Rice(1, sigma^2)
        # We use noncentral chi^2 distribution to generate the squared amplitude.
        gains_amplitude_2 = yield Prior(
            tfpd.NoncentralChi2(
                noncentrality=ones / self.gain_stddev ** 2,
                df=2,
            ),
            name=f'{name}_gains_amplitude_squared'
        ).parametrised()
        gains_amplitude = self.gain_stddev * jnp.sqrt(gains_amplitude_2)
        gains = gains_amplitude  # [num_source, num_ant]
        return gains

    def _make_gains_model_rice(self, shape, name: str):
        gains_amplitude = yield from self._make_gains_model_amplitude(shape, name)
        gains_phase = yield from self._make_gains_model_phase(shape, name)
        gains = gains_amplitude * gains_phase
        return gains

    def build_prior_model(self, num_source: int, num_ant: int, freqs: FloatArray, times: FloatArray) -> PriorModelType:
        T = len(times)
        F = len(freqs)
        D = num_source
        A = num_ant

        def prior_model():
            def di_make_gains_model(shape):
                if self.di_type == 'unconstrained':
                    return (yield from self._make_gains_model_unconstrained(shape, 'di'))
                elif self.di_type == 'rice':
                    return (yield from self._make_gains_model_rice(shape, 'di'))
                elif self.di_type == 'phase_only':
                    return (yield from self._make_gains_model_phase(shape, 'di'))
                elif self.di_type == 'amplitude_only':
                    return (yield from self._make_gains_model_amplitude(shape, 'di'))
                else:
                    raise ValueError(f"Unsupported di_type, {self.di_type}")

            def dd_make_gains_model(shape):
                if self.dd_type == 'unconstrained':
                    return (yield from self._make_gains_model_unconstrained(shape, 'dd'))
                elif self.dd_type == 'rice':
                    return (yield from self._make_gains_model_rice(shape, 'dd'))
                elif self.dd_type == 'phase_only':
                    return (yield from self._make_gains_model_phase(shape, 'dd'))
                elif self.dd_type == 'amplitude_only':
                    return (yield from self._make_gains_model_amplitude(shape, 'dd'))
                else:
                    raise ValueError(f"Unsupported dd_type, {self.dd_type}")

            if self.full_stokes:
                if self.dd_dof == 1:
                    gains = yield from dd_make_gains_model((D, T, A, F))
                    # Set diag
                    gains = jax.vmap(jax.vmap(jax.vmap(jax.vmap(lambda g: jnp.diag(jnp.stack([g, g]))))))(
                        gains)  # [D,T,A,F,2,2]
                elif self.dd_dof == 2:
                    gains = yield from dd_make_gains_model((D, T, A, F, 2))
                    # Set diag
                    gains = jax.vmap(jax.vmap(jax.vmap(jax.vmap(jnp.diag))))(gains)  # [D,T,A,F,2,2]
                elif self.dd_dof == 4:
                    gains = yield from dd_make_gains_model((D, T, A, F, 2, 2))  # [D,T,A,F,2,2]
                else:
                    raise ValueError(f"Unsupported dof, {self.dd_dof}")
            else:
                if self.dd_dof != 1:
                    raise ValueError("Cannot have full_stokes=False and dof > 1")
                gains = yield from dd_make_gains_model((D, T, A, F))  # [D,T,A,F]
            if self.double_differential:
                # construct an outer gains, without the leading D dimension, and multiply the gains by them
                if self.full_stokes:
                    if self.di_dof == 1:
                        gains_di = yield from di_make_gains_model((T, A, F))
                        # Set diag
                        gains_di = jax.vmap(jax.vmap(jax.vmap(lambda g: jnp.diag(jnp.stack([g, g])))))(
                            gains_di)  # [T,A,F,2,2]
                    elif self.di_dof == 2:
                        gains_di = yield from di_make_gains_model((T, A, F, 2))
                        # Set diag
                        gains_di = jax.vmap(jax.vmap(jax.vmap(jnp.diag)))(gains_di)
                    elif self.di_dof == 4:
                        gains_di = yield from di_make_gains_model((T, A, F, 2, 2))
                    else:
                        raise ValueError(f"Unsupported double_differential_dof, {self.di_dof}")
                    gains = gains_di @ gains  # [D,T,A,F,2,2]
                else:
                    if self.di_dof != 1:
                        raise ValueError("Cannot have full_stokes=False and double_differential_dof > 1")
                    gains_di = yield from di_make_gains_model((T, A, F))
                    gains = gains_di * gains  # [D,T,A,F]
            return mp_policy.cast_to_gain(gains)

        return prior_model

GainPriorModel.register_pytree()