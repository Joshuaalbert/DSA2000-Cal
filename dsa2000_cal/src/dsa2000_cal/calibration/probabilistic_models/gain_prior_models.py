import dataclasses
from abc import ABC, abstractmethod
from typing import Literal

import jax
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jaxns import PriorModelType, Prior

from dsa2000_cal.common.array_types import FloatArray

tfpd = tfp.distributions


class AbstractGainPriorModel(ABC):
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
    gain_stddev: float = 2.
    full_stokes: bool = True

    dd_type: Literal['unconstrained', 'rice'] = 'unconstrained'
    dd_dof: int = 4

    double_differential: bool = True
    di_dof: int = 4
    di_type: Literal['unconstrained', 'rice'] = 'unconstrained'

    def _make_gains_model_unconstrained(self, shape):
        ones = jnp.ones(shape)
        scale = jnp.full(shape, self.gain_stddev)
        gains_real = yield Prior(
            tfpd.Normal(loc=ones,
                        scale=scale
                        ),
            name='gains_real'
        ).parametrised()
        gains_imag = yield Prior(
            tfpd.Normal(loc=ones,
                        scale=scale
                        ),
            name='gains_imag'
        ).parametrised()
        gains = jax.lax.complex(gains_real, gains_imag)
        return gains

    def _make_gains_model_rice(self, shape):
        ones = jnp.ones(shape)
        # Rice distribution for X ~ N[1, sigma^2], Y ~ U[0, sigma^2] then R^2 = X^2 + Y^2 ~ Rice(1, sigma^2)
        # We use noncentral chi^2 distribution to generate the squared amplitude.
        gains_amplitude_2 = yield Prior(
            tfpd.NoncentralChi2(
                noncentrality=ones / self.gain_stddev ** 2,
                df=2,
            ),
            name='gains_amplitude_squared'
        ).parametrised()
        gains_amplitude = self.gain_stddev * jnp.sqrt(gains_amplitude_2)
        gains_phase = yield Prior(
            tfpd.Uniform(
                low=-jnp.pi * ones,
                high=jnp.pi * ones
            ),
            name='gains_phase'
        ).parametrised()
        gains = gains_amplitude * jax.lax.complex(jnp.cos(gains_phase),
                                                  jnp.sin(gains_phase))  # [num_source, num_ant]
        return gains

    def build_prior_model(self, num_source: int, num_ant: int, freqs: FloatArray, times: FloatArray) -> PriorModelType:
        T = len(times)
        F = len(freqs)
        D = num_source
        A = num_ant

        def prior_model():
            def di_make_gains_model(shape):
                if self.di_type == 'unconstrained':
                    return (yield from self._make_gains_model_unconstrained(shape))
                elif self.di_type == 'rice':
                    return (yield from self._make_gains_model_rice(shape))
                else:
                    raise ValueError(f"Unsupported di_type, {self.di_type}")

            def dd_make_gains_model(shape):
                if self.dd_type == 'unconstrained':
                    return (yield from self._make_gains_model_unconstrained(shape))
                elif self.dd_type == 'rice':
                    return (yield from self._make_gains_model_rice(shape))
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
                else:
                    if self.di_dof != 1:
                        raise ValueError("Cannot have full_stokes=False and double_differential_dof > 1")
                    gains_di = yield from di_make_gains_model((T, A, F))
                gains = gains_di @ gains  # [D,T,A,F,2,2]

            return gains

        return prior_model
