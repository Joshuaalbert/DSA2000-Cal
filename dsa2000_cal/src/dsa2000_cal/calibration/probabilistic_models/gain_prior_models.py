import dataclasses
from abc import ABC, abstractmethod

import jax
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jaxns import PriorModelType, Prior

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
class UnconstrainedGain(AbstractGainPriorModel):
    """
    A gain model with unconstrained complex Gaussian priors with zero mean.
    """
    gain_stddev: float = 2.
    full_stokes: bool = True
    dof: int = 4

    def build_prior_model(self, num_source: int, num_ant: int, freqs: jax.Array, times: jax.Array) -> PriorModelType:
        T = len(times)
        F = len(freqs)
        D = num_source
        A = num_ant

        def prior_model():
            def make_gains_model(shape):
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

            if self.full_stokes:
                if self.dof == 1:
                    gains = yield from make_gains_model((D, T, A, F))
                    # Set diag
                    gains = jax.vmap(jax.vmap(jax.vmap(jax.vmap(lambda g: jnp.diag(jnp.stack([g, g]))))))(
                        gains)  # [D,T,A,F,2,2]
                elif self.dof == 2:
                    gains = yield from make_gains_model((D, T, A, F, 2))
                    # Set diag
                    gains = jax.vmap(jax.vmap(jax.vmap(jax.vmap(jnp.diag))))(gains)  # [D,T,A,F,2,2]
                elif self.dof == 4:
                    gains = yield from make_gains_model((D, T, A, F, 2, 2))  # [D,T,A,F,2,2]
                else:
                    raise ValueError(f"Unsupported dof, {self.dof}")
            else:
                if self.dof != 1:
                    raise ValueError("Cannot have full_stokes=False and dof > 1")
                gains = yield from make_gains_model((D, T, A, F))  # [D,T,A,F]

            return gains

        return prior_model


@dataclasses.dataclass(eq=False)
class ScalarRiceGain(AbstractGainPriorModel):
    """
    A gain model with unconstrained complex Gaussian priors with zero mean, but only on the diagonal.
    """
    gain_stddev: float = 0.1
    full_stokes: bool = True
    dof: int = 4

    def build_prior_model(self, num_source: int, num_ant: int, freqs: jax.Array, times: jax.Array) -> PriorModelType:
        T = len(times)
        F = len(freqs)
        D = num_source
        A = num_ant

        def prior_model():
            def make_gains_model(shape):
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

            if self.full_stokes:
                if self.dof == 1:
                    gains = yield from make_gains_model((D, T, A, F))
                    # Set diag
                    gains = jax.vmap(jax.vmap(jax.vmap(jax.vmap(lambda g: jnp.diag(jnp.stack([g, g]))))))(
                        gains)  # [D,T,A,F,2,2]
                elif self.dof == 2:
                    gains = yield from make_gains_model((D, T, A, F, 2))
                    # Set diag
                    gains = jax.vmap(jax.vmap(jax.vmap(jax.vmap(jnp.diag))))(gains)  # [D,T,A,F,2,2]
                elif self.dof == 4:
                    gains = yield from make_gains_model((D, T, A, F, 2, 2))  # [D,T,A,F,2,2]
                else:
                    raise ValueError(f"Unsupported dof, {self.dof}")
            else:
                if self.dof != 1:
                    raise ValueError("Cannot have full_stokes=False and dof > 1")
                gains = yield from make_gains_model((D, T, A, F))  # [D,T,A,F]

            return gains

        return prior_model
