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
            gains: [num_source, num_time, num_ant, num_chan[, 2, 2]].
        """
        ...


@dataclasses.dataclass(eq=False)
class UnconstrainedGain(AbstractGainPriorModel):
    """
    A gain model with unconstrained complex Gaussian priors with zero mean.
    """
    gain_stddev: float = 2.

    def build_prior_model(self, num_source: int, num_ant: int, freqs: jax.Array, times: jax.Array) -> PriorModelType:
        loc = jnp.zeros((num_source, num_ant, 2, 2))
        scale = jnp.full((num_source, num_ant, 2, 2), self.gain_stddev)

        def prior_model():
            gains_real = yield Prior(
                tfpd.Normal(loc=loc + 1.,
                            scale=scale
                            ),
                name='gains_real'
            )
            gains_imag = yield Prior(
                tfpd.Normal(loc=loc,
                            scale=scale
                            ),
                name='gains_imag'
            )
            gains = gains_real + 1j * gains_imag  # [num_source, num_ant, 2, 2]
            # Broadcast to shape
            gains = jnp.broadcast_to(
                gains[:, None, :, None, ...],
                (num_source, len(times), num_ant, len(freqs), 2, 2)
            )  # [num_source, num_time, num_ant, num_chan, 2, 2]
            return gains

        return prior_model


@dataclasses.dataclass(eq=False)
class DiagonalUnconstrainedGain(AbstractGainPriorModel):
    """
    A gain model with unconstrained complex Gaussian priors with zero mean, but only on the diagonal.
    """
    gain_stddev: float = 2.

    def build_prior_model(self, num_source: int, num_ant: int, freqs: jax.Array, times: jax.Array) -> PriorModelType:
        loc = jnp.zeros((num_source, num_ant, 2))
        scale = jnp.full((num_source, num_ant, 2), self.gain_stddev)

        def prior_model():
            gains_real = yield Prior(
                tfpd.Normal(
                    loc=loc + 1.,
                    scale=scale
                ),
                name='gains_real'
            )
            gains_imag = yield Prior(
                tfpd.Normal(
                    loc=loc,
                    scale=scale
                ),
                name='gains_imag'
            )
            diag_gains = gains_real + 1j * gains_imag  # [num_source, num_ant, 2]
            gains = jax.vmap(jax.vmap(jnp.diag))(diag_gains)  # [num_source, num_ant, 2, 2]
            # Broadcast to shape
            gains = jnp.broadcast_to(
                gains[:, None, :, None, ...],
                (num_source, len(times), num_ant, len(freqs), 2, 2)
            )  # [num_source, num_time, num_ant, num_chan, 2, 2]
            return gains

        return prior_model



@dataclasses.dataclass(eq=False)
class ScalarUnconstrainedGain(AbstractGainPriorModel):
    """
    A gain model with unconstrained complex Gaussian priors with zero mean, but only on the diagonal.
    """
    gain_stddev: float = 2.

    def build_prior_model(self, num_source: int, num_ant: int, freqs: jax.Array, times: jax.Array) -> PriorModelType:
        loc = jnp.zeros((num_source, num_ant))
        scale = jnp.full((num_source, num_ant), self.gain_stddev)

        def prior_model():
            gains_real = yield Prior(
                tfpd.Normal(
                    loc=loc + 1.,
                    scale=scale
                ),
                name='gains_real'
            )
            gains_imag = yield Prior(
                tfpd.Normal(
                    loc=loc,
                    scale=scale
                ),
                name='gains_imag'
            )
            gains = gains_real + 1j * gains_imag  # [num_source, num_ant]
            # Broadcast to shape
            gains = jnp.broadcast_to(
                gains[:, None, :, None],
                (num_source, len(times), num_ant, len(freqs))
            )  # [num_source, num_time, num_ant, num_chan]
            return gains

        return prior_model
