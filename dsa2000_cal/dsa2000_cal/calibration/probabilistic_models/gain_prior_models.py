import dataclasses
from abc import ABC, abstractmethod

import jax
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jaxns import PriorModelType, Prior

tfpd = tfp.distributions


class AbstractGainPriorModel(ABC):
    @abstractmethod
    def prior_model(self, num_source: int, num_ant: int, freqs: jax.Array) -> PriorModelType:
        """
        Define the prior model for the gains.

        Args:
            num_source: the number of sources
            num_ant: the number of antennas
            freqs: [num_chan] the frequencies, should use for sharding

        Returns:
            gains: [num_source, num_ant, num_chan[, 2, 2]], using `freqs` to shard axis 2
        """
        ...


@dataclasses.dataclass(eq=False)
class UnconstrainedGain(AbstractGainPriorModel):
    """
    A gain model with unconstrained complex Gaussian priors with zero mean.
    """
    gain_stddev: float = 2.

    def prior_model(self, num_source: int, num_ant: int, freqs: jax.Array) -> PriorModelType:
        loc = jnp.zeros((num_source, num_ant, 2, 2))
        scale = jnp.full((num_source, num_ant, 2, 2), self.gain_stddev)
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
        gains = jax.vmap(lambda _freq: gains, out_axes=2)(freqs)  # [num_source, num_ant, num_chan, 2, 2]
        return gains


@dataclasses.dataclass(eq=False)
class DiagonalUnconstrainedGain(AbstractGainPriorModel):
    """
    A gain model with unconstrained complex Gaussian priors with zero mean, but only on the diagonal.
    """
    gain_stddev: float = 2.

    def prior_model(self, num_source: int, num_ant: int, freqs: jax.Array) -> PriorModelType:
        loc = jnp.zeros((num_source, num_ant, 2))
        scale = jnp.full((num_source, num_ant, 2), self.gain_stddev)
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
        gains = jax.vmap(lambda _freq: gains, out_axes=2)(freqs)  # [num_source, num_ant, num_chan, 2, 2]
        return gains
