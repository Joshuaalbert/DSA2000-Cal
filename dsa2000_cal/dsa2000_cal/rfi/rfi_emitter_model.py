import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(eq=False)
class RFIEmitterModel:
    delays: jax.Array  # [n_delays]
    auto_correlation_function: jax.Array  # [n_delays]
    emitter_location_uvw: jax.Array  # [3]

    def __post_init__(self):
        if np.shape(self.delays) != np.shape(self.auto_correlation_function):
            raise ValueError('Delays and auto correlation function must have the same shape')
        if len(np.shape(self.delays)) != 1:
            raise ValueError('Delays and auto correlation function must be 1D arrays')

    def compute_auto_correlation(self, delay: jax.Array) -> jax.Array:
        return jnp.interp(delay, self.delays, self.auto_correlation_function)

    def compute_coherence(self, antenna1_uvw: jax.Array, antenna2_uvw: jax.Array, wavelength: jax.Array) -> jax.Array:
        """
        Computes the electric field coherence between two antennas <E_1 E_2^H>.

        Args:
            antenna1_uvw: [3] first antenna UVW coordinates
            antenna2_uvw: [3] second antenna UVW coordinates
            wavelength: the wavelength of the signal

        Returns:
            coherence: [2, 2] the electric field coherence matrix in linear polarisation
        """
            # E_1 = a(r_1) e^(-2pi i/wavelength * delay_1)
        # a(r) = E_0/|r| f