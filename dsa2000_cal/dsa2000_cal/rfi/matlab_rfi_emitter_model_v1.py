import dataclasses

import jax

from dsa2000_cal.rfi.rfi_emitter_model import AbstractRFIEmitterModel


@dataclasses.dataclass(eq=False)
class MatlabRFIEmitterModelV1(AbstractRFIEmitterModel):
    delays: jax.Array  # [n_delays]
    auto_correlation_function: jax.Array  # [n_delays]

    def __post_init__(self):
        if np.shape(self.delays) != np.shape(self.auto_correlation_function):
            raise ValueError('Delays and auto correlation function must have the same shape')
        if len(np.shape(self.delays)) != 1:
            raise ValueError('Delays and auto correlation function must be 1D arrays')

    def