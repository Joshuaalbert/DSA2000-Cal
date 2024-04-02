from dataclasses import dataclass
from typing import Literal, NamedTuple, Tuple

import jaxopt
from jax import lax
from jax import numpy as jnp
from jax._src.typing import SupportsDType

from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.dft_predict import DFTPredict, DFTModelData
from dsa2000_cal.common.jax_utils import pytree_unravel


class CalibrationParams(NamedTuple):
    gains_real: jnp.ndarray  # [source, time, ant, chan, 2, 2]
    gains_imag: jnp.ndarray  # [source, time, ant, chan, 2, 2]


class CalibrationData(NamedTuple):
    visibility_coords: VisibilityCoords
    image: jnp.ndarray  # [source, chan, 2, 2]
    lmn: jnp.ndarray  # [source, 3]
    freqs: jnp.ndarray  # [chan]
    obs_vis: jnp.ndarray  # [row, chan, 2, 2]
    obs_vis_weight: jnp.ndarray  # [row, chan, 2, 2]


@dataclass(eq=False)
class Calibration:
    num_iterations: int
    convention: Literal['fourier', 'casa'] = 'fourier'
    dtype: SupportsDType = jnp.complex64

    def _objective_fun(self, params: CalibrationParams, data: CalibrationData) -> jnp.ndarray:
        residuals = self._residual_fun(params=params, data=data)
        return jnp.mean(lax.square(residuals))

    def _residual_fun(self, params: CalibrationParams, data: CalibrationData) -> jnp.ndarray:
        dft_predict = DFTPredict(
            dtype=self.dtype,
            convention=self.convention
        )
        gains = jnp.asarray(params.gains_real + 1j * params.gains_imag, dtype=self.dtype)
        dft_model_data = DFTModelData(
            image=data.image,
            lmn=data.lmn,
            gains=gains
        )
        vis_model = dft_predict.predict(
            dft_model_data=dft_model_data,
            visibility_coords=data.visibility_coords,
            freqs=data.freqs
        )
        residual = (vis_model - data.obs_vis) / data.obs_vis_weight
        residual = residual.ravel()
        residual = jnp.concatenate([residual.real, residual.imag])
        return residual

    @property
    def float_dtype(self):
        # Given self.dtype is complex, find float dtype
        return jnp.real(jnp.zeros((), dtype=self.dtype)).dtype

    def get_init_params(self, num_source: int, num_time: int, num_ant: int, num_chan: int) -> CalibrationParams:
        """
        Get initial parameters.

        Args:
            num_source: number of sources
            num_time: number of times
            num_ant: number of antennas
            num_chan: number of channels

        Returns:
            initial parameters: (gains_real, gains_imag) of shape (num_source, num_time, num_ant, num_chan, 2, 2)
        """
        return CalibrationParams(
            gains_real=jnp.tile(jnp.eye(2, dtype=self.float_dtype)[None, None, None, None, ...],
                                (num_source, num_time, num_ant, num_chan, 1, 1)),
            gains_imag=jnp.tile(jnp.zeros((2, 2), dtype=self.float_dtype)[None, None, None, None, ...],
                                (num_source, num_time, num_ant, num_chan, 1, 1))
        )

    def solve(self, init_params: CalibrationParams, data: CalibrationData) -> Tuple[CalibrationParams, jaxopt.OptStep]:
        ravel_fn, unravel_fn = pytree_unravel(init_params)

        solver = jaxopt.LBFGS(
            fun=lambda x, *args, **kwargs: self._objective_fun(unravel_fn(x), *args, **kwargs),
            maxiter=1000,
            jit=False,
            unroll=False,
            use_gamma=True
        )

        def body_fn(carry, x):
            params, state = carry
            params, state = solver.update(params=params, state=state, data=data)
            return (params, state), state.value

        carry = (init_params, solver.init_state(init_params=init_params, data=data))

        (params, _), results = lax.scan(body_fn, carry, xs=jnp.arange(self.num_iterations))

        params = unravel_fn(params)
        return params, results
