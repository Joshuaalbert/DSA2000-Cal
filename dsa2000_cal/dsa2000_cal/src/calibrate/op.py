from dataclasses import dataclass
from typing import Literal, NamedTuple, Tuple, Callable, TypeVar

import jax
import jaxopt
from jax import lax
from jax import numpy as jnp, Array

from dsa2000_cal.src.dft_predict.op import DFTModelData, DFTPredict
from dsa2000_cal.src.common.vec_ops import VisibilityCoords

V = TypeVar('V')


def pytree_unravel(example_tree: V) -> Tuple[Callable[[V], Array], Callable[[Array], V]]:
    """
    Returns functions to ravel and unravel a pytree.

    Returns:
        ravel_fun: a function to ravel a pytree.
        unravel_fun: a function to unravel a pytree.
    """
    leaf_list, tree_def = jax.tree_util.tree_flatten(example_tree)

    sizes = [leaf.size for leaf in leaf_list]
    shapes = [leaf.shape for leaf in leaf_list]

    def ravel_fun(pytree):
        leaf_list, tree_def = jax.tree_util.tree_flatten(pytree)
        return jnp.concatenate([leaf.ravel() for leaf in leaf_list])

    def unravel_fun(flat_array):
        leaf_list = []
        start = 0
        for size, shape in zip(sizes, shapes):
            leaf_list.append(flat_array[start:start + size].reshape(shape))
            start += size
        return jax.tree_util.tree_unflatten(tree_def, leaf_list)

    return ravel_fun, unravel_fun


class CalibrationParams(NamedTuple):
    gains_real: jnp.ndarray  # [source, time, ant, chan, 2, 2]
    gains_imag: jnp.ndarray  # [source, time, ant, chan, 2, 2]


class CalibrationData(NamedTuple):
    visibility_coords: VisibilityCoords
    image: jnp.ndarray  # [source, chan, 2, 2]
    lm: jnp.ndarray  # [source, 2]
    freq: jnp.ndarray  # [chan]
    obs_vis: jnp.ndarray  # [row, chan, 2, 2]
    obs_vis_weight: jnp.ndarray  # [row, chan, 2, 2]


@dataclass(eq=False)
class Calibration:
    convention: Literal['fourier', 'casa'] = 'fourier'
    dtype: jnp.dtype = jnp.complex64
    chunksize: int = 1
    unroll: int = 1

    def _objective_fun(self, params: CalibrationParams, data: CalibrationData) -> jnp.ndarray:
        residuals = self._residual_fun(params=params, data=data)
        return jnp.mean(lax.square(residuals))

    def _residual_fun(self, params: CalibrationParams, data: CalibrationData) -> jnp.ndarray:
        dft_predict = DFTPredict(
            chunksize=self.chunksize,
            dtype=self.dtype,
            convention=self.convention,
            unroll=self.unroll
        )
        gains = jnp.asarray(params.gains_real + 1j * params.gains_imag, dtype=self.dtype)
        model_data = DFTModelData(
            image=data.image,
            lm=data.lm,
            gains=gains
        )
        vis_model = dft_predict.predict(
            model_data=model_data,
            visibility_coords=data.visibility_coords,
            freq=data.freq
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
            jit=True,
            unroll=False,
            use_gamma=True
        )
        # solver = LevenbergMarquardt(
        #     residual_fun=lambda x, *args, **kwargs: self._residual_fun(unravel_fn(x), *args, **kwargs),
        #     maxiter=10000,
        #     jit=True,
        #     unroll=False,
        #     materialize_jac=False,
        #     geodesic=False,
        #     implicit_diff=True,
        #     atol=0.,
        #     rtol=0.,
        #     gtol=1e-3,
        #     stop_criterion='madsen-nielsen'
        # )
        opt_result = solver.run(init_params=ravel_fn(init_params), data=data)
        params = unravel_fn(opt_result.params)
        return params, opt_result
