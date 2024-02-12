from dataclasses import dataclass
from typing import Literal, NamedTuple, Tuple, Callable, TypeVar

import jax
import jaxopt
import numpy as np
from jax import lax, pmap
from jax import numpy as jnp, Array
from jax._src.numpy.lax_numpy import ndim, shape
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec

from dsa2000_cal.jax_utils import add_chunk_dim, cumulative_op_static, pad_to_chunksize

# Lightspeed
c = 2.99792458e8

two_pi_over_c = 2 * jnp.pi / c
minus_two_pi_over_c = -two_pi_over_c

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


class ScalarGain(NamedTuple):
    gain: jnp.ndarray  # scalar


class DiagonalGain(NamedTuple):
    gain: jnp.ndarray  # [2] (the diagonal)


class FullGain(NamedTuple):
    gain: jnp.ndarray  # [2, 2]


def vec(a):
    return jnp.ravel(a.T)


def unvec(a, shape: Tuple[int, ...] | None = None):
    if shape is None:
        # assume square
        n = int(np.sqrt(a.shape[-1]))
        if n * n != a.shape[-1]:
            raise ValueError(f"a is not square. Can't infer unvec shape.")
        shape = (n, n)
    return jnp.reshape(a, shape).T


def kron_product(a, b, c):
    # return unvec(kron(c.T, a) @ vec(b), (a.shape[0], c.shape[1]))
    # Fewer bytes accessed, better utilisation (2x as many flops though -- which is better than memory access)
    return unvec(jnp.sum(kron(c.T, a) * vec(b), axis=-1), (a.shape[0], c.shape[1]))


class GeodesicCoord(NamedTuple):
    """
    Coordinates for a single geodesic.
    """
    direction_cosines: jnp.ndarray  # [2] the direction cosines l,m
    antenna: jnp.ndarray  # [3] the antenna coordinates
    time: jnp.ndarray
    frequency: jnp.ndarray


class VisibilityCoords(NamedTuple):
    """
    Coordinates for a single visibility.
    """
    uvw: jnp.ndarray  # [rows, 3] the uvw coordinates
    time: jnp.ndarray  # [rows] the time
    antenna_1: jnp.ndarray  # [rows] the first antenna
    antenna_2: jnp.ndarray  # [rows] the second antenna
    time_idx: jnp.ndarray  # [rows] the time index


class ModelData(NamedTuple):
    """
    Data for predict.
    """
    image: jnp.ndarray  # [source, chan, 2, 2]
    gains: jnp.ndarray  # [source, time, ant, chan, 2, 2]
    lm: jnp.ndarray  # [source, 2]


@dataclass(eq=False)
class DFTPredict:
    """
    Class to predict visibilities from an image and gains.
    """
    convention: Literal['fourier', 'casa'] = 'casa'
    dtype: jnp.dtype = jnp.complex64
    chunksize: int = 1
    unroll: int = 1
    use_pjit: bool = True

    def predict(
            self,
            model_data: ModelData,
            visibility_coords: VisibilityCoords,
            freq: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Convert an image to visibilities.

        Args:
            model_data:
            visibility_coords:
            freq: [chan] freqs in Hz

        Returns:

        """
        if not jnp.iscomplexobj(model_data.image):
            raise ValueError(f"Image should be complex type.")
        if not jnp.iscomplexobj(model_data.gains):
            raise ValueError(f"Gains should be complex type.")

        if self.convention == 'fourier':
            constant = minus_two_pi_over_c
        elif self.convention == 'casa':
            constant = two_pi_over_c
        else:
            raise ValueError("convention not in ('fourier', 'casa')")

        # We will distribute rows over devices.
        # On each device, we will sum over source.

        def replica(vis_coords: VisibilityCoords):
            u, v, w = vis_coords.uvw.T  # [row]

            class Accumulate(NamedTuple):
                vis: jnp.ndarray  # [row, chan, 2, 2]

            def accumulate_op(accumulate: Accumulate, xs: ModelData):
                """
                Computes the visibilities for a given set of visibility coordinates for a single direction and adds to the
                current set of accumulated visibilities.

                Args:
                    accumulate: the current accumulated visibilities (over direction).
                    xs: the data for this chunk.

                Returns:

                """
                g1 = xs.gains[vis_coords.time_idx, vis_coords.antenna_1, :, :, :]  # [row, chan, 2, 2]
                g2 = xs.gains[vis_coords.time_idx, vis_coords.antenna_2, :, :, :]  # [row, chan, 2, 2]

                l, m = xs.lm  # [scalar]
                n = jnp.sqrt(1. - l ** 2 - m ** 2)  # [scalar]
                # -2*pi*freq/c*(l*u + m*v + (n-1)*w)
                delay = l * u + m * v + (n - 1.) * w  # [scalar]

                def vis_row_chan(_g1, _g2, _delay):
                    vis_chan = jax.vmap(
                        lambda _g1, _image, _g2, _fringe: _fringe * kron_product(_g1, _image, _g2.T.conj()))

                    phi = jnp.asarray(
                        1.0j * (_delay * constant * freq),
                        dtype=self.dtype
                    )  # [chan]
                    fringe = (jnp.exp(phi) / n)  # [chan]

                    return vis_chan(_g2, xs.image, _g1, fringe)  # [chan, 2, 2]

                vis_s = jax.vmap(vis_row_chan)(g1, g2, delay)  # [row, chan, 2, 2]
                return Accumulate(vis=accumulate.vis + vis_s)

            row = vis_coords.uvw.shape[0]
            chan = freq.shape[0]
            init = Accumulate(vis=jnp.zeros((row, chan, 2, 2), dtype=self.dtype))
            final_accumulate, _ = cumulative_op_static(
                op=accumulate_op, init=init, xs=model_data, unroll=self.unroll
            )
            return final_accumulate.vis

        # Distribute replicas over devices in chunks.

        if self.chunksize == 1:
            visibilities = replica(vis_coords=visibility_coords)
        else:
            if self.use_pjit:
                padded_visibility_coords, remove_extra_fn = pad_to_chunksize(visibility_coords,
                                                                             chunk_size=self.chunksize)
                with Mesh(np.array(jax.devices()), ('row',)):
                    f_pjit = pjit(replica,
                                  in_shardings=PartitionSpec('row'),
                                  out_shardings=PartitionSpec())  # Adjust as needed for your setup
                    # Now, calling f_pjit will distribute the computation and gather results on the host
                    visibilities = f_pjit(padded_visibility_coords)
                visibilities = remove_extra_fn(visibilities)
            else:
                chunked_visibility_coords, unchunk_fn = add_chunk_dim(visibility_coords, chunk_size=self.chunksize)
                replice_pmap = pmap(
                    fun=replica
                )
                chunked_visibilities = replice_pmap(chunked_visibility_coords)
                visibilities = unchunk_fn(chunked_visibilities)
        return visibilities


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
    convention: Literal['fourier', 'casa'] = 'casa'
    dtype: jnp.dtype = jnp.complex64
    chunksize: int = 1
    unroll: int = 1
    use_pjit: bool = True

    def _objective_fun(self, params: CalibrationParams, data: CalibrationData) -> jnp.ndarray:
        residuals = self._residual_fun(params=params, data=data)
        return jnp.mean(lax.square(residuals))

    def _residual_fun(self, params: CalibrationParams, data: CalibrationData) -> jnp.ndarray:
        dft_predict = DFTPredict(
            chunksize=self.chunksize,
            dtype=self.dtype,
            convention=self.convention,
            unroll=self.unroll,
            use_pjit=self.use_pjit
        )
        gains = jnp.asarray(params.gains_real + 1j * params.gains_imag, dtype=self.dtype)
        model_data = ModelData(
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
            initial parameters
        """
        return CalibrationParams(
            gains_real=jnp.tile(jnp.eye(2, dtype=self.float_dtype)[None, None, None, None, ...],
                                (num_source, num_time, num_ant, num_chan, 1, 1)),
            gains_imag=jnp.tile(jnp.zeros((2, 2), dtype=self.float_dtype)[None, None, None, None, ...],
                                (num_source, num_time, num_ant, num_chan, 1, 1))
        )
        # return CalibrationParams(
        #     gains_real=jnp.ones((num_source, num_time, num_ant, num_chan, 2, 2), self.float_dtype),
        #     # [source, time, ant, chan, 2, 2]
        #     gains_imag=jnp.zeros((num_source, num_time, num_ant, num_chan, 2, 2), self.float_dtype)
        #     # [source, time, ant, chan, 2, 2]
        # )

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


def kron(a, b):
    """
    Compute the Kronecker product of two arrays.

    Args:
        a: [n, m]
        b: [p, q]

    Returns:
        [n*p, m*q]
    """
    if ndim(a) < ndim(b):
        a = lax.expand_dims(a, range(ndim(b) - ndim(a)))
    elif ndim(b) < ndim(a):
        b = lax.expand_dims(b, range(ndim(a) - ndim(b)))
    a_reshaped = lax.expand_dims(a, range(1, 2 * ndim(a), 2))
    b_reshaped = lax.expand_dims(b, range(0, 2 * ndim(b), 2))
    out_shape = tuple(np.multiply(shape(a), shape(b)))
    return lax.reshape(lax.mul(a_reshaped, b_reshaped), out_shape)
