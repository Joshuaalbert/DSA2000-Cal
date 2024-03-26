from dataclasses import dataclass
from typing import NamedTuple, Literal

import jax

from dsa2000_cal.src.common.vec_ops import two_pi_over_c, minus_two_pi_over_c, kron_product, VisibilityCoords

jax.config.update('jax_threefry_partitionable', True)

from jax import numpy as jnp, pmap

from jax.sharding import PartitionSpec

P = PartitionSpec

from dsa2000_cal.src.common.jax_utils import cumulative_op_static, add_chunk_dim


class RFIModelData(NamedTuple):
    """
    Data for predict.
    """
    image: jnp.ndarray  # [source, chan, 2, 2]
    gains: jnp.ndarray  # [source, time, ant, chan, 2, 2]
    lm: jnp.ndarray  # [source, 2]


@dataclass(eq=False)
class RFIPredict:
    """
    Class to predict visibilities from an image and gains.
    """
    convention: Literal['fourier', 'casa'] = 'fourier'
    dtype: jnp.dtype = jnp.complex64
    chunksize: int = 1
    unroll: int = 1

    def _compute_visibilty(self, lm, uvw, g1, g2, freq, image):
        """
        Compute the visibility for a single direction.

        Args:
            lm: [2]
            uvw: [3]
            g1: [2, 2]
            g2: [2, 2]
            freq: scalar
            image: [2]

        Returns:
            [2, 2] visibility in given direction for given baseline
        """
        if self.convention == 'fourier':
            constant = minus_two_pi_over_c
        elif self.convention == 'casa':
            constant = two_pi_over_c
        else:
            raise ValueError("convention not in ('fourier', 'casa')")

        u, v, w = uvw

        l, m = lm  # [scalar]
        n = jnp.sqrt(1. - l ** 2 - m ** 2)  # [scalar]
        # -2*pi*freq/c*(l*u + m*v + (n-1)*w)
        delay = l * u + m * v + (n - 1.) * w  # [scalar]

        phi = jnp.asarray(
            1.0j * (delay * constant * freq),
            dtype=self.dtype
        )  # [chan]
        fringe = (jnp.exp(phi) / n)
        return fringe * kron_product(g1, image, g2.T.conj())

    def predict(
            self,
            model_data: RFIModelData,
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
            visibilities: [row, chan, 2, 2]
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
        # TODO: compare with distributing over channels.

        # On each replica we will sum over source.

        def replica(vis_coords: VisibilityCoords):

            class Accumulate(NamedTuple):
                vis: jnp.ndarray  # [row, chan, 2, 2]

            def accumulate_op(accumulate: Accumulate, xs: RFIModelData):
                """
                Computes the visibilities for a given set of visibility coordinates for a single direction and adds to
                the current set of accumulated visibilities.

                Args:
                    accumulate: the current accumulated visibilities (over direction).
                    xs: the data for this chunk.

                Returns:
                    accumulated visibilities
                """
                g1 = xs.gains[vis_coords.time_idx, vis_coords.antenna_1, :, :, :]  # [row, chan, 2, 2]
                g2 = xs.gains[vis_coords.time_idx, vis_coords.antenna_2, :, :, :]  # [row, chan, 2, 2]

                # Data could be sharded over: frequency, time so don't reduce over these dimensions, or else
                # communication happens.
                # The args are lm, uvw, g1, g2, freq, image

                # Over chan -> [chan, ...]
                _compute_visibilty = jax.vmap(self._compute_visibilty, in_axes=[None, None, 1, 1, 0, 0])

                # Over row -> [row, chan, ...]
                _compute_visibilty = jax.vmap(_compute_visibilty, in_axes=[None, 0, 0, 0, None, None])

                vis_s = _compute_visibilty(
                    lm=xs.lm, uvw=vis_coords.uvw, g1=g1, g2=g2, freq=freq, image=xs.image
                )  # [row, chan, 2, 2]

                # TODO: Best to make sure sharding is still the same here.

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
            chunked_visibility_coords, unchunk_fn = add_chunk_dim(visibility_coords, chunk_size=self.chunksize)
            replice_pmap = pmap(
                fun=replica
            )
            chunked_visibilities = replice_pmap(chunked_visibility_coords)
            visibilities = unchunk_fn(chunked_visibilities)
        return visibilities
