from dataclasses import dataclass
from typing import NamedTuple, Literal

import jax
from jax import numpy as jnp, pmap

from dsa2000_cal.src.common.jax_utils import cumulative_op_static, add_chunk_dim
from dsa2000_cal.src.common.vec_ops import two_pi_over_c, minus_two_pi_over_c, kron_product, VisibilityCoords


class FFTModelData(NamedTuple):
    """
    Data for predict.
    """
    image: jnp.ndarray  # [n_l, n_l, chan, 2, 2]
    image_lm_rad: jnp.ndarray  # [n_l, n_l, 2]
    gains: jnp.ndarray  # [num_l, num_m, time, ant, chan, 2, 2]
    gains_lm_rad: jnp.ndarray  # [num_l, num_m, 2]


@dataclass(eq=False)
class WProjKernel:
    """
    Class for constructing the W-projection kernel
    """
    convention: Literal['fourier', 'casa'] = 'fourier'
    dtype: jnp.dtype = jnp.complex64

    def kernel(self, l: jnp.ndarray, m: jnp.ndarray, w: jnp.ndarray, freq: jnp.ndarray) -> jnp.ndarray:
        if self.convention == 'fourier':
            constant = minus_two_pi_over_c
        elif self.convention == 'casa':
            constant = two_pi_over_c
        else:
            raise ValueError("convention not in ('fourier', 'casa')")
        n = jnp.sqrt(1. - l ** 2 - m ** 2)
        phase = constant * freq * w * (n - 1.)
        return jnp.exp(jnp.asarray(1.0j * phase, self.dtype))


def test_nested_vmap():
    def f(a, b, c, d):
        return a + b + c + d

    # Reversed order
    f = jax.vmap(f, in_axes=(None, None, None, 0))
    f = jax.vmap(f, in_axes=(None, None, 0, None))
    f = jax.vmap(f, in_axes=(0, 0, None, None))

    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    c = jnp.array([7, 8, 9, 10])
    d = jnp.array([11, 12, 13, 14, 15])

    assert f(a, b, c, d).shape == (3, 4, 5)

    expected = (a + b)[:, None, None] + c[None, :, None] + d[None, None, :]
    assert jnp.all(f(a, b, c, d) == expected)


@dataclass(eq=False)
class FFTPredict:
    """
    Class to predict visibilities from an image and gains using FFT and degridding.

    1. V_G(u, v, 0) = FFT(I(l,m)/T(l,m)) for 'fourier' convention, n**2 * IFFT(I(l,m)*G(l,m)) for 'casa' convention.
    2. V_D(u, v, w) = Conv(V_G(u, v, 0), H
    """
    convention: Literal['fourier', 'casa'] = 'fourier'
    dtype: jnp.dtype = jnp.complex64
    chunksize: int = 1
    unroll: int = 1

    def predict(
            self,
            model_data: FFTModelData,
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

        n = jnp.sqrt(1. - jnp.sum(jnp.square(model_data.image_lm_rad), axis=-1))  # [n_l, n_m]
        image = model_data.image / n[:, :, None, None, None]  # [n_l, n_m, chan, 2, 2]
        n_l, n_m, chan, _, _ = image.shape
        image_dlm_rad = jnp.array([
            model_data.image_lm_rad[1, 0, 0] - model_data.image_lm_rad[0, 0, 0],
            model_data.image_lm_rad[0, 1, 1] - model_data.image_lm_rad[0, 0, 1]
        ])  # [2]

        # uv_max = 0.5 * jnp.array([n_l, n_m]) * duv
        image_u = jnp.fft.ifftshift(jnp.fft.fftfreq(n_l, image_dlm_rad[0]))
        image_v = jnp.fft.ifftshift(jnp.fft.fftfreq(n_m, image_dlm_rad[1]))

        image_l = model_data.image_lm_rad[:, 0]  # [n_l]
        image_m = model_data.image_lm_rad[:, 1]  # [n_m]

        w = visibility_coords.uvw[:, 2]  # [row]

        w_proj_kernel = WProjKernel(convention=self.convention, dtype=self.dtype).kernel
        w_proj_kernel = jax.vmap(w_proj_kernel, in_axes=(None, None, None, 0))
        w_proj_kernel = jax.vmap(w_proj_kernel, in_axes=(None, None, 0, None))
        w_proj_kernel = jax.vmap(w_proj_kernel, in_axes=(0, 0, None, None))

        w_kernel = w_proj_kernel(image_l, image_m, w, freq)  # [n_l, n_m, chan]

        if self.convention == 'fourier':
            image_scale = jnp.prod(image_dlm_rad)
            V_G = jnp.fft.fftshift(jnp.fft.fft2(image, axes=(0, 1))) * image_scale  # [n_l, n_m, chan, 2, 2]
            # H_ij = kron(g_j(l,m).T.conj, g_i(l,m)) * W_proj(l, m; w)
            constant = minus_two_pi_over_c
            w_proj = jnp.exp(constant * freq * w * (n[:, :, None] - 1.))  # [n_l, n_m, chan]
        elif self.convention == 'casa':
            # constant = two_pi_over_c
            image_scale = jnp.prod(image_dlm_rad) * n_l * n_m
            V_G = jnp.fft.ifft2(jnp.fft.ifftshift(image), axes=(0, 1)) * image_scale  # [n_l, n_m, chan, 2, 2]
        else:
            raise ValueError("convention not in ('fourier', 'casa')")

        # We will distribute rows over devices.
        # On each device, we will sum over source.

        def replica(vis_coords: VisibilityCoords):
            u, v, w = vis_coords.uvw.T  # [row]

            class Accumulate(NamedTuple):
                vis: jnp.ndarray  # [row, chan, 2, 2]

            def accumulate_op(accumulate: Accumulate, xs: FFTModelData):
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
            chunked_visibility_coords, unchunk_fn = add_chunk_dim(visibility_coords, chunk_size=self.chunksize)
            replice_pmap = pmap(
                fun=replica
            )
            chunked_visibilities = replice_pmap(chunked_visibility_coords)
            visibilities = unchunk_fn(chunked_visibilities)
        return visibilities
