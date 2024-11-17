import asyncio
import logging
import os
from functools import partial
from typing import NamedTuple, List

import jax
import numpy as np
from jax import numpy as jnp
from ray import serve
from ray.serve.handle import DeploymentHandle

from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.wgridder import vis_to_image
from dsa2000_cal.forward_models.streaming.distributed.calibrator import CalibratorResponse
from dsa2000_cal.forward_models.streaming.distributed.common import ForwardModellingRunParams

logger = logging.getLogger('ray')


class GridderResponse(NamedTuple):
    image: np.ndarray  # [npix_l, npix_m]
    psf: np.ndarray  # [npix_l, npix_m]


@serve.deployment
class Gridder:
    """
    Performs gridding of visibilities per sub-band. Grid a number of solution intervals per image.

    A single output image is formed from (sol_int_time_per_image, sol_int_freq_per_sub_band) solution interval chunks.
    The output shape is [num_l, num_m, num_corrs].
    """

    def __init__(self, params: ForwardModellingRunParams, calibrator: DeploymentHandle):
        self.params = params
        self._calibrator = calibrator
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'gridder')
        os.makedirs(self.params.plot_folder, exist_ok=True)

        self.uvw = np.zeros(
            (
                params.chunk_params.num_sol_ints_per_accumlate,
                params.chunk_params.num_times_per_sol_int,
                params.chunk_params.num_baselines,
                3
            ),
            dtype=np.float64
        )

        self.freqs = np.zeros(
            (
                params.chunk_params.num_sol_ints_per_sub_band,
                params.chunk_params.num_freqs_per_sol_int
            ),
            dtype=np.float64
        )

        vis_shape = (
            params.chunk_params.num_sol_ints_per_accumlate,
            params.chunk_params.num_times_per_sol_int,
            params.chunk_params.num_baselines,
            params.chunk_params.num_sol_ints_per_sub_band,
            params.chunk_params.num_freqs_per_sol_int
        )
        if self.params.full_stokes:
            vis_shape = vis_shape + (2, 2)

        self.visibilities = np.zeros(vis_shape, dtype=np.complex128)
        self.weights = np.zeros(vis_shape, dtype=np.float16)
        self.flags = np.zeros(vis_shape, dtype=np.bool_)

        if self.params.full_stokes:
            vis_mapping = "[Sa,Ts,B,Sb,Fs,P,Q]"
            output_mapping = "[~Nl,~Nm,P,Q]"
        else:
            vis_mapping = "[Sa,Ts,B,Sb,Fs]"
            output_mapping = "[~Nl,~Nm]"

        @partial(
            multi_vmap,
            in_mapping=f"[Sa,Ts,B,3],{vis_mapping},{vis_mapping},{vis_mapping},[Sb,Fs]",
            out_mapping=f"{output_mapping}",
            verbose=True
        )
        def compute_image(uvw, vis, weights, flags, freqs):
            weights = jnp.where(flags, 0.0, weights)

            num_rows = self.params.chunk_params.num_times_per_accumulate * self.params.chunk_params.num_baselines
            num_chan = self.params.chunk_params.num_freqs_per_sub_band

            uvw = uvw.reshape((num_rows, 3))
            vis = vis.reshape((num_rows, num_chan))
            wgt = weights.reshape((num_rows, num_chan))
            freqs = freqs.reshape((num_chan,))

            image = vis_to_image(
                uvw=uvw,
                vis=vis,
                wgt=wgt,
                freqs=freqs,
                pixsize_l=quantity_to_jnp(self.params.image_params.dl, 'rad'),
                pixsize_m=quantity_to_jnp(self.params.image_params.dm, 'rad'),
                center_l=quantity_to_jnp(self.params.image_params.l0, 'rad'),
                center_m=quantity_to_jnp(self.params.image_params.m0, 'rad'),
                epsilon=self.params.image_params.epsilon,
                npix_l=self.params.image_params.num_l,
                npix_m=self.params.image_params.num_m
            )

            # Can't use the shortcut here because of weighting.
            psf = vis_to_image(
                uvw=uvw,
                vis=jnp.ones_like(vis),
                wgt=wgt,
                freqs=freqs,
                pixsize_l=quantity_to_jnp(self.params.image_params.dl, 'rad'),
                pixsize_m=quantity_to_jnp(self.params.image_params.dm, 'rad'),
                center_l=quantity_to_jnp(self.params.image_params.image_params.l0, 'rad'),
                center_m=quantity_to_jnp(self.params.image_params.image_params.m0, 'rad'),
                epsilon=self.params.image_params.epsilon,
                npix_l=self.params.image_params.num_l,
                npix_m=self.params.image_params.num_m
            )

            return image, psf

        self.compute_image_jit = jax.jit(compute_image)

    async def __call__(self, accumulate_idx: int, sub_band_idx: int) -> GridderResponse:

        # Get sol_ints in image_idx and sub_band_idx
        sol_int_time_idxs = accumulate_idx * self.params.chunk_params.num_sol_ints_per_accumlate + np.arange(
            self.params.chunk_params.num_sol_ints_per_accumlate)
        sol_int_freq_idxs = sub_band_idx * self.params.chunk_params.num_sol_ints_per_sub_band + np.arange(
            self.params.chunk_params.num_sol_ints_per_sub_band)
        T, F = np.meshgrid(sol_int_time_idxs, sol_int_freq_idxs, indexing='ij')
        sol_idxs = np.ravel_multi_index(
            (T, F),
            (self.params.chunk_params.num_sol_ints_time_per_image, self.params.chunk_params.num_sol_ints_freq_per_image)
        ).flatten().tolist()

        sol_slice_map = dict(
            (s, (t, f))
            for s, t, f in zip(sol_idxs, T.flatten(), F.flatten())
        )  # sol_idx -> (sol_int_time_idx, sol_int_freq_idx)

        async def get_cal_results(sol_idxs: List[int]):
            responses: List[CalibratorResponse] = await asyncio.gather(
                *[self._calibrator.remote(sol_idx) for sol_idx in sol_idxs]
            )
            for sol_idx, cal_response in enumerate(responses):
                sol_int_time_idx, sol_int_freq_idx = sol_slice_map[sol_idx]
                self.visibilities[sol_int_time_idx, :, sol_int_freq_idx, :, :] = cal_response.visibilities
                self.weights[sol_int_time_idx, :, sol_int_freq_idx, :, :] = cal_response.weights
                self.flags[sol_int_time_idx, :, sol_int_freq_idx, :, :] = cal_response.flags
                self.uvw[sol_int_time_idx, :, :] = cal_response.uvw
                self.freqs[sol_int_freq_idx, :] = cal_response.freqs

        await get_cal_results(sol_idxs)
        image, psf = self.compute_image_jit(
            jnp.asarray(self.uvw, mp_policy.length_dtype),
            jnp.asarray(self.visibilities, mp_policy.vis_dtype),
            jnp.asarray(self.weights, mp_policy.weight_dtype),
            jnp.asarray(self.freqs, mp_policy.freq_dtype)
        )
        yield GridderResponse(
            image=np.asarray(image),
            psf=np.asarray(psf)
        )
