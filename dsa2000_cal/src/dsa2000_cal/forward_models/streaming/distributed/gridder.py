import itertools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

from dsa2000_cal.common.array_types import FloatArray, ComplexArray, BoolArray
from dsa2000_cal.common.quantity_utils import quantity_to_np
from dsa2000_cal.common.wgridder import vis_to_image_np
from dsa2000_cal.forward_models.streaming.distributed.calibrator import CalibratorResponse
from dsa2000_cal.forward_models.streaming.distributed.common import ForwardModellingRunParams

logger = logging.getLogger('ray')


class GridderResponse(NamedTuple):
    image: np.ndarray  # [npix_l, npix_m[,2,2]]
    psf: np.ndarray  # [npix_l, npix_m[,2,2]]


@serve.deployment
class Gridder:
    """
    Performs gridding of visibilities per solution interval.
    """

    def __init__(self, params: ForwardModellingRunParams, calibrator: DeploymentHandle):
        self.params = params
        self._calibrator = calibrator
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'gridder')
        os.makedirs(self.params.plot_folder, exist_ok=True)

    def _grid_vis(self, sol_int_freq_idx: int, uvw: FloatArray, visibilities: ComplexArray, weights: FloatArray,
                  flags: BoolArray) -> GridderResponse:
        """
        Grids the visibilities for a single solution interval.

        Args:
            sol_int_freq_idx: the solution interval frequency index
            uvw: [Ts, B, 3] the uvw coordinates
            visibilities: [Ts, B, F[,2,2]] the visibilities
            weights: [Ts, B, F[,2,2]] the weights
            flags: [Ts, B, F[,2,2]] the flags, True means flagged, don't grid.

        Returns:
            the gridded image and psf
        """
        freq_idxs = np.arange(
            self.params.chunk_params.num_freqs_per_sol_int
        ) + sol_int_freq_idx * self.params.chunk_params.num_freqs_per_sol_int
        freqs = quantity_to_np(
            self.params.ms_meta.freqs[freq_idxs]
        )
        num_rows = self.params.chunk_params.num_times_per_sol_int * self.params.chunk_params.num_baselines
        num_chan = self.params.chunk_params.num_freqs_per_sol_int
        if self.params.full_stokes:
            image_buffer = np.zeros((self.params.image_params.num_l, self.params.image_params.num_m, 2, 2),
                                    dtype=np.float32, order='F')
            psf_buffer = np.zeros((self.params.image_params.num_l, self.params.image_params.num_m, 2, 2),
                                  dtype=np.float32, order='F')
            pq_array = itertools.product(range(2), range(2))
        else:
            image_buffer = np.zeros((self.params.image_params.num_l, self.params.image_params.num_m, 1, 1),
                                    dtype=np.float32, order='F')
            psf_buffer = np.zeros((self.params.image_params.num_l, self.params.image_params.num_m, 1, 1),
                                  dtype=np.float32, order='F')
            pq_array = itertools.product(range(1), range(1))

        def single_run(p_idx, q_idx):
            vis_to_image_np(
                uvw=uvw.reshape((num_rows, 3)),
                freqs=freqs,
                vis=visibilities[..., p_idx, q_idx].reshape((num_rows, num_chan)),
                pixsize_m=quantity_to_np(self.params.image_params.dm, 'rad'),
                pixsize_l=quantity_to_np(self.params.image_params.dl, 'rad'),
                center_l=quantity_to_np(self.params.image_params.l0, 'rad'),
                center_m=quantity_to_np(self.params.image_params.m0, 'rad'),
                npix_l=self.params.image_params.num_l,
                npix_m=self.params.image_params.num_m,
                wgt=weights[..., p_idx, q_idx].reshape((num_rows, num_chan)),
                mask=np.logical_not(flags[..., p_idx, q_idx]).reshape((num_rows, num_chan)),
                epsilon=self.params.image_params.epsilon,
                double_precision_accumulation=False,
                scale_by_n=True,
                normalise=True,
                output_buffer=image_buffer[:, :, p_idx, q_idx],
                num_threads=num_chan
            )
            vis_to_image_np(
                uvw=uvw.reshape((num_rows, 3)),
                freqs=freqs,
                vis=visibilities[..., p_idx, q_idx].reshape((num_rows, num_chan)),
                pixsize_m=quantity_to_np(self.params.image_params.dm, 'rad'),
                pixsize_l=quantity_to_np(self.params.image_params.dl, 'rad'),
                center_l=quantity_to_np(self.params.image_params.l0, 'rad'),
                center_m=quantity_to_np(self.params.image_params.m0, 'rad'),
                npix_l=self.params.image_params.num_l,
                npix_m=self.params.image_params.num_m,
                wgt=weights[..., p_idx, q_idx].reshape((num_rows, num_chan)),
                mask=np.logical_not(flags[..., p_idx, q_idx]).reshape((num_rows, num_chan)),
                epsilon=self.params.image_params.epsilon,
                double_precision_accumulation=False,
                scale_by_n=True,
                normalise=True,
                output_buffer=psf_buffer[:, :, p_idx, q_idx],
                num_threads=num_chan
            )

        # TODO: Tune size of thread pool executor.
        #  Note, that each thread internally uses self.params.chunk_params.num_freqs_per_sol_int threads.
        with ThreadPoolExecutor(max_workers=1) as executor:
            results = executor.map(single_run, *zip(*pq_array))
        # Get all results (None's)
        list(results)
        if self.params.full_stokes:
            return GridderResponse(image=image_buffer, psf=psf_buffer)
        else:
            # remove the last dimensions
            return GridderResponse(image=image_buffer[..., 0, 0], psf=psf_buffer[..., 0, 0])

    async def __call__(self, key, sol_int_time_idx: int, sol_int_freq_idx: int) -> GridderResponse:

        cal_response: CalibratorResponse = await self._calibrator.remote(key, sol_int_time_idx, sol_int_freq_idx)

        return self._grid_vis(
            sol_int_freq_idx=sol_int_freq_idx,
            uvw=cal_response.uvw,
            visibilities=cal_response.visibilities,
            weights=cal_response.weights,
            flags=cal_response.flags
        )
