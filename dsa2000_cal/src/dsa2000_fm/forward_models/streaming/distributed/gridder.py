import asyncio
import itertools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import NamedTuple

import numpy as np
import pylab as plt
import ray

from dsa2000_cal.common.array_types import FloatArray, ComplexArray, BoolArray
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.quantity_utils import quantity_to_np
from dsa2000_cal.common.ray_utils import TimerLog, memory_logger
from dsa2000_cal.common.wgridder import vis_to_image_np
from dsa2000_fm.forward_models.streaming.distributed.calibrator import CalibratorResponse
from dsa2000_fm.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_fm.forward_models.streaming.distributed.supervisor import Supervisor

logger = logging.getLogger('ray')


class GridderResponse(NamedTuple):
    image: np.ndarray  # [npix_l, npix_m[,2,2]]
    psf: np.ndarray  # [npix_l, npix_m[,2,2]]


def compute_gridder_options(run_params: ForwardModellingRunParams):
    # Memory is 2 * num_pix^2 * num_coh * itemsize(image)
    num_coh = 4 if run_params.full_stokes else 1
    num_pix_l = run_params.image_params.num_l
    num_pix_m = run_params.image_params.num_m
    # image is f64
    itemsize_image = np.dtype(np.float64).itemsize
    memory = 2 * num_pix_l * num_pix_m * num_coh * itemsize_image
    return {
        "num_cpus": run_params.chunk_params.num_freqs_per_sol_int,  # 1 thread per channel * (num coh)
        "num_gpus": 0,  # Doesn't use GPU
        'memory': 1.1 * memory
    }


@ray.remote
class Gridder:
    """
    Performs gridding of visibilities per solution interval.
    """

    def __init__(self, params: ForwardModellingRunParams, calibrator: Supervisor[CalibratorResponse]):
        self.params = params
        self._calibrator = calibrator
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'gridder')
        os.makedirs(self.params.plot_folder, exist_ok=True)
        self._initialised = False
        self._memory_logger_task: asyncio.Task | None = None

    async def init(self):
        if self._initialised:
            return
        self._initialised = True
        self._memory_logger_task = asyncio.create_task(memory_logger(task='gridder', cadence=timedelta(seconds=5)))

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
            # Add extra axes
            visibilities = visibilities[..., None, None]
            weights = weights[..., None, None]
            flags = flags[..., None, None]
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
        results = list(results)
        if np.all(image_buffer == 0) or not np.all(np.isfinite(image_buffer)):
            logger.warning(f"Image buffer is all zeros or contains NaNs/Infs for freq_idx={sol_int_freq_idx}")
        if np.all(psf_buffer == 0) or not np.all(np.isfinite(psf_buffer)):
            logger.warning(f"PSF buffer is all zeros or contains NaNs/Infs for freq_idx={sol_int_freq_idx}")

        # Plot histogram of image
        fig, ax = plt.subplots(1, 1)
        ax.hist(image_buffer.flatten(), bins='auto')
        ax.set_title(f"Image buffer histogram for freq_idx={sol_int_freq_idx}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        fig.savefig(os.path.join(self.params.plot_folder, f"image_buffer_hist_{sol_int_freq_idx}.png"))
        plt.close(fig)

        if self.params.full_stokes:
            return GridderResponse(image=image_buffer, psf=psf_buffer)
        else:
            # remove the last dimensions
            return GridderResponse(image=image_buffer[..., 0, 0], psf=psf_buffer[..., 0, 0])

    async def __call__(self, key, sol_int_time_idx: int, sol_int_freq_idx: int) -> GridderResponse:
        logger.info(f"Gridding visibilities for time_idx={sol_int_time_idx} and freq_idx={sol_int_freq_idx}")
        await self.init()

        with TimerLog("Getting bright source subtracted data..."):
            cal_response = await self._calibrator(key, sol_int_time_idx, sol_int_freq_idx)
            visibilities = np.asarray(cal_response.visibilities, order='F')
            weights = np.asarray(cal_response.weights, order='F')
            flags = np.asarray(cal_response.flags, order='F')
            uvw = np.asarray(cal_response.uvw, order='F')
            del cal_response

        with TimerLog("Gridding..."):
            return block_until_ready(self._grid_vis(
                sol_int_freq_idx=sol_int_freq_idx,
                uvw=uvw,
                visibilities=visibilities,
                weights=weights,
                flags=flags
            ))
