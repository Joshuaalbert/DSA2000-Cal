import asyncio
import logging
import os
from typing import NamedTuple, AsyncGenerator, List

import jax
import numpy as np
from astropy import units as au
from jax import numpy as jnp
from ray import serve
from ray.serve.handle import DeploymentHandle

from dsa2000_cal.adapter.utils import broadcast_translate_corrs
from dsa2000_cal.common.fits_utils import ImageModel, save_image_to_fits
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_cal.forward_models.streaming.distributed.gridder import GridderResponse
from dsa2000_cal.imaging.base_imagor import fit_beam

logger = logging.getLogger('ray')


class CallerResponse(NamedTuple):
    image_idx: int
    image_path: str
    psf_path: str


@serve.deployment
class Caller:
    def __init__(self, params: ForwardModellingRunParams, gridder: DeploymentHandle):
        self.params = params
        self._gridder = gridder
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'caller')
        os.makedirs(self.params.plot_folder, exist_ok=True)

        self._num_coh = len(self.params.ms_meta.coherencies)

        self._image = np.zeros(
            (params.image_params.num_l, params.image_params.num_m, params.chunk_params.num_sub_bands, self._num_coh),
            dtype=np.float64)
        self._psf = np.zeros(
            (params.image_params.num_l, params.image_params.num_m, params.chunk_params.num_sub_bands, self._num_coh),
            dtype=np.float64)

        self._fit_beam_jit = jax.jit(fit_beam)

    def save_image_to_fits(self, image_idx: int):
        image_path = os.path.join(self.params.plot_folder, f"{self.params.run_name}_{image_idx:03d}_image.fits")
        psf_path = os.path.join(self.params.plot_folder, f"{self.params.run_name}_{image_idx:03d}_psf.fits")

        if self.params.ms_meta.coherencies not in [('I',), ('I', 'Q'), ('I', 'Q', 'U'), ('I', 'Q', 'U', 'V')]:
            image = au.Quantity(
                np.asarray(broadcast_translate_corrs(
                    jnp.asarray(self._image),
                    self.params.ms_meta.coherencies, ('I', 'Q', 'U', 'V')
                )),
                'Jy'
            )
            psf = au.Quantity(
                np.asarray(broadcast_translate_corrs(
                    jnp.asarray(self._psf),
                    self.params.ms_meta.coherencies, ('I', 'Q', 'U', 'V')
                )), 'Jy'
            )
            coherencies = ('I', 'Q', 'U', 'V')
        else:
            image = au.Quantity(self._image, 'Jy')
            psf = au.Quantity(self._psf, 'Jy')
            coherencies = self.params.ms_meta.coherencies

        bandwidth = self.params.ms_meta.channel_width * len(self.params.ms_meta.freqs)

        num_freqs_per_sub_band = self.params.chunk_params.num_freqs_per_sol_int * self.params.chunk_params.num_sol_ints_per_sub_band
        central_freqs = au.Quantity(
            [
                np.mean(self.params.ms_meta.freqs[i:i + num_freqs_per_sub_band])
                for i in range(0, len(self.params.ms_meta.freqs), num_freqs_per_sub_band)
            ]
        )

        # Fit beam on Stokes I, on central sub band.

        major, minor, posang = self._fit_beam_jit(
            psf=psf[:, :, self.params.chunk_params.num_sub_bands // 2, 0],
            dl=quantity_to_jnp(self.params.image_params.dl, 'rad'),
            dm=quantity_to_jnp(self.params.image_params.dm, 'rad')
        )

        image_model = ImageModel(
            phase_tracking=self.params.ms_meta.phase_tracking,
            obs_time=self.params.ms_meta.ref_time,
            dl=self.params.image_params.dl,
            dm=self.params.image_params.dm,
            freqs=central_freqs,
            bandwidth=bandwidth,
            coherencies=coherencies,
            beam_major=np.asarray(major) * au.rad,
            beam_minor=np.asarray(minor) * au.rad,
            beam_pa=np.asarray(posang) * au.rad,
            unit='JY/PIXEL',
            object_name=self.params.run_name,
            image=image
        )
        save_image_to_fits(image_path, image_model=image_model, overwrite=True)
        image_model.image = psf
        save_image_to_fits(psf_path, image_model=image_model, overwrite=True)
        return CallerResponse(
            image_idx=image_idx,
            image_path=image_path,
            psf_path=psf_path
        )

    async def __call__(self) -> AsyncGenerator[CallerResponse, None]:
        async def get_image_subbands(accumulate_idx: int):
            sub_band_idxs = list(range(self.params.chunk_params.num_sub_bands))
            # Submit all sub bands for gridding, and wait for all to complete.
            responses: List[GridderResponse] = await asyncio.gather(
                *[self._gridder.remote(accumulate_idx, sub_band_idx) for sub_band_idx in sub_band_idxs]
            )
            for sub_band_idx, gridder_response in enumerate(responses):
                self._image[:, :, sub_band_idx, :] = gridder_response.image
                self._psf[:, :, sub_band_idx, :] = gridder_response.psf

        # Response generator can be used in an `async for` block.
        for accumulate_idx in range(self.params.chunk_params.num_images_time):
            await get_image_subbands(accumulate_idx)
            yield self.save_image_to_fits(accumulate_idx)
