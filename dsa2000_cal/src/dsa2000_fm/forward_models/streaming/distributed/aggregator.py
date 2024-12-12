import asyncio
import logging
import os
from datetime import timedelta
from typing import List, NamedTuple
from typing import Type

import jax
import numpy as np
import ray
from astropy import units as au
from jax import numpy as jnp
from pydantic import Field
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from dsa2000_cal.common.corr_utils import broadcast_translate_corrs
from dsa2000_cal.common.fits_utils import ImageModel, save_image_to_fits
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.ray_utils import resource_logger
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.imaging.base_imagor import fit_beam
from dsa2000_fm.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_fm.forward_models.streaming.distributed.gridder import GridderResponse
from dsa2000_fm.forward_models.streaming.distributed.supervisor import Supervisor
from dsa2000_rcp.actors.namespace import NAMESPACE

logger = logging.getLogger('ray')


class AggregatorParams(SerialisableBaseModel):
    sol_int_freq_idxs: List[int] = Field(
        description="The solution interval frequency indices to use for the aggregation into this sub-band."
    )
    fm_run_params: ForwardModellingRunParams
    gridder: Supervisor[GridderResponse]
    image_suffix: str


class AggregatorResponse(NamedTuple):
    image_path: str | None
    psf_path: str | None


def compute_aggregator_options(run_params: ForwardModellingRunParams):
    # memory is 2 * num_pix^2 * num_coh * itemsize(image)
    num_coh = 4 if run_params.full_stokes else 1
    num_pix_l = run_params.image_params.num_l
    num_pix_m = run_params.image_params.num_m
    # image is f64
    itemsize_image = np.dtype(np.float64).itemsize
    memory = 2 * num_pix_l * num_pix_m * num_coh * itemsize_image
    return {
        "num_cpus": 0,  # Doesn't use CPU
        "num_gpus": 0,  # Doesn't use GPU
        'memory': 1.1 * memory
    }


class Aggregator:
    """
    Collects images from the gridder and aggregates them into a single image for a sub-band.

    Expected memory usage:
        - 9 GB for image [17.5k, 17.5k, 2, 2] (float64)
        - 9 GB for psf [17.5k, 17.5k, 2, 2] (float64)

    Expected compute:
        - negligible
    """

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        return (self._deserialise, (self._serialised_data,))

    @classmethod
    def _deserialise(cls, kwargs):
        # Create a new instance, bypassing __init__ and setting the actor directly
        return cls(**kwargs)

    def __init__(self, worker_id: str, params: AggregatorParams | None = None, num_cpus: int | None = 0,
                 num_gpus: int | None = 0, memory: int | None = 18 * 1024 ** 3):

        self._serialised_data = dict(
            worker_id=worker_id,
            params=None
        )
        actor_name = self.actor_name(worker_id)

        try:
            actor = ray.get_actor(actor_name, namespace=NAMESPACE)
            logger.info(f"Connected to existing {actor_name}")
        except ValueError:
            if params is None:
                raise ValueError(f"Actor {actor_name} does not exist, and params is None")

            placement_node_id = ray.get_runtime_context().get_node_id()

            actor_options = {
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
                "memory": memory,
                "name": actor_name,
                # "lifetime": "detached",
                "max_restarts": -1,
                "max_task_retries": -1,
                # Schedule the controller on the head node with a soft constraint. This
                # prefers it to run on the head node in most cases, but allows it to be
                # restarted on other nodes in an HA cluster.
                "scheduling_strategy": NodeAffinitySchedulingStrategy(placement_node_id, soft=True),
                "namespace": NAMESPACE,
                "max_concurrency": 15000  # Needs to be large, as there should be no limit.
            }

            dynamic_cls = self.dynamic_cls()

            actor_kwargs = dict(
                params=params
            )

            actor = ray.remote(dynamic_cls).options(**actor_options).remote(**actor_kwargs)
            ray.get(actor.health_check.remote())

        self._actor = actor

    @staticmethod
    def dynamic_cls() -> Type:
        """
        Create a dynamic class that will be parsed properly by ray dashboard, so that it has a nice class name.

        Returns:
            a dynamic class
        """
        # a dynamic class that will be parsed properly by ray dashboard, so that it has a nice class name.
        return type(
            f"Aggregator",
            (_Aggregator,),
            dict(_Aggregator.__dict__),
        )

    @staticmethod
    def actor_name(node_id: str) -> str:
        return f"AGGREGATOR#{node_id}"

    def __call__(self, key, sol_int_time_idx: int, save_to_disk: bool) -> AggregatorResponse:
        return ray.get(self._actor.call.remote(key, sol_int_time_idx, save_to_disk))


class _Aggregator:
    def __init__(self, params: AggregatorParams):
        self.params = params
        self.params.fm_run_params.plot_folder = os.path.join(self.params.fm_run_params.plot_folder, 'aggregator')
        os.makedirs(self.params.fm_run_params.plot_folder, exist_ok=True)
        self._initialised = False
        self._memory_logger_task: asyncio.Task | None = None

    async def init(self):
        if self._initialised:
            return
        self._initialised = True
        self._memory_logger_task = asyncio.create_task(resource_logger(task='aggregator', cadence=timedelta(seconds=5)))

        shape = (self.params.fm_run_params.image_params.num_l, self.params.fm_run_params.image_params.num_m)
        if self.params.fm_run_params.full_stokes:
            shape += (2, 2)

        self._image = np.zeros(
            shape,
            dtype=np.float64)
        self._psf = np.zeros(
            shape,
            dtype=np.float64)

        total_memory = self._image.nbytes + self._psf.nbytes
        logger.info(f"Aggregator using {total_memory / 1024 ** 3:.2f} GB of memory")

        self._fit_beam_jit = jax.jit(fit_beam)

        self._freq_idxs = []
        for sol_int_freq_idx in self.params.sol_int_freq_idxs:
            freq_idxs = np.arange(
                self.params.fm_run_params.chunk_params.num_freqs_per_sol_int
            ) + sol_int_freq_idx * self.params.fm_run_params.chunk_params.num_freqs_per_sol_int
            self._freq_idxs.extend(freq_idxs.tolist())

    def health_check(self):
        """
        Announce health check.
        """
        logger.info(f"Healthy {self.__class__.__name__}")
        return

    def save_image_to_fits(self):
        image_path = os.path.join(self.params.fm_run_params.plot_folder,
                                  f"{self.params.fm_run_params.run_name}_{self.params.image_suffix}_image.fits")
        psf_path = os.path.join(self.params.fm_run_params.plot_folder,
                                f"{self.params.fm_run_params.run_name}_{self.params.image_suffix}_psf.fits")

        # TODO: remove primary beam

        # Easier than averaging taking into account the weights and flags (which are already done per-gridding)
        normalisation = np.max(self._psf, axis=(0, 1))
        image = self._image / normalisation
        psf = self._psf / normalisation

        if self.params.fm_run_params.full_stokes:
            coherencies = ('I', 'Q', 'U', 'V')
            image = au.Quantity(
                np.asarray(broadcast_translate_corrs(
                    jnp.asarray(image),
                    (('XX', 'XY'), ('YX', 'YY')), coherencies
                )),
                'Jy'
            )  # [num_l, num_m, 4]
            psf = au.Quantity(
                np.asarray(broadcast_translate_corrs(
                    jnp.asarray(psf),
                    (('XX', 'XY'), ('YX', 'YY')), coherencies
                )), 'Jy'
            )  # [num_l, num_m, 4]
        else:
            coherencies = ('I',)
            image = au.Quantity(image[..., None], 'Jy')  # [num_l, num_m, 1]
            psf = au.Quantity(psf[..., None], 'Jy')  # [num_l, num_m, 1]

        bandwidth = self.params.fm_run_params.ms_meta.channel_width * len(self._freq_idxs)

        central_freq = np.mean(self.params.fm_run_params.ms_meta.freqs[self._freq_idxs])

        # Fit beam on Stokes I.

        logger.info(
            f"Fitting beam to psf stokes I {psf.shape} "
            f"dl={self.params.fm_run_params.image_params.dl} dm={self.params.fm_run_params.image_params.dm}"
        )

        major, minor, posang = self._fit_beam_jit(
            psf=psf[:, :, 0],
            dl=quantity_to_jnp(self.params.fm_run_params.image_params.dl, 'rad'),
            dm=quantity_to_jnp(self.params.fm_run_params.image_params.dm, 'rad')
        )
        rad2arcsec = 3600 * 180 / np.pi
        logger.info(
            f"Beam major: {major * rad2arcsec:.2f}arcsec, "
            f"minor: {minor * rad2arcsec:.2f}arcsec, "
            f"posang: {posang * 180 * np.pi:.2f}deg"
        )

        image_model = ImageModel(
            phase_center=self.params.fm_run_params.ms_meta.phase_center,
            obs_time=self.params.fm_run_params.ms_meta.ref_time,
            dl=self.params.fm_run_params.image_params.dl,
            dm=self.params.fm_run_params.image_params.dm,
            freqs=central_freq[None],
            bandwidth=bandwidth,
            coherencies=coherencies,
            beam_major=np.asarray(major) * au.rad,
            beam_minor=np.asarray(minor) * au.rad,
            beam_pa=np.asarray(posang) * au.rad,
            unit='JY/PIXEL',
            object_name=self.params.fm_run_params.run_name,
            image=image[:, :, None, :]  # [num_l, num_m, 1, 4/1]
        )
        save_image_to_fits(image_path, image_model=image_model, overwrite=True)
        image_model.image = psf[:, :, None, :]  # [num_l, num_m, 1, 4/1]
        save_image_to_fits(psf_path, image_model=image_model, overwrite=True)
        return AggregatorResponse(
            image_path=image_path,
            psf_path=psf_path
        )

    async def call(self, key, sol_int_time_idx: int, save_to_disk: bool) -> AggregatorResponse:
        logger.info(f"Aggregating {sol_int_time_idx}")
        await self.init()

        keys = jax.random.split(key, len(self.params.sol_int_freq_idxs))

        # Submit them and get one at a time, to avoid memory issues.
        for sol_int_freq_idx, key in zip(self.params.sol_int_freq_idxs, keys):
            response = await self.params.gridder(key, sol_int_time_idx, sol_int_freq_idx)
            self._image += response.image
            self._psf += response.psf

        if save_to_disk:
            logger.info(f"Saving image to disk")
            return self.save_image_to_fits()

        return AggregatorResponse(
            image_path=None,
            psf_path=None
        )
