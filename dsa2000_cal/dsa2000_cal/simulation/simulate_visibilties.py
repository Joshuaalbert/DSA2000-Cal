import dataclasses
from functools import partial
from typing import List

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.typing import SupportsDType

from dsa2000_cal.common.coord_utils import lmn_to_icrs
from dsa2000_cal.common.noise import calc_baseline_noise
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords, MeasurementSet, VisibilityData
from dsa2000_cal.predict.fft_stokes_I_predict import FFTStokesIPredict, FFTStokesIModelData
from dsa2000_cal.predict.gaussian_predict import GaussianPredict, GaussianModelData
from dsa2000_cal.predict.point_predict import PointPredict, PointModelData
from dsa2000_cal.source_models.corr_translation import stokes_to_linear, flatten_coherencies
from dsa2000_cal.source_models.fits_stokes_I_source_model import FitsStokesISourceModel
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel


@dataclasses.dataclass(eq=False)
class SimulateVisibilities:
    # source models to simulate. Each source gets a gain direction in the flux weighted direction.
    wsclean_source_models: List[WSCleanSourceModel]
    fits_source_models: List[FitsStokesISourceModel]

    num_shards: int = 1  # must divide channels
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64
    verbose: bool = False
    seed: int = 42

    def __post_init__(self):
        self.num_sources = len(self.wsclean_source_models) + len(self.fits_source_models)
        self.key = jax.random.PRNGKey(self.seed)

    def _stokes_I_to_linear(self, image_I: jax.Array) -> jax.Array:
        """
        Convert Stokes I to linear.

        Args:
            image_I: [...]

        Returns:
            image_linear: [..., 2, 2]
        """
        shape = np.shape(image_I)
        image_I = lax.reshape(image_I, (np.size(image_I),))
        zero = jnp.zeros_like(image_I)
        image_stokes = jnp.stack([image_I, zero, zero, image_I], axis=-1)  # [..., 4]
        image_linear = jax.vmap(partial(stokes_to_linear, flat_output=False))(image_stokes)  # [..., 2, 2]
        return lax.reshape(image_linear, shape + (2, 2))

    def get_source_directions(self, obs_time: at.Time, phase_tracking: ac.ICRS) -> ac.ICRS:
        """
        Get the source directions in ICRS coordinates.

        Args:
            obs_time: the observation time
            phase_tracking: the phase tracking center

        Returns:
            source_directions: [num_calibrators]
        """
        sources_lmn = au.Quantity(
            np.stack(
                [
                    model.flux_weighted_lmn() for model in self.wsclean_source_models
                ] + [
                    model.flux_weighted_lmn() for model in self.fits_source_models
                ],
                axis=0)
        )  # [num_calibrators, 3]
        return lmn_to_icrs(
            sources_lmn,
            time=obs_time,
            phase_tracking=phase_tracking
        )

    def predict_model_visibilities(self, freqs: jax.Array, apply_gains: jax.Array | None, vis_coords: VisibilityCoords,
                                   flat_coherencies: bool = False) -> jax.Array:
        """
        Simulate visibilities for a set of source models.

        Args:
            freqs: [num_chans]
            apply_gains: [num_cal, num_time, num_ant, num_chan, 2, 2] or None
            vis_coords: [num_row] visibility coordinates
            flat_coherencies: whether to return the visibilities as a flat coherencies

        Returns:
            vis: [num_source_models, num_row, num_chans, 2, 2]
                or  [num_source_models, num_row, num_chans, 4] if flat_output is True
        """
        num_rows, _ = np.shape(vis_coords.uvw)
        num_freqs = np.shape(freqs)[0]
        vis = jnp.zeros((self.num_sources, num_rows, num_freqs, 2, 2),
                        self.dtype)  # [cal_dirs, num_rows, num_chans, 2, 2]
        # Predict the visibilities with pre-applied gains
        dft_predict = PointPredict(convention=self.convention,
                                   dtype=self.dtype)
        gaussian_predict = GaussianPredict(convention=self.convention,
                                           dtype=self.dtype)
        faint_predict = FFTStokesIPredict(convention=self.convention,
                                          dtype=self.dtype)

        # Each calibrator has a source model which is a collection of sources that make up the calibrator.
        cal_idx = 0
        for wsclean_source_model in self.wsclean_source_models:
            preapply_gains_cal = apply_gains[cal_idx]  # [num_time, num_ant, num_chan, 2, 2]

            if wsclean_source_model.point_source_model is not None:
                # Points
                l0 = quantity_to_jnp(wsclean_source_model.point_source_model.l0)  # [source]
                m0 = quantity_to_jnp(wsclean_source_model.point_source_model.m0)  # [source]
                n0 = jnp.sqrt(1. - l0 ** 2 - m0 ** 2)  # [source]

                lmn = jnp.stack([l0, m0, n0], axis=-1)  # [source, 3]
                image_I = quantity_to_jnp(wsclean_source_model.point_source_model.A)  # [source, chan]
                image_linear = self._stokes_I_to_linear(image_I)  # [source, chan, 2, 2]

                dft_model_data = PointModelData(
                    gains=preapply_gains_cal,  # [num_time, num_ant, num_chan, 2, 2]
                    lmn=lmn,
                    image=image_linear
                )
                vis = vis.at[cal_idx].set(
                    dft_predict.predict(
                        freqs=freqs,
                        dft_model_data=dft_model_data,
                        visibility_coords=vis_coords
                    )  # [num_rows, num_chans, 2, 2]
                )

            if wsclean_source_model.gaussian_source_model is not None:
                # Gaussians
                l0 = quantity_to_jnp(wsclean_source_model.gaussian_source_model.l0)  # [source]
                m0 = quantity_to_jnp(wsclean_source_model.gaussian_source_model.m0)  # [source]
                n0 = jnp.sqrt(1. - l0 ** 2 - m0 ** 2)  # [source]

                lmn = jnp.stack([l0, m0, n0], axis=-1)  # [source, 3]
                image_I = quantity_to_jnp(wsclean_source_model.gaussian_source_model.A)  # [source, chan]
                image_linear = self._stokes_I_to_linear(image_I)  # [source, chan, 2, 2]

                ellipse_params = jnp.stack([
                    quantity_to_jnp(wsclean_source_model.gaussian_source_model.major),
                    quantity_to_jnp(wsclean_source_model.gaussian_source_model.minor),
                    quantity_to_jnp(wsclean_source_model.gaussian_source_model.theta)
                ],
                    axis=-1)  # [source, 3]

                gaussian_model_data = GaussianModelData(
                    image=image_linear,  # [source, chan, 2, 2]
                    gains=preapply_gains_cal,  # [num_time, num_ant, num_chan, 2, 2]
                    ellipse_params=ellipse_params,  # [source, 3]
                    lmn=lmn  # [source, 3]
                )

                vis = vis.at[cal_idx].add(
                    gaussian_predict.predict(
                        freqs=freqs,  # [chan]
                        gaussian_model_data=gaussian_model_data,  # [source, chan, 2, 2]
                        visibility_coords=vis_coords  # [row, 3]
                    ),  # [num_rows, num_chans, 2, 2]
                    indices_are_sorted=True,
                    unique_indices=True
                )

            cal_idx += 1

        for fits_source_model in self.fits_source_models:
            preapply_gains_cal = apply_gains[cal_idx]  # [num_time, num_ant, num_chan, 2, 2]
            l0 = quantity_to_jnp(fits_source_model.l0)  # [num_chan]
            m0 = quantity_to_jnp(fits_source_model.m0)  # [num_chan]
            dl = quantity_to_jnp(fits_source_model.dl)  # [num_chan]
            dm = quantity_to_jnp(fits_source_model.dm)  # [num_chan]
            image = jnp.stack(
                [
                    quantity_to_jnp(img)
                    for img in fits_source_model.images
                ],
                axis=0
            )  # [num_chan, Nx, Ny]

            faint_model_data = FFTStokesIModelData(
                image=image,  # [num_chan, Nx, Ny]
                gains=preapply_gains_cal,  # [num_time, num_ant, num_chan, 2, 2]
                l0=l0,  # [num_chan]
                m0=m0,  # [num_chan]
                dl=dl,  # [num_chan]
                dm=dm  # [num_chan]
            )

            vis = vis.at[cal_idx].set(
                faint_predict.predict(
                    freqs=freqs,
                    faint_model_data=faint_model_data,
                    visibility_coords=vis_coords
                )
            )
        if flat_coherencies:
            # Transform
            vis = lax.reshape(vis, (self.num_sources * num_rows * num_freqs, 2, 2))
            vis = jax.vmap(flatten_coherencies)(vis)  # [num_sources*num_rows*num_freqs, 4]
            vis = lax.reshape(vis, (self.num_sources, num_rows, num_freqs, 4))
        return vis

    def simulate(self, ms: MeasurementSet, system_gain_model: GainModel):
        """
        Simulate visibilities using the system gain model, and store the results in the MS.

        Args:
            ms: the measurement set to store the results
            system_gain_model: the system gain model
        """

        from jax.experimental import mesh_utils
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec
        from jax.sharding import NamedSharding

        P = PartitionSpec

        devices = mesh_utils.create_device_mesh((self.num_shards,))
        mesh = Mesh(devices, axis_names=('chan',))

        def tree_device_put(tree, sharding):
            return jax.tree_map(lambda x: jax.device_put(x, sharding), tree)

        source_directions = self.get_source_directions(
            obs_time=ms.ref_time,
            phase_tracking=ms.meta.phase_tracking
        )

        gen = ms.create_block_generator(vis=False, weights=False, flags=False)
        gen_response = None
        while True:
            try:
                time, visibility_coords, _ = gen.send(gen_response)
            except StopIteration:
                break

            # Get gains
            system_gains = system_gain_model.compute_gain(
                freqs=ms.meta.freqs,
                sources=source_directions,
                phase_tracking=ms.meta.phase_tracking,
                array_location=ms.meta.array_location,
                time=time,
                mode='fft'
            )  # [num_sources, num_ant, num_freq, 2, 2]

            visibility_coords = jax.tree_map(jnp.asarray, visibility_coords)

            self.key, key = jax.random.split(self.key, 2)

            data_dict = dict(
                key=key,
                freqs=quantity_to_jnp(ms.meta.freqs),
                channel_width_Hz=quantity_to_jnp(ms.meta.channel_width),
                integration_time_s=quantity_to_jnp(ms.meta.integration_time),
                system_equivalent_flux_density_Jy=quantity_to_jnp(
                    ms.meta.system_equivalent_flux_density, 'Jy'),
                apply_gains=system_gains,
                visibility_coords=visibility_coords
            )

            data_dict = dict(
                key=tree_device_put(data_dict['key'], NamedSharding(mesh, P())),
                freqs=tree_device_put(data_dict['freqs'], NamedSharding(mesh, P('chan'))),
                channel_width_Hz=tree_device_put(data_dict['channel_width_Hz'], NamedSharding(mesh, P())),
                integration_time_s=tree_device_put(data_dict['integration_time_s'], NamedSharding(mesh, P())),
                system_equivalent_flux_density_Jy=tree_device_put(data_dict['system_equivalent_flux_density_Jy'],
                                                                  NamedSharding(mesh, P())),
                apply_gains=tree_device_put(data_dict['apply_gains'],
                                            NamedSharding(mesh, P(None, None, 'chan'))),
                visibility_coords=tree_device_put(data_dict['visibility_coords'], NamedSharding(mesh, P()))
            )

            sim_vis_data = self._simulate_jax(
                **data_dict
            )

            # Save the results by pushing the response back to the generator
            gen_response = jax.tree_map(np.asarray, sim_vis_data)

    @partial(jax.jit, static_argnums=(0,))
    def _simulate_jax(self, key, freqs: jax.Array,
                      channel_width_Hz: jax.Array, integration_time_s: jax.Array,
                      system_equivalent_flux_density_Jy: jax.Array,
                      apply_gains: jax.Array,
                      vis_coords: VisibilityCoords) -> VisibilityData:

        vis = self.predict_model_visibilities(
            freqs=freqs,
            apply_gains=apply_gains,
            vis_coords=vis_coords,
            flat_coherencies=True
        )  # [num_cal, num_row, num_chan, 4]
        vis = jnp.sum(vis, axis=0)  # [num_row, num_chan, 4]

        noise_scale = calc_baseline_noise(
            system_equivalent_flux_density=system_equivalent_flux_density_Jy,
            chan_width_hz=channel_width_Hz,
            t_int_s=integration_time_s,
        )

        # TODO: Simulation RFI

        # Simulate measurement noise
        key1, key2 = jax.random.split(key)
        # Divide by sqrt(2) to account for polarizations
        noise = (noise_scale / jnp.sqrt(2.)) * (
                jax.random.normal(key1, vis.shape) + 1j * jax.random.normal(key2, vis.shape)
        )

        vis += noise

        weights = jnp.full(vis.shape, 1. / noise_scale ** 2)
        return VisibilityData(vis=vis, weights=weights)
