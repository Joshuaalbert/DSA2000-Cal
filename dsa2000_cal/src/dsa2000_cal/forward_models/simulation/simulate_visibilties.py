import dataclasses
import os
import time as time_mod
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dsa2000_cal.common.corr_translation import flatten_coherencies
from dsa2000_cal.common.noise import calc_baseline_noise
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.measurement_sets import MeasurementSet, VisibilityData
from dsa2000_cal.visibility_model.rime_model import RIMEModel


@dataclasses.dataclass(eq=False)
class SimulateVisibilities:
    # source models to simulate. Each source gets a gain direction in the flux weighted direction.
    rime_model: RIMEModel

    plot_folder: str

    add_noise: bool = True
    num_shards: int = 1  # must divide channels
    verbose: bool = False
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)

    @property
    def key(self):
        return jax.random.PRNGKey(self.seed)

    def simulate(self, ms: MeasurementSet):
        """
        Simulate visibilities using the system gain model, and store the results in the MS.

        Args:
            ms: the measurement set to store the results
        """

        from jax.experimental import mesh_utils
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec
        from jax.sharding import NamedSharding

        P = PartitionSpec

        if len(jax.devices()) < self.num_shards:
            raise ValueError(
                f"Number of devices {len(jax.devices())} is less than the number of shards {self.num_shards}"
            )

        devices = mesh_utils.create_device_mesh((self.num_shards,),
                                                devices=jax.devices()[:self.num_shards])
        mesh = Mesh(devices, axis_names=('chan',))

        def tree_device_put(tree, sharding):
            return jax.tree.map(lambda x: jax.device_put(x, sharding), tree)

        # Metrics
        vis_sum = 0.
        t0 = time_mod.time()
        fig, axs = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
        gen = ms.create_block_generator(vis=False, weights=False, flags=False)
        gen_response = None
        key = self.key
        while True:
            try:
                times, visibility_coords, _ = gen.send(gen_response)
            except StopIteration:
                break

            axs[0][0].scatter(visibility_coords.uvw[:, 0], visibility_coords.uvw[:, 1], s=1, alpha=0.1)

            # Add time dim

            times_jax = jnp.asarray((times.tt - ms.ref_time).sec)

            visibility_coords = jax.tree.map(jnp.asarray, visibility_coords)

            key, sim_key = jax.random.split(key, 2)

            data_dict = dict(
                key=tree_device_put(sim_key, NamedSharding(mesh, P())),
                times=tree_device_put(times_jax, NamedSharding(mesh, P())),
                channel_width_Hz=tree_device_put(quantity_to_jnp(ms.meta.channel_width), NamedSharding(mesh, P())),
                integration_time_s=tree_device_put(quantity_to_jnp(ms.meta.integration_time), NamedSharding(mesh, P())),
                system_equivalent_flux_density_Jy=tree_device_put(
                    quantity_to_jnp(ms.meta.system_equivalent_flux_density, 'Jy'),
                    NamedSharding(mesh, P())
                ),
                vis_coords=tree_device_put(visibility_coords, NamedSharding(mesh, P()))
            )

            sim_vis_data = self._simulate_jax(
                **data_dict
            )

            # Save the results by pushing the response back to the generator
            gen_response = jax.tree.map(np.asarray, sim_vis_data)

            vis_sum += np.sum(sim_vis_data.vis)
        t1 = time_mod.time()
        print(f"Completed simulation in {t1 - t0} seconds, with total visibility sum: {vis_sum}")
        axs[0][0].set_xlabel('u')
        axs[0][0].set_ylabel('v')
        axs[0][0].set_title('UV Coverage')
        fig.savefig(os.path.join(self.plot_folder, 'uv_coverage.png'))
        plt.close(fig)

        # print(f"Plots saved to {self.plot_folder}.")

    @partial(jax.jit, static_argnums=(0,))
    def _simulate_jax(self, key, times: jax.Array,
                      channel_width_Hz: jax.Array,
                      integration_time_s: jax.Array,
                      system_equivalent_flux_density_Jy: jax.Array,
                      vis_coords: VisibilityCoords) -> VisibilityData:
        """
        Simulate visibilities for a set of source models, creating one row of visibilities per data model.

        Args:
            key: the random key
            times: the times to simulate visibilities for, in TT since start of obs.
            channel_width_Hz: the channel width in Hz
            integration_time_s: the integration time in seconds
            system_equivalent_flux_density_Jy: the system equivalent flux density in Jy
            vis_coords: the visibility coordinates

        Returns:
            vis: [num_row, num_chan, 4/1] the simulated visibilities
        """
        model_data = self.rime_model.get_model_data(times=times)
        vis = self.rime_model.predict_visibilities(
            model_data=model_data,
            visibility_coords=vis_coords
        )  # [num_cal, num_row, num_chan[,2,2]]
        vis = jnp.sum(vis, axis=0)  # [num_row, num_chan[2,2]]

        # if full_stokes flatten coherencies
        if len(np.shape(vis)) == 4 and np.shape(vis)[-2:] == (2, 2):
            vis = jax.vmap(jax.vmap(flatten_coherencies))(vis)  # [num_row, num_chan, 4]
            num_pol = 2
        else:
            vis = vis[:, :, None]  # [num_row, num_chan, 1]
            num_pol = 1

        noise_scale = calc_baseline_noise(
            system_equivalent_flux_density=system_equivalent_flux_density_Jy,
            chan_width_hz=channel_width_Hz,
            t_int_s=integration_time_s,
        )

        # Simulate measurement noise
        key1, key2 = jax.random.split(key)
        # Divide by sqrt(2) to account for 2 polarizations
        noise = (noise_scale / np.sqrt(num_pol)) * (
                jax.random.normal(key1, vis.shape) + 1j * jax.random.normal(key2, vis.shape)
        )

        # jax.debug.print("noise_scale={noise_scale}", noise_scale=noise_scale)
        #
        # jax.debug.print("vis={vis}", vis=vis)
        if self.add_noise:
            vis += noise

        weights = jnp.full(vis.shape, 1. / noise_scale ** 2)
        return VisibilityData(vis=vis, weights=weights)
