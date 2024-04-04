import dataclasses
import os
from datetime import timedelta
from functools import partial

import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.common.noise import calc_baseline_noise
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.gain_models.dish_effects_gain_model import DishEffectsGainModel, DishEffectsGainModelParams
from dsa2000_cal.gain_models.ionosphere_gain_model import ionosphere_gain_model_factory
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords, MeasurementSet, VisibilityData
from dsa2000_cal.predict.dft_predict import DFTPredict, DFTModelData
from dsa2000_cal.source_models.discrete_sky_model import DiscreteSkyModel


@dataclasses.dataclass(eq=False)
class ForwardModel:
    ms: MeasurementSet
    discrete_sky_model: DiscreteSkyModel

    # Dish effect model parameters
    dish_effect_params: DishEffectsGainModelParams

    # Ionosphere model parameters
    ionosphere_specification: SPECIFICATION

    plot_folder: str
    cache_folder: str
    ionosphere_seed: int
    dish_effects_seed: int
    seed: int

    def __post_init__(self):

        self.beam_gain_model = beam_gain_model_factory(self.ms.meta.array_name)
        if len(self.ms.meta.times) > 1:
            temporal_resolution = timedelta(
                seconds=(self.ms.meta.times[1] - self.ms.meta.times[0]).to(au.s).value
            )

            observation_duration = timedelta(
                seconds=(self.ms.meta.times[-1] - self.ms.meta.times[0]).to(au.s).value
            )
        else:
            temporal_resolution = timedelta()
            observation_duration = timedelta()
        self.ionosphere_gain_model = ionosphere_gain_model_factory(
            phase_tracking=self.ms.meta.phase_tracking,
            field_of_view=self.discrete_sky_model.get_angular_diameter() + 32 * au.arcmin,
            angular_separation=32 * au.arcmin,
            spatial_separation=1 * au.km,
            observation_start_time=self.ms.meta.times[0],
            observation_duration=observation_duration,
            temporal_resolution=temporal_resolution,
            specification=self.ionosphere_specification,
            array_name=self.ms.meta.array_name,
            plot_folder=os.path.join(self.plot_folder, 'plot_ionosphere'),
            cache_folder=os.path.join(self.cache_folder, 'cache_ionosphere'),
            seed=self.ionosphere_seed

        )
        self.dish_effect_gain_model = DishEffectsGainModel(
            beam_gain_model=self.beam_gain_model,
            model_times=self.ms.meta.times,
            dish_effect_params=self.dish_effect_params,
            seed=self.dish_effects_seed
        )

        self.dft_predict = DFTPredict()
        self.key = jax.random.PRNGKey(self.seed)

    def forward(self):
        # Build full gain model
        system_gain_model = self.ionosphere_gain_model @ self.dish_effect_gain_model
        brightness = self.discrete_sky_model.get_source_model_linear(
            freqs=quantity_to_jnp(self.ms.meta.freqs)
        )  # [num_sources, num_freqs, 2, 2]

        gen = self.ms.create_block_generator()
        gen_response = None
        while True:
            try:
                time, visibility_coords, _ = gen.send(gen_response)
            except StopIteration:
                break

            # Compute the lmn coordinates of the sources
            lmn = self.discrete_sky_model.compute_lmn(
                phase_tracking=self.ms.meta.phase_tracking,
                array_location=self.ms.meta.array_location,
                time=time
            )  # [num_sources, 3]

            # Get gains
            gains = system_gain_model.compute_gain(
                freqs=self.ms.meta.freqs,
                sources=self.discrete_sky_model.coords_icrs,
                phase_tracking=self.ms.meta.phase_tracking,
                array_location=self.ms.meta.array_location,
                time=time,
                mode='dft'
            )  # [num_sources, num_ant, num_freq, 2, 2]

            # Simulate visibilities

            dft_model_data = DFTModelData(
                image=brightness,
                gains=gains[:, None, ...],  # Add time dim [1]
                lmn=lmn
            )

            # Save the results by pushing the response back to the generator
            self.key, key = jax.random.split(self.key, 2)
            sim_vis_data = self._predict_jax(
                key=key,
                freqs=quantity_to_jnp(self.ms.meta.freqs),
                dft_model_data=dft_model_data,
                visibility_coords=visibility_coords
            )
            gen_response = jax.tree_map(np.asarray, sim_vis_data)

    @partial(jax.jit, static_argnums=(0,))
    def _predict_jax(self, key, freqs: jax.Array, dft_model_data: DFTModelData, visibility_coords: VisibilityCoords):
        visibilities = self.dft_predict.predict(
            freqs=freqs,
            dft_model_data=dft_model_data,
            visibility_coords=visibility_coords
        )
        visibilities = jnp.reshape(visibilities, np.shape(visibilities)[:-2] + (4,))

        chan_width_hz = quantity_to_jnp(self.ms.meta.channel_width)
        noise_scale = calc_baseline_noise(
            system_equivalent_flux_density=quantity_to_jnp(
                self.ms.meta.system_equivalent_flux_density(), 'Jy'),
            chan_width_hz=chan_width_hz,
            t_int_s=quantity_to_jnp(self.ms.meta.integration_time, 's'),
        )

        # TODO: Simulation RFI

        # TODO: Simulate faint sky with FFT

        # Simulate measurement noise

        key1, key2 = jax.random.split(key)
        # Divide by sqrt(2) to account for polarizations
        noise = (noise_scale / jnp.sqrt(2.)) * (
                jax.random.normal(key1, visibilities.shape) + 1j * jax.random.normal(key2, visibilities.shape)
        )

        visibilities += noise

        weights = jnp.full(visibilities.shape, 1. / noise_scale ** 2)
        return VisibilityData(vis=visibilities, weights=weights)
