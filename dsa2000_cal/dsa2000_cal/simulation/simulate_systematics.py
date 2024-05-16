import dataclasses
import os

import astropy.units as au
import matplotlib.pyplot as plt
import numpy as np
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.antenna_model.utils import get_dish_model_beam_widths
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.gain_models.dish_effects_gain_model import DishEffectsGainModel, DishEffectsGainModelParams
from dsa2000_cal.gain_models.gain_model import ProductGainModel
from dsa2000_cal.gain_models.ionosphere_gain_model import ionosphere_gain_model_factory
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet


@dataclasses.dataclass(eq=False)
class SimulateSystematics:
    # Dish effect model parameters
    dish_effect_params: DishEffectsGainModelParams

    # Ionosphere model parameters
    ionosphere_specification: SPECIFICATION

    plot_folder: str
    cache_folder: str
    ionosphere_seed: int
    dish_effects_seed: int
    verbose: bool = False

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)

    def simulate(self, ms: MeasurementSet) -> ProductGainModel:
        """
        Simulate systematics such as ionosphere and dish effects.

        Returns:
            system_gain_model: the system gain model
        """

        beam_gain_model = beam_gain_model_factory(ms.meta.array_name)

        fill_registries()
        array = array_registry.get_instance(array_registry.get_match(ms.meta.array_name))
        dish_model = array.get_antenna_beam().get_model()

        _freqs, _half_power_widths = get_dish_model_beam_widths(antenna_model=dish_model, threshold=0.5)

        fig, axs = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
        axs[0][0].plot(_freqs.to('Hz'), _half_power_widths)
        axs[0][0].axvline(ms.meta.freqs[0].to('Hz').value, color='red', linestyle='--', label='Min Frequency')
        axs[0][0].set_xlabel('Frequency [Hz]')
        axs[0][0].set_ylabel('Beam Half Power Width [deg]')
        axs[0][0].set_title('Beam Half Power Width vs Frequency')
        axs[0][0].legend()
        plt.savefig(os.path.join(self.plot_folder, 'beam_half_power_width_vs_frequency.png'))
        plt.close(fig)

        field_of_view = np.interp(np.min(ms.meta.freqs), _freqs, _half_power_widths)
        print(f"Field of view: {field_of_view}")

        if len(ms.meta.times) > 1:
            # temporal_resolution = (self.ms.meta.times[1] - self.ms.meta.times[0]).sec * au.s
            observation_duration = (ms.meta.times[-1] - ms.meta.times[0]).sec * au.s
            # So that there are two time points, making it scalable but not realistic
            temporal_resolution = observation_duration
        else:
            temporal_resolution = 0 * au.s
            observation_duration = 0 * au.s

        # TODO: Improve performance -- Will take too long for realistic sized datasets
        ionosphere_gain_model = ionosphere_gain_model_factory(
            phase_tracking=ms.meta.phase_tracking,
            field_of_view=field_of_view + 32 * au.arcmin,
            angular_separation=48 * au.arcmin,
            spatial_separation=1 * au.km,
            observation_start_time=ms.meta.times[0],
            observation_duration=observation_duration,
            temporal_resolution=temporal_resolution,
            specification=self.ionosphere_specification,
            array_name=ms.meta.array_name,
            plot_folder=os.path.join(self.plot_folder, 'plot_ionosphere'),
            cache_folder=os.path.join(self.cache_folder, 'cache_ionosphere'),
            seed=self.ionosphere_seed

        )
        dish_effect_gain_model = DishEffectsGainModel(
            beam_gain_model=beam_gain_model,
            model_times=ms.meta.times,
            dish_effect_params=self.dish_effect_params,
            seed=self.dish_effects_seed,
            cache_folder=os.path.join(self.cache_folder, 'cache_dish_effects_model'),
            plot_folder=os.path.join(self.plot_folder, 'plot_dish_effects_model'),
        )

        # Order is by right multiplication of systematics encountered by radiation from source to observer
        system_gain_model = dish_effect_gain_model @ ionosphere_gain_model

        return system_gain_model
