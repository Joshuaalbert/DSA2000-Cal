import dataclasses
import os

import astropy.units as au
import matplotlib.pyplot as plt
import numpy as np
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.antenna_model.antenna_model_utils import get_dish_model_beam_widths
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.forward_models.systematics.dish_effects_gain_model import dish_effects_gain_model_factory
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsParams
from dsa2000_cal.forward_models.systematics.ionosphere_gain_model import build_ionosphere_gain_model
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet


@dataclasses.dataclass(eq=False)
class SimulateSystematics:
    # Dish effect model parameters
    dish_effect_params: DishEffectsParams | None

    # Ionosphere model parameters
    ionosphere_specification: SPECIFICATION | None

    full_stokes: bool

    plot_folder: str
    cache_folder: str
    ionosphere_seed: int
    dish_effects_seed: int
    verbose: bool = False

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)

    def simulate_ionosphere(self, ms: MeasurementSet) -> GainModel:
        if self.ionosphere_specification is None:
            raise ValueError("Ionosphere specification must be provided to simulate ionosphere.")

        array = array_registry.get_instance(array_registry.get_match(ms.meta.array_name))
        dish_model = array.get_antenna_model()
        _freqs, _half_power_widths = get_dish_model_beam_widths(antenna_model=dish_model, threshold=0.5)
        field_of_view = au.Quantity(np.interp(np.min(ms.meta.freqs), _freqs, _half_power_widths))

        fig, axs = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
        axs[0][0].plot(_freqs.to('Hz'), _half_power_widths)
        axs[0][0].axvline(ms.meta.freqs[0].to('Hz').value, color='red', linestyle='--', label='Min Frequency')
        axs[0][0].set_xlabel('Frequency [Hz]')
        axs[0][0].set_ylabel('Beam Half Power Width [deg]')
        axs[0][0].set_title('Beam Half Power Width vs Frequency')
        axs[0][0].legend()
        plt.savefig(os.path.join(self.plot_folder, 'beam_half_power_width_vs_frequency.png'))
        plt.close(fig)

        if len(ms.meta.times) > 1:
            # temporal_resolution = (self.ms.meta.times[1] - self.ms.meta.times[0]).sec * au.s
            observation_duration = (ms.meta.times[-1] - ms.meta.times[0]).sec * au.s
            # So that there are two time points, making it scalable but not realistic
            temporal_resolution = observation_duration
        else:
            temporal_resolution = 0 * au.s
            observation_duration = 0 * au.s

        gain_model = build_ionosphere_gain_model(
            pointing=ms.meta.phase_tracking,
            field_of_view=field_of_view,
            model_freqs=ms.meta.freqs,
            spatial_resolution=2. * au.km,
            observation_start_time=ms.meta.times[0],
            observation_duration=observation_duration,
            temporal_resolution=temporal_resolution,
            specification=self.ionosphere_specification,
            array_name=ms.meta.array_name,
            plot_folder=os.path.join(self.plot_folder, 'ionosphere'),
            cache_folder=os.path.join(self.cache_folder, 'ionosphere'),
            seed=self.ionosphere_seed
        )

        # screens
        l_screen, m_screen = np.meshgrid(gain_model.lvec_jax, gain_model.mvec_jax, indexing='ij')
        sc = plt.scatter(l_screen.flatten(), m_screen.flatten(), s=1,
                         c=np.log10(np.abs(gain_model.model_gains_jax[0, :, :, 0].flatten())))
        plt.colorbar(sc, label='log10(Amplitude)')
        plt.xlabel('l')
        plt.ylabel('m')
        plt.title('Ionosphere log10(Amplitude)')
        plt.show()

        sc = plt.scatter(l_screen.flatten(), m_screen.flatten(), s=1,
                         c=np.angle(gain_model.model_gains_jax[0, :, :, 0].flatten()), cmap='hsv', vmin=-np.pi,
                         vmax=np.pi)
        plt.colorbar(sc, label='radians')
        plt.xlabel('l')
        plt.ylabel('m')
        plt.title('Ionosphere Phase')
        plt.show()
        return gain_model

    def simulate_dish_effects(self, ms: MeasurementSet) -> GainModel:
        """
        Simulate systematics such as ionosphere and dish effects.

        Returns:
            system_gain_model: the system gain model
            dish_effect_gain_model: the dish effect gain model
        """

        if self.dish_effect_params is None:
            raise ValueError("Dish effect parameters must be provided to simulate dish effects.")

        beam_gain_model = beam_gain_model_factory(ms)

        gain_model = dish_effects_gain_model_factory(
            pointings=ms.meta.pointings[0],
            beam_gain_model=beam_gain_model,
            dish_effect_params=self.dish_effect_params,
            seed=self.dish_effects_seed,
            cache_folder=os.path.join(self.cache_folder, 'dish_effects'),
            plot_folder=os.path.join(self.plot_folder, 'dish_effects'),
        )

        # screens
        l_screen, m_screen = np.meshgrid(gain_model.lvec_jax, gain_model.mvec_jax, indexing='ij')
        sc = plt.scatter(l_screen.flatten(), m_screen.flatten(), s=1,
                         c=np.log10(np.abs(gain_model.model_gains_jax[0, :, :, 0].flatten())))
        plt.colorbar(sc, label='log10(Amplitude)')
        plt.xlabel('l')
        plt.ylabel('m')
        plt.title('Dish Effects log10(Amplitude)')
        plt.show()

        sc = plt.scatter(l_screen.flatten(), m_screen.flatten(), s=1,
                         c=np.angle(gain_model.model_gains_jax[0, :, :, 0].flatten()), cmap='hsv', vmin=-np.pi,
                         vmax=np.pi)
        plt.colorbar(sc, label='radians')
        plt.xlabel('l')
        plt.ylabel('m')
        plt.title('Dish Effects Phase')
        plt.show()
        return gain_model
