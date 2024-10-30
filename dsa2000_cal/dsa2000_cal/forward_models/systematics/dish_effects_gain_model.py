import dataclasses
import os.path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as au, coordinates as ac
from jax._src.typing import SupportsDType

from dsa2000_cal.common.coord_utils import lmn_to_icrs
from dsa2000_cal.common.mixed_precision_utils import complex_type
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsParams, DishEffectsSimulation
from dsa2000_cal.gain_models.base_spherical_interpolator import phi_theta_from_lmn
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.geodesic_model import GeodesicModel


@dataclasses.dataclass(eq=False)
class DishEffectsGainModel(SphericalInterpolatorGainModel):
    ...


def dish_effects_gain_model_factory(pointings: ac.ICRS | None,
                                    beam_gain_model: GainModel,
                                    dish_effect_params: DishEffectsParams,
                                    plot_folder: str, cache_folder: str, seed: int = 42,
                                    convention: Literal['physical', 'engineering'] = 'physical',
                                    dtype: SupportsDType = complex_type):
    os.makedirs(plot_folder, exist_ok=True)

    dish_effects_simulation = DishEffectsSimulation(
        pointings=pointings,
        beam_gain_model=beam_gain_model,
        dish_effect_params=dish_effect_params,
        plot_folder=plot_folder,
        cache_folder=cache_folder,
        seed=seed,
        convention=convention,
        dtype=dtype
    )

    simulation_results = dish_effects_simulation.simulate_dish_effects()

    Nm, Nl, _ = np.shape(simulation_results.model_lmn)

    model_gains = au.Quantity(np.reshape(
        simulation_results.model_gains,
        (len(simulation_results.model_times), Nm * Nl, len(simulation_results.antennas),
         len(simulation_results.model_freqs), 2, 2)
    ))  # [num_time, Nm*Nl, num_ant, num_model_freq, 2, 2]

    model_phi, model_theta = phi_theta_from_lmn(
        simulation_results.model_lmn[..., 0].flatten(),
        simulation_results.model_lmn[..., 1].flatten(),
        simulation_results.model_lmn[..., 2].flatten()
    )  # [Nm*Nl, 3]

    model_phi = model_phi * au.rad
    model_theta = model_theta * au.rad

    dish_effects_gain_model = DishEffectsGainModel(
        antennas=simulation_results.antennas,
        model_freqs=simulation_results.model_freqs,
        model_times=simulation_results.model_times,
        model_phi=model_phi,
        model_theta=model_theta,
        model_gains=model_gains,
        tile_antennas=False
    )

    geodesic_model = build_geodesic_model()

    # Plot the image plane effects
    for elevation in [45, 90] * au.deg:
        phase_tracking = ac.AltAz(alt=elevation, az=0 * au.deg,
                                  location=simulation_results.antennas[0],
                                  obstime=dish_effects_simulation.ref_time).transform_to(ac.ICRS())
        sources = lmn_to_icrs(dish_effects_simulation.model_lmn, phase_tracking=phase_tracking)

        gain = dish_effects_gain_model.compute_gain(freqs=dish_effects_simulation.model_freqs[:1], sources=sources,
                                                    array_location=simulation_results.antennas[0],
                                                    time=dish_effects_simulation.ref_time,
                                                    pointing=phase_tracking,
                                                    mode='fft')  # [Nm, Nl, num_ant, num_model_freq, 2, 2]
        gain = gain[:, :, 0, 0, 0, 0]  # [Nm, Nl]
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), squeeze=False, sharex=True, sharey=True)
        im = axs[0, 0].imshow(
            np.abs(gain),  # rows are M, columns are L
            origin='lower',
            extent=(dish_effects_simulation.lvec.min().value, dish_effects_simulation.lvec.max().value,
                    dish_effects_simulation.mvec.min().value, dish_effects_simulation.mvec.max().value),
            cmap='PuOr'
        )
        fig.colorbar(im, ax=axs[0, 0])
        axs[0, 0].set_xlabel('l')
        axs[0, 0].set_ylabel('m')
        axs[0, 0].set_title(f'Beam gain amplitude {elevation} elevation')
        im = axs[1, 0].imshow(
            np.angle(gain) * 180 / np.pi,  # rows are M, columns are L
            origin='lower',
            extent=(dish_effects_simulation.lvec.min().value, dish_effects_simulation.lvec.max().value,
                    dish_effects_simulation.mvec.min().value, dish_effects_simulation.mvec.max().value),
            cmap='coolwarm',
            # vmin=-np.pi,
            # vmax=np.pi
        )
        fig.colorbar(im, ax=axs[1, 0], label='degrees')
        axs[1, 0].set_xlabel('l')
        axs[1, 0].set_ylabel('m')
        axs[1, 0].set_title(f'Beam gain phase {elevation} elevation')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"beam_gain_{elevation.value}.png"))
        plt.close(fig)

    return dish_effects_gain_model
