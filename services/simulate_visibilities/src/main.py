import os

import numpy as np
from astropy import coordinates as ac, units as au, time as at
from jax import config

config.update("jax_enable_x64", True)
# Set num jax devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.astropy_utils import create_spherical_grid
from dsa2000_cal.forward_model.forward_model import ForwardModel
from dsa2000_cal.gain_models.dish_effects_gain_model import DishEffectsGainModelParams
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet, MeasurementSetMetaV0
from dsa2000_cal.source_models.discrete_sky_model import DiscreteSkyModel


def main(ms_folder: str):
    if os.path.exists(ms_folder):
        ms = MeasurementSet(ms_folder=ms_folder)
        print(f"Loaded {ms}")
    else:

        fill_registries()
        array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
        array_location = array.get_array_location()
        antennas = array.get_antennas()

        meta = MeasurementSetMetaV0(
            array_name='dsa2000W',
            array_location=array_location,
            phase_tracking=ac.ICRS(0 * au.deg, 0 * au.deg),
            channel_width=array.get_channel_width(),
            integration_time=au.Quantity(1.5, 's'),
            coherencies=['XX', 'XY', 'YX', 'YY'],
            pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
            times=at.Time("2021-01-01T00:00:00", scale='utc') + np.arange(1.5) * au.s,
            freqs=array.get_channel_width() * np.arange(32) + 700e6*au.Hz,
            antennas=antennas,
            antenna_names=array.get_antenna_names(),
            antenna_diameters=array.get_antenna_diameter(),
            with_autocorr=True,
            mount_types='ALT-AZ',
            system_equivalent_flux_density=array.get_system_equivalent_flux_density()
        )
        ms = MeasurementSet.create_measurement_set(ms_folder=ms_folder, meta=meta)
        print(f"Created {ms}")

    sources = create_spherical_grid(
        pointing=ac.ICRS(0 * au.deg, 0 * au.deg),
        angular_radius=au.Quantity(1, au.deg),
        dr=au.Quantity(0.5, au.deg)
    )
    brightness_I = np.ones((len(sources), len(ms.meta.freqs))) * au.Jy
    discrete_sky_model = DiscreteSkyModel(coords_icrs=sources, freqs=ms.meta.freqs, brightness=brightness_I)

    dish_effect_params = DishEffectsGainModelParams(
        dish_diameter=ms.meta.antenna_diameters[0] # Use the first one.
    )

    forward_model = ForwardModel(
        ms=ms,
        discrete_sky_model=discrete_sky_model,
        dish_effect_params=dish_effect_params,
        ionosphere_specification='light_dawn',
        plot_folder='forward_model_plots',
        cache_folder='forward_model_cache',
        ionosphere_seed=0,
        dish_effects_seed=1,
        seed=2
    )
    forward_model.forward()


if __name__ == '__main__':
    ms_folder = os.environ.get('MEASUREMENT_SET')
    main(ms_folder)
