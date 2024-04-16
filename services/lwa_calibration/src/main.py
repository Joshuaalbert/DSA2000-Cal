import os

import numpy as np
from astropy import coordinates as ac, units as au, time as at
from jax import config

from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel

config.update("jax_enable_x64", True)
# Set num jax devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry, source_model_registry
from dsa2000_cal.common.astropy_utils import create_spherical_grid
from dsa2000_cal.forward_model.forward_model import ForwardModel
from dsa2000_cal.gain_models.dish_effects_gain_model import DishEffectsGainModelParams
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet, MeasurementSetMetaV0
from dsa2000_cal.source_models.discrete_sky_model import DiscreteSkyModel


def main(ms_folder: str):
    ms = MeasurementSet(ms_folder=ms_folder)

    # Create a sky model for calibration
    fill_registries()
    cas_a_source_file = source_model_registry.get_instance(source_model_registry.get_match('cas_a')).get_wsclean_clean_component_file()
    cas_a_sky_model = WSCleanSourceModel.from_wsclean_model(
        wsclean_clean_component_file=cas_a_source_file,
        freqs=ms.meta.freqs,
        time=ms.meta.times[0],
        phase_tracking=ms.meta.phase_tracking
    )
    cyg_a_source_file = source_model_registry.get_instance(source_model_registry.get_match('cyg_a')).get_wsclean_clean_component_file()
    cyg_a_sky_model = WSCleanSourceModel.from_wsclean_model(
        wsclean_clean_component_file=cyg_a_source_file,
        freqs=ms.meta.freqs,
        time=ms.meta.times[0],
        phase_tracking=ms.meta.phase_tracking
    )
    sky_model = cas_a_sky_model + cyg_a_sky_model

    # Run calibration
    ...

    # Save the subtracted visibilities
    ...


if __name__ == '__main__':
    ms_folder = os.environ.get('MEASUREMENT_SET')
    main(ms_folder)
