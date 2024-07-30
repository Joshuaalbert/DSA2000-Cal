import os

import numpy as np
from astropy import coordinates as ac, units as au, time as at
from jax import config

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.forward_models.dsa2000_forward_model import DSA2000ForwardModel
from dsa2000_cal.forward_models.synthetic_sky_model.synthetic_sky_model_producer import SyntheticSkyModelProducer
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsParams
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0, MeasurementSet

# Set num jax devices
config.update("jax_enable_x64", True)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


def test_forward_model():
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W_small'))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    meta = MeasurementSetMetaV0(
        array_name='dsa2000W_small',
        array_location=array_location,
        phase_tracking=ac.ICRS(0 * au.deg, 0 * au.deg),
        channel_width=array.get_channel_width(),
        integration_time=au.Quantity(1.5, 's'),
        coherencies=['XX', 'XY', 'YX', 'YY'],
        pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
        times=at.Time("2021-01-01T00:00:00", scale='utc') + 1.5 * np.arange(1) * au.s,
        freqs=au.Quantity([700, 1400, 2000], unit=au.MHz),
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=True,
        mount_types='ALT-AZ'
    )
    ms = MeasurementSet.create_measurement_set(ms_folder='forward_model_ms', meta=meta)

    sky_model_producer = SyntheticSkyModelProducer(
        phase_tracking=ms.meta.pointing,
        freqs=ms.meta.freqs,
        field_of_view=4 * au.deg
    )
    sky_model = sky_model_producer.create_sky_model()

    wsclean_source_models = sky_model.to_wsclean_source_models()

    forward_model = DSA2000ForwardModel(
        dish_effect_params=DishEffectsParams(),
        ionosphere_specification='light_dawn',
        plot_folder='forward_model_plots',
        cache_folder='forward_model_cache',

    )
    forward_model.forward(ms=ms)
