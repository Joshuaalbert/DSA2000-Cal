import numpy as np
from astropy import coordinates as ac, units as au, time as at

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.astropy_utils import create_spherical_grid
from dsa2000_cal.forward_model.forward_model import ForwardModel
from dsa2000_cal.gain_models.dish_effects_gain_model import DishEffectsGainModelParams
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0, MeasurementSet
from dsa2000_cal.source_models.discrete_sky_model import DiscreteSkyModel


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
        integration_time=au.Quantity(1, 's'),
        coherencies='linear',
        pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
        times=at.Time("2021-01-01T00:00:00", scale='utc') + np.arange(1) * au.s,
        freqs=au.Quantity([700, 1400, 2000], unit=au.MHz),
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=True
    )
    ms = MeasurementSet.create_measurement_set(ms_folder='forward_model_ms', meta=meta)

    sources = create_spherical_grid(
        pointing=ac.ICRS(0 * au.deg, 0 * au.deg),
        angular_width=au.Quantity(1, au.deg),
        dr=au.Quantity(0.1, au.deg)
    )
    freqs = au.Quantity([1, 2, 3], unit=au.Hz)
    brightness = np.ones((len(sources), len(freqs), 4)) * au.Jy
    discrete_sky_model = DiscreteSkyModel(coords_icrs=sources, freqs=freqs, brightness=brightness)

    forward_model = ForwardModel(
        ms=ms,
        discrete_sky_model=discrete_sky_model,
        dish_effect_params=DishEffectsGainModelParams(),
        ionosphere_specification='light_dawn',
        plot_folder='forward_model_plots',
        cache_folder='forward_model_cache',
        ionosphere_seed=0,
        dish_effects_seed=1,
        seed=2
    )
    forward_model.forward()
