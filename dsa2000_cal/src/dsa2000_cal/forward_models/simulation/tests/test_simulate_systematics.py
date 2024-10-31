import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pytest

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.forward_models.simulation.simulate_systematics import SimulateSystematics
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsParams
from dsa2000_cal.measurement_sets.measurement_set import  MeasurementSetMetaV0, MeasurementSet


@pytest.fixture(scope='function')
def mock_calibrator_source_models(tmp_path):
    fill_registries()

    # Load array
    array = array_registry.get_instance(array_registry.get_match('dsa2000W_small'))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    # -00:36:29.015,58.45.50.398
    phase_tracking = ac.SkyCoord("-00h36m29.015s", "58d45m50.398s", frame='icrs')
    phase_tracking = ac.ICRS(phase_tracking.ra, phase_tracking.dec)

    meta = MeasurementSetMetaV0(
        array_name='dsa2000W_small',
        array_location=array_location,
        phase_tracking=phase_tracking,
        channel_width=array.get_channel_width(),
        integration_time=au.Quantity(1.5, 's'),
        coherencies=['XX', 'XY', 'YX', 'YY'],
        pointings=phase_tracking,
        times=at.Time("2021-01-01T00:00:00", scale='utc') + 1.5 * np.arange(2) * au.s,
        freqs=au.Quantity([700, 1400], unit=au.MHz),
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=True,
        mount_types='ALT-AZ'
    )
    ms = MeasurementSet.create_measurement_set(str(tmp_path), meta)

    return ms


def _test_simulate_systematics(mock_calibrator_source_models):
    ms = mock_calibrator_source_models

    simulator = SimulateSystematics(
        dish_effect_params=DishEffectsParams(),
        ionosphere_specification='light_dawn',
        plot_folder='plots',
        cache_folder='cache',
        ionosphere_seed=0,
        dish_effects_seed=1,
        full_stokes=False
    )
    ionosphere_gain_model = simulator.simulate_ionosphere(ms=ms)
    dish_effects_gain_model = simulator.simulate_dish_effects(ms=ms)
    system_gain_model = dish_effects_gain_model @ ionosphere_gain_model
    print(system_gain_model)
