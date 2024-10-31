import os

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

from jax import numpy as jnp
from astropy import coordinates as ac, units as au, time as at
import numpy as np

from dsa2000_cal.forward_models.systematics.dish_effects_gain_model import dish_effects_gain_model_factory
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsSimulation, DishEffectsParams
from src.dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_cal.common.astropy_utils import create_spherical_earth_grid, create_spherical_grid_old
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.forward_models.systematics.ionosphere_gain_model import build_ionosphere_gain_model
from dsa2000_cal.forward_models.systematics.ionosphere_simulation import IonosphereSimulation


def test_real_ionosphere_gain_model():
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    field_of_view = 4 * au.deg
    spatial_resolution = 2.0 * au.km
    observation_start_time = at.Time('2021-01-01T00:00:00', scale='utc')
    observation_duration = 0 * au.s
    temporal_resolution = 0 * au.s
    model_freqs = [700e6, 2000e6] * au.Hz
    ionosphere_gain_model = build_ionosphere_gain_model(
        pointing=phase_tracking,
        field_of_view=field_of_view,
        spatial_resolution=spatial_resolution,
        observation_start_time=observation_start_time,
        observation_duration=observation_duration,
        temporal_resolution=temporal_resolution,
        model_freqs=model_freqs,
        specification='light_dawn',
        array_name='dsa2000W',
        plot_folder='plot_ionosphere',
        cache_folder='cache_ionosphere',
        seed=42
    )


def test_ionosphere_simulation():
    array_location = ac.EarthLocation(lat=0 * au.deg, lon=0 * au.deg, height=0 * au.m)

    radius = 10 * au.km
    spatial_separation = 3 * au.km
    model_antennas = create_spherical_earth_grid(
        center=array_location,
        radius=radius,
        dr=spatial_separation
    )

    phase_tracking = ac.ICRS(0 * au.deg, 0 * au.deg)
    model_directions = create_spherical_grid_old(
        pointing=phase_tracking,
        angular_radius=2 * au.deg,
        dr=50. * au.arcmin
    )
    # ac.ICRS(ra=[0, 0.] * au.deg, dec=[0., 1.] * au.deg)
    model_times = at.Time(['2021-01-01T00:00:00', '2021-01-01T00:10:00'], scale='utc')
    model_lmn = icrs_to_lmn(sources=model_directions, phase_tracking=phase_tracking)
    print(model_antennas.shape, model_directions.shape, model_lmn.shape)

    ionosphere_simulation = IonosphereSimulation(
        array_location=array_location,
        pointing=phase_tracking,
        model_lmn=model_lmn,
        model_times=model_times,
        model_antennas=model_antennas,
        specification='light_dawn',
        plot_folder='plot_ionosphere_small_test_2',
        cache_folder='cache_ionosphere_small_test_2',
        # interp_mode='kriging'
    )

    simulation_results = ionosphere_simulation.simulate_ionosphere()

    assert simulation_results.ref_ant == array_location
    assert simulation_results.ref_time == model_times[0]
    assert np.all(np.isfinite(simulation_results.dtec))


def test_dish_effects_simulation():
    pointing = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    beam_gain_model = build_beam_gain_model(array_name='lwa')
    dish_effects_simulation = DishEffectsSimulation(
        pointings=pointing,
        dish_effect_params=DishEffectsParams(),
        beam_gain_model=beam_gain_model,
        plot_folder='dish_effects_simulation_plots',
        cache_folder='dish_effects_simulation_cache'
    )
    simulation_results = dish_effects_simulation.simulate_dish_effects()
    Nm, Nl, _ = np.shape(simulation_results.model_lmn)
    assert np.shape(simulation_results.model_gains) == (
        len(beam_gain_model.model_times), Nm, Nl, len(beam_gain_model.antennas), len(beam_gain_model.model_freqs), 2, 2)
    assert jnp.all(jnp.isfinite(simulation_results.model_gains))

    # np.testing.assert_allclose(dish_effects_simulation.dy / dish_effects_simulation.dl, 2.5 * au.m, atol=0.1 * au.m)
    assert dish_effects_simulation.dx.unit.is_equivalent(au.m)
    assert dish_effects_simulation.dy.unit.is_equivalent(au.m)
    assert dish_effects_simulation.dl.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_simulation.dm.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_simulation.model_lmn.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_simulation.X.unit.is_equivalent(au.m)
    assert dish_effects_simulation.Y.unit.is_equivalent(au.m)
    assert dish_effects_simulation.lvec.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_simulation.mvec.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_simulation.model_freqs.unit.is_equivalent(au.Hz)
    assert dish_effects_simulation.aperture_sampling_interval.unit.is_equivalent(au.m)


def test_dish_effects_gain_model_real_data(tmp_path):
    freqs = au.Quantity([700e6], unit=au.Hz)
    beam_gain_model = build_beam_gain_model(array_name='dsa2000W')
    pointing = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)

    dish_effects_gain_model = dish_effects_gain_model_factory(
        pointings=pointing,
        beam_gain_model=beam_gain_model,
        dish_effect_params=DishEffectsParams(),
        plot_folder='dish_effects_gain_model_plots',
        cache_folder='dish_effects_gain_model_cache'
    )

    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:30', scale='utc')

    sources = ac.ICRS(ra=[0, 0.] * au.deg, dec=[0., 1.] * au.deg)
    dish_effects_gain_model.compute_gain(
        freqs=freqs,
        sources=sources,
        pointing=pointing,
        array_location=array_location,
        time=time
    )
