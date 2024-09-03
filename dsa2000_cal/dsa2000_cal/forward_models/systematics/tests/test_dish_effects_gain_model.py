import numpy as np
from astropy import units as au, time as at, coordinates as ac
from jax import numpy as jnp

from dsa2000_cal.forward_models.systematics.dish_effects_gain_model import dish_effects_gain_model_factory
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsParams, DishEffectsSimulation
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model


def _test_dish_effects_simulation():
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


def _test_dish_effects_gain_model_real_data(tmp_path):
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
