import numpy as np
import pytest
from astropy import units as au, time as at, coordinates as ac
from jax import numpy as jnp

from dsa2000_cal.common.coord_utils import lmn_to_icrs
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.gain_models.dish_effects_gain_model import DishEffectsGainModel, DishEffectsGainModelParams


@pytest.mark.parametrize('mode', ['fft', 'dft'])
def test_dish_effects_gain_model_real_data(tmp_path, mode):
    freqs = au.Quantity([700e6, 2000e6], unit=au.Hz)
    beam_gain_model = beam_gain_model_factory(array_name='dsa2000W')

    dish_effects_gain_model = DishEffectsGainModel(
        antennas=beam_gain_model.antennas,
        beam_gain_model=beam_gain_model,
        model_times=at.Time(['2021-01-01T00:00:00', '2021-01-01T00:01:00'], scale='utc'),
        dish_effect_params=DishEffectsGainModelParams(),
        cache_folder=str(tmp_path / f'dish_effects_gain_model_cache_{mode}'),
        plot_folder=f'dish_effects_gain_model_plots_{mode}'
    )

    np.testing.assert_allclose(dish_effects_gain_model.dy / dish_effects_gain_model.dl, 2.5 * au.m, atol=0.1 * au.m)
    assert jnp.all(jnp.isfinite(dish_effects_gain_model.aperture_gains))
    assert dish_effects_gain_model.dx.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.dy.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.dl.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.dm.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.lmn_data.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.X.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.Y.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.lvec.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.mvec.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.model_wavelengths.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.sampling_interval.unit.is_equivalent(au.m)

    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg, )
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:30', scale='utc')

    if mode == 'dft':
        sources = ac.ICRS(ra=[0, 0.] * au.deg, dec=[0., 1.] * au.deg)
    elif mode == 'fft':
        sources = lmn_to_icrs(dish_effects_gain_model.lmn_data, time=time, phase_tracking=phase_tracking)
    else:
        raise ValueError(f"Unknown mode {mode}")

    aperture_field = dish_effects_gain_model.compute_aperture_field_model(
        freqs=freqs,
        time=time,
        elevation=90. * au.deg
    )  # [2n+1, 2n+1, num_ant, num_freq]
    assert aperture_field.shape == (dish_effects_gain_model.X.shape[0], dish_effects_gain_model.Y.shape[1],
                                    len(dish_effects_gain_model.antennas), len(freqs), 2, 2)
    gains = dish_effects_gain_model.compute_gain(freqs=freqs, sources=sources, array_location=array_location,
                                                 time=time, pointing=phase_tracking, mode=mode)
    if mode == 'fft':
        import pylab as plt
        plt.imshow(np.abs(aperture_field[:, :, 0, 0, 0, 0]),
                   origin='lower')
        plt.colorbar()
        plt.show()
        plt.imshow(np.angle(gains[..., 0, 0, 0, 0]),
                   origin='lower')
        plt.colorbar()
        plt.show()
    else:
        print(gains[..., 0, 0])
    assert gains.shape == sources.shape + (
        len(dish_effects_gain_model.antennas), len(freqs), 2, 2)

    # Test from cache
    dish_effects_gain_model = DishEffectsGainModel(
        antennas=beam_gain_model.antennas,
        beam_gain_model=beam_gain_model,
        model_times=at.Time(['2021-01-01T00:00:00', '2021-01-01T00:01:00'], scale='utc'),
        dish_effect_params=DishEffectsGainModelParams(),
        cache_folder=str(tmp_path / f'dish_effects_gain_model_cache_{mode}'),
        plot_folder=f'dish_effects_gain_model_plots_{mode}'
    )
