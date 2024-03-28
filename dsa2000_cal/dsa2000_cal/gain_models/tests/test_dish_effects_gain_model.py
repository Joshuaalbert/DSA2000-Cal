import pylab as plt
import pytest
from astropy import units as au, time as at, coordinates as ac
from jax import numpy as jnp

from dsa2000_cal.common.coord_utils import lmn_to_icrs
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factor
from dsa2000_cal.gain_models.dish_effects_gain_model import DishEffectsGainModel


@pytest.mark.parametrize('mode', ['fft', 'dft'])
def test_dish_effects_gain_model_real_data(mode):
    freqs = au.Quantity([700e6, 2000e6], unit=au.Hz)
    beam_gain_model = beam_gain_model_factor(freqs=freqs, array_name='dsa2000W')

    dish_effects_gain_model = DishEffectsGainModel(
        beam_gain_model=beam_gain_model,
        model_times=at.Time(['2021-01-01T00:00:00', '2021-01-01T00:01:00'], scale='utc'),
        elevation_pointing_error_stddev=0.5 * au.deg,
    )

    assert jnp.allclose(dish_effects_gain_model.dy / dish_effects_gain_model.dl, 2.5 * au.m, atol=0.1)
    assert jnp.all(jnp.isfinite(dish_effects_gain_model.aperture_amplitude))
    assert dish_effects_gain_model.dx.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.dy.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.dl.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.dm.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.lmn_data.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.X.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.Y.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.lvec.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.mvec.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.wavelengths.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.sampling_interval.unit.is_equivalent(au.m)

    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg, )
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:30', scale='utc')

    if mode == 'dft':
        sources = ac.ICRS(ra=[0, 0.] * au.deg, dec=[0., 1.] * au.deg)
    elif mode == 'fft':
        sources = lmn_to_icrs(dish_effects_gain_model.lmn_data, array_location=array_location, time=time,
                              phase_tracking=phase_tracking)
    else:
        raise ValueError(f"Unknown mode {mode}")

    aperture_field = dish_effects_gain_model.compute_aperture_field_model(
        time=time,
        elevation=90. * au.deg
    )  # [2n+1, 2n+1, num_ant, num_freq]

    plt.imshow(
        jnp.abs(aperture_field[:, :, 0, 0]),
        origin='lower',
        extent=(dish_effects_gain_model.X.min().value, dish_effects_gain_model.X.max().value,
                dish_effects_gain_model.Y.min().value, dish_effects_gain_model.Y.max().value)
    )
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.colorbar()
    plt.show()

    plt.imshow(
        jnp.angle(aperture_field[:, :, 0, 0]),
        origin='lower',
        extent=(dish_effects_gain_model.X.min().value, dish_effects_gain_model.X.max().value,
                dish_effects_gain_model.Y.min().value, dish_effects_gain_model.Y.max().value)
    )
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.colorbar()
    plt.show()

    gains = dish_effects_gain_model.compute_beam(
        sources=sources,
        phase_tracking=phase_tracking,
        array_location=array_location,
        time=time,
        mode=mode
    )
    if mode == 'fft':
        plt.imshow(
            jnp.abs(gains[:, :, 0, 0, 0, 0]),
            origin='lower',
            extent=(
                dish_effects_gain_model.mvec.min().value,
                dish_effects_gain_model.mvec.max().value,
                dish_effects_gain_model.lvec.min().value,
                dish_effects_gain_model.lvec.max().value)
        )
        plt.colorbar()
        plt.show()
    else:
        print(gains[..., 0:2, :, 0, 0])
    assert gains.shape == sources.shape + (
        dish_effects_gain_model.num_antenna, len(dish_effects_gain_model.freqs), 2, 2)
