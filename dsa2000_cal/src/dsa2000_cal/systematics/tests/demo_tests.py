import jax.random
import numpy as np
import pytest
from astropy import coordinates as ac, units as au, time as at

from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_cal.geodesics.base_geodesic_model import build_geodesic_model
from dsa2000_cal.systematics.dish_aperture_effects import build_dish_aperture_effects


@pytest.mark.parametrize('array_name', ['dsa2000_31b'])
def test_dish_aperture_effects(array_name: str):
    beam_gain_model = build_beam_gain_model(array_name='dsa2000_31b', full_stokes=True)
    antennas = ac.EarthLocation.from_geocentric(
        x=np.random.normal(size=5) * au.km,
        y=np.random.normal(size=5) * au.km,
        z=np.random.normal(size=5) * au.km
    )
    array_location = ac.EarthLocation.from_geocentric(
        x=0 * au.km,
        y=0 * au.km,
        z=0 * au.km
    )
    phase_center = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    obstimes = at.Time(np.linspace(0, 1, 10), format='mjd')
    geodesic_model = build_geodesic_model(
        antennas=antennas,
        array_location=array_location,
        phase_center=phase_center,
        obstimes=obstimes,
        ref_time=obstimes[0],
        pointings=None
    )

    d = build_dish_aperture_effects(
        num_antennas=5,
        dish_diameter=5 * au.m,
        focal_length=2 * au.m,
        elevation_pointing_error_stddev=2 * au.arcmin,
        cross_elevation_pointing_error_stddev=2 * au.arcmin,
        axial_focus_error_stddev=3 * au.mm,
        elevation_feed_offset_stddev=3 * au.mm,
        cross_elevation_feed_offset_stddev=3 * au.mm,
        horizon_peak_astigmatism_stddev=5 * au.mm,
        surface_error_mean=0 * au.mm,
        surface_error_stddev=1 * au.mm
    )

    beam_gain_model.plot_regridded_beam()
    # beam_gain_model.to_aperture().plot_regridded_beam()
    # beam_gain_model.to_aperture().to_image().plot_regridded_beam()

    beam_with_effects = jax.block_until_ready(
        jax.jit(d.apply_dish_aperture_effects)(jax.random.PRNGKey(0), beam_gain_model, geodesic_model))
    beam_with_effects.plot_regridded_beam()
