import jax.random
import numpy as np
import pytest
from astropy import coordinates as ac, units as au, time as at
from matplotlib import pyplot as plt

from dsa2000_common.common.plot_utils import figs_to_gif
from dsa2000_common.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model
from dsa2000_fm.systematics.dish_aperture_effects import build_dish_aperture_effects, DishApertureEffects


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
        dish_diameter=5 * au.m,
        focal_length=2 * au.m,
        elevation_pointing_error_stddev=3 * au.arcmin,
        cross_elevation_pointing_error_stddev=3 * au.arcmin,
        # axial_focus_error_stddev=3 * au.mm,
        # elevation_feed_offset_stddev=3 * au.mm,
        # cross_elevation_feed_offset_stddev=3 * au.mm,
        # horizon_peak_astigmatism_stddev=5 * au.mm,
        # surface_error_mean=0 * au.mm,
        # surface_error_stddev=1 * au.mm
    )

    @jax.jit
    def f(d: DishApertureEffects) -> DishApertureEffects:
        return d

    jax.block_until_ready(f(d))

    beam_gain_model.plot_regridded_beam()
    # beam_gain_model.to_aperture().plot_regridded_beam()
    # beam_gain_model.to_aperture().to_image().plot_regridded_beam()

    beam_with_effects = jax.block_until_ready(
        jax.jit(d.apply_dish_aperture_effects)(jax.random.PRNGKey(0), beam_gain_model, geodesic_model))
    beam_with_effects.plot_regridded_beam()



@pytest.mark.parametrize('array_name', ['dsa2000_31b'])
def test_different_magnitude_pointing(array_name: str):
    beam_gain_model = build_beam_gain_model(array_name=array_name, full_stokes=True)
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

    def fig_gen():

        for pointing_offset in np.linspace(0, 10, 100) * au.arcmin:
            d = build_dish_aperture_effects(
                dish_diameter=5 * au.m,
                focal_length=2 * au.m,
                elevation_pointing_error_stddev=pointing_offset,
                cross_elevation_pointing_error_stddev=pointing_offset,
                # axial_focus_error_stddev=3 * au.mm,
                # elevation_feed_offset_stddev=3 * au.mm,
                # cross_elevation_feed_offset_stddev=3 * au.mm,
                # horizon_peak_astigmatism_stddev=5 * au.mm,
                # surface_error_mean=0 * au.mm,
                # surface_error_stddev=1 * au.mm
            )

            beam_with_effects = jax.block_until_ready(
                jax.jit(d.apply_dish_aperture_effects)(jax.random.PRNGKey(0), beam_gain_model, geodesic_model))
            fig = beam_with_effects.plot_regridded_beam(show=False)
            yield fig
            plt.close(fig)

    figs_to_gif(fig_gen(), 'pointing_offset.gif', duration=5, loop=0, dpi=100)




@pytest.mark.parametrize('array_name', ['dsa2000_31b'])
def test_different_magnitude_feed_offset(array_name: str):
    beam_gain_model = build_beam_gain_model(array_name=array_name, full_stokes=True)
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

    def fig_gen():

        for feed_offset_error in np.linspace(0, 10, 100) * au.mm:
            d = build_dish_aperture_effects(
                dish_diameter=5 * au.m,
                focal_length=2 * au.m,
                # elevation_pointing_error_stddev=pointing_offset,
                # cross_elevation_pointing_error_stddev=pointing_offset,
                axial_focus_error_stddev=feed_offset_error,
                elevation_feed_offset_stddev=feed_offset_error,
                cross_elevation_feed_offset_stddev=feed_offset_error,
                # horizon_peak_astigmatism_stddev=5 * au.mm,
                # surface_error_mean=0 * au.mm,
                # surface_error_stddev=1 * au.mm
            )

            beam_with_effects = jax.block_until_ready(
                jax.jit(d.apply_dish_aperture_effects)(jax.random.PRNGKey(0), beam_gain_model, geodesic_model))
            fig = beam_with_effects.plot_regridded_beam(show=False)
            yield fig
            plt.close(fig)

    figs_to_gif(fig_gen(), 'feed_offset.gif', duration=5, loop=0, dpi=100)




@pytest.mark.parametrize('array_name', ['dsa2000_31b'])
def test_different_magnitude_horizon_peak_astigmatism_stddev(array_name: str):
    beam_gain_model = build_beam_gain_model(array_name=array_name, full_stokes=True)
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

    def fig_gen():

        for horizon_peak_astigmatism_stddev in np.linspace(0, 10, 100) * au.mm:
            d = build_dish_aperture_effects(
                dish_diameter=5 * au.m,
                focal_length=2 * au.m,
                # elevation_pointing_error_stddev=pointing_offset,
                # cross_elevation_pointing_error_stddev=pointing_offset,
                # axial_focus_error_stddev=feed_offset_error,
                # elevation_feed_offset_stddev=feed_offset_error,
                # cross_elevation_feed_offset_stddev=feed_offset_error,
                horizon_peak_astigmatism_stddev=horizon_peak_astigmatism_stddev,
                # surface_error_mean=0 * au.mm,
                # surface_error_stddev=1 * au.mm
            )

            beam_with_effects = jax.block_until_ready(
                jax.jit(d.apply_dish_aperture_effects)(jax.random.PRNGKey(0), beam_gain_model, geodesic_model))
            fig = beam_with_effects.plot_regridded_beam(show=False)
            yield fig
            plt.close(fig)

    figs_to_gif(fig_gen(), 'horizon_peak_astigmatism.gif', duration=5, loop=0, dpi=100)





@pytest.mark.parametrize('array_name', ['dsa2000_31b'])
def test_different_magnitude_surface_rms_stddev(array_name: str):
    beam_gain_model = build_beam_gain_model(array_name=array_name, full_stokes=True)
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

    def fig_gen():

        for surface_error_stddev in np.linspace(0, 10, 100) * au.mm:
            d = build_dish_aperture_effects(
                dish_diameter=5 * au.m,
                focal_length=2 * au.m,
                # elevation_pointing_error_stddev=pointing_offset,
                # cross_elevation_pointing_error_stddev=pointing_offset,
                # axial_focus_error_stddev=feed_offset_error,
                # elevation_feed_offset_stddev=feed_offset_error,
                # cross_elevation_feed_offset_stddev=feed_offset_error,
                # horizon_peak_astigmatism_stddev=horizon_peak_astigmatism_stddev,
                surface_error_mean=0 * au.mm,
                surface_error_stddev=surface_error_stddev
            )

            beam_with_effects = jax.block_until_ready(
                jax.jit(d.apply_dish_aperture_effects)(jax.random.PRNGKey(0), beam_gain_model, geodesic_model))
            fig = beam_with_effects.plot_regridded_beam(show=False)
            yield fig
            plt.close(fig)

    figs_to_gif(fig_gen(), 'surface_error.gif', duration=5, loop=0, dpi=100)