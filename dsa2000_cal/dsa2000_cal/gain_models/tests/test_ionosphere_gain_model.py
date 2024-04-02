from datetime import timedelta

import jax
import numpy as np
import pylab as plt
from astropy import units as au, coordinates as ac, time as at

from dsa2000_cal.common.astropy_utils import create_spherical_grid, create_spherical_earth_grid
from dsa2000_cal.gain_models.ionosphere_gain_model import IonosphereGainModel, msqrt, ionosphere_gain_model_factory


def test_msqrt():
    M = jax.random.normal(jax.random.PRNGKey(42), (100, 100))
    A = M @ M.T
    max_eig, min_eig, L = msqrt(A)
    np.testing.assert_allclose(A, L @ L.T, atol=2e-4)


def test_real_ionosphere_gain_model():
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    field_of_view = 4 * au.deg
    angular_separation = 32 * au.arcmin
    spatial_separation = 1000 * au.m
    observation_start_time = at.Time('2021-01-01T00:00:00', scale='utc')
    observation_duration = timedelta(minutes=0)
    temporal_resolution = timedelta(seconds=0)
    freqs = [700e6, 2000e6] * au.Hz
    ionosphere_gain_model = ionosphere_gain_model_factory(
        phase_tracking=phase_tracking,
        field_of_view=field_of_view,
        angular_separation=angular_separation,
        spatial_separation=spatial_separation,
        observation_start_time=observation_start_time,
        observation_duration=observation_duration,
        temporal_resolution=temporal_resolution,
        specification='light_dawn',
        array_name='dsa2000W',
        plot_folder='plot_ionosphere',
        cache_folder='cache_ionosphere',
        seed=42
    )


def test_ionosphere_gain_model():
    freqs = au.Quantity([700e6, 2000e6] * au.Hz)
    array_location = ac.EarthLocation(lat=0 * au.deg, lon=0 * au.deg, height=0 * au.m)

    radius = 10 * au.km
    spatial_separation = 1 * au.km

    antennas = create_spherical_earth_grid(
        center=array_location,
        radius=radius,
        dr=spatial_separation
    )

    radius = 10 * au.km
    spatial_separation = 3 * au.km
    model_antennas = create_spherical_earth_grid(
        center=array_location,
        radius=radius,
        dr=spatial_separation
    )

    phase_tracking = ac.ICRS(0 * au.deg, 0 * au.deg)
    model_directions = create_spherical_grid(
        pointing=phase_tracking,
        angular_width=0.5 * 4 * au.deg,
        dr=50 * au.arcmin
    )
    # ac.ICRS(ra=[0, 0.] * au.deg, dec=[0., 1.] * au.deg)
    model_times = at.Time(['2021-01-01T00:00:00', '2021-01-01T00:10:00'], scale='utc')

    ionosphere_gain_model = IonosphereGainModel(
        antennas=antennas,
        array_location=array_location,
        phase_tracking=phase_tracking,
        model_directions=model_directions,
        model_times=model_times,
        model_antennas=model_antennas,
        specification='light_dawn',
        plot_folder='plot_ionosphere_small_test',
        cache_folder='cache_ionosphere_small_test',
        # interp_mode='kriging'
    )

    assert ionosphere_gain_model.num_antenna == len(antennas)
    assert ionosphere_gain_model.ref_ant == array_location
    assert ionosphere_gain_model.ref_time == model_times[0]
    assert np.all(np.isfinite(ionosphere_gain_model.dtec))
    assert np.all(np.isfinite(ionosphere_gain_model.enu_geodesics_data))

    # Test calculate gain model

    sources = phase_tracking.reshape((-1,))
    # create_spherical_grid(
    #     pointing=phase_tracking,
    #     angular_width=0.5 * 4 * au.deg,
    #     dr=32 * au.arcmin
    # )

    times = at.Time(np.linspace(model_times[0].jd, model_times[-1].jd, 10), format='jd')
    for time in times:
        gains = ionosphere_gain_model.compute_gain(
            freqs=freqs,
            sources=sources,
            phase_tracking=phase_tracking,
            array_location=array_location,
            time=time
        )

        assert gains.shape == (len(sources), len(antennas), len(freqs), 2, 2)

        dtec = np.angle(gains[0, :, 0, 0, 0]) * freqs[0].to('MHz').value / ionosphere_gain_model.TEC_CONV
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))
        sc = ax[0][0].scatter(antennas.geodetic.lon.deg, antennas.geodetic.lat.deg, c=dtec, marker='o')
        fig.colorbar(sc, ax=ax[0][0])
        ax[0][0].scatter(model_antennas.geodetic.lon.deg, model_antennas.geodetic.lat.deg, marker='*',
                         c='red')
        ax[0][0].set_xlabel('Longitude')
        ax[0][0].set_ylabel('Latitude')
        ax[0][0].set_title(f"Dtec in Antennas at {time}")
        plt.show()

        # # Plot model directions
        # wcs = WCS(naxis=2)
        # wcs.wcs.ctype = ['RA---AIT', 'DEC--AIT']  # AITOFF projection
        # wcs.wcs.crval = [0, 0]  # Center of the projection
        # wcs.wcs.crpix = [0, 0]
        # wcs.wcs.cdelt = [-1, 1]
        # phase = np.angle(gains[:, i, 0, 0, 0])*180./np.pi
        # fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 10), subplot_kw=dict(projection=wcs))
        # sc = ax[0][0].scatter(sources.ra.deg, sources.dec.deg, c=phase, marker='o',
        #                       transform=ax[0][0].get_transform('world'))
        # fig.colorbar(sc, ax=ax[0][0])
        # # plot model directions
        # ax[0][0].scatter(model_directions.ra.deg, model_directions.dec.deg, marker='*',
        #                  transform=ax[0][0].get_transform('world'),
        #                  c='red')
        # ax[0][0].set_xlabel('Right Ascension')
        # ax[0][0].set_ylabel('Declination')
        # ax[0][0].set_title(f"Phase in Directions: {antennas[i]}")
        # plt.show()
    # fig.savefig(os.path.join(self.plot_folder, "model_directions.png"))
    # plt.close(fig)
