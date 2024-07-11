import os

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=10"

import jax
import numpy as np
from astropy import units as au, coordinates as ac, time as at
from jax import numpy as jnp

from dsa2000_cal.common.astropy_utils import create_spherical_grid, create_spherical_earth_grid
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.forward_model.systematics.ionosphere_gain_model import ionosphere_gain_model_factory, \
    interpolate_antennas
from dsa2000_cal.forward_model.systematics.ionosphere_simulation import IonosphereSimulation, msqrt


def test_msqrt():
    M = jax.random.normal(jax.random.PRNGKey(42), (100, 100))
    A = M @ M.T
    max_eig, min_eig, L = msqrt(A)
    np.testing.assert_allclose(A, L @ L.T, atol=2e-4)


def test_real_ionosphere_gain_model():
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    field_of_view = 4 * au.deg
    spatial_resolution = 2.0 * au.km
    observation_start_time = at.Time('2021-01-01T00:00:00', scale='utc')
    observation_duration = 0 * au.s
    temporal_resolution = 0 * au.s
    model_freqs = [700e6, 2000e6] * au.Hz
    ionosphere_gain_model = ionosphere_gain_model_factory(
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
    model_directions = create_spherical_grid(
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


def test_interpolate_antennas():
    N = 4
    M = 5
    num_time = 6
    num_dir = 7
    dtec_interp = interpolate_antennas(
        antennas_enu=jnp.ones((N, 3)),
        model_antennas_enu=jnp.ones((M, 3)),
        dtec=jnp.ones((num_time, num_dir, M))
    )
    assert dtec_interp.shape == (num_time, num_dir, N)
