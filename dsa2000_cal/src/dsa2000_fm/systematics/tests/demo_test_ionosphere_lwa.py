import jax
import numpy as np
from astropy import time as at, units as au, coordinates as ac

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.astropy_utils import create_spherical_spiral_grid, get_time_of_local_meridean
from dsa2000_common.common.enu_frame import ENU
from dsa2000_fm.systematics.ionosphere import construct_canonical_ionosphere, \
    compute_x0_radius, simulate_ionosphere


def test_ionosphere_tec_simulation():
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('lwa'))
    antennas = array.get_antennas()
    ref_location = array.get_array_location()
    phase_center = ENU(
        0, 0, 1, obstime=at.Time('2025-06-10T09:00:00', scale='utc'),
        location=ref_location
    ).transform_to(ac.ICRS())
    # Or if you know the ICRS coord you can find the time when it is at the local transit
    ref_time = get_time_of_local_meridean(phase_center, array.get_array_location(),
                                          at.Time('2025-06-10T09:00:00', scale='utc'))
    times = ref_time + 10 * np.arange(10) * au.s

    directions = create_spherical_spiral_grid(
        pointing=phase_center,
        num_points=20,
        angular_radius=1 * au.deg
    )
    print(f"Number of directions: {len(directions)}")

    x0_radius = compute_x0_radius(ref_location, ref_time)
    ionosphere = construct_canonical_ionosphere(
        x0_radius=x0_radius,
        turbulent=True,
        dawn=True,
        high_sun_spot=True
    )

    simulate_ionosphere(
        key=jax.random.PRNGKey(0),
        ionosphere=ionosphere,
        antennas=antennas,
        ref_location=ref_location,
        times=times,
        ref_time=ref_time,
        directions=directions,
        spatial_resolution=0.5 * au.km,
        predict_batch_size=512,
        do_tec=True,
        save_file='simulated_lwa_tec.json'
    )
