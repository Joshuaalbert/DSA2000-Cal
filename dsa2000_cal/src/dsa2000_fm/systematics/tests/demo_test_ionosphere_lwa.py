import jax
import numpy as np
from astropy import time as at, units as au, coordinates as ac

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.astropy_utils import create_spherical_spiral_grid
from dsa2000_common.common.enu_frame import ENU
from dsa2000_fm.systematics.ionosphere import build_ionosphere_gain_model, construct_canonical_ionosphere, \
    compute_x0_radius


def test_ionosphere_dtec_gain_model():
    ref_time = at.Time.now()
    times = ref_time + 10 * np.arange(1) * au.s
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('lwa'))
    antennas = array.get_antennas()
    ref_location = array.get_array_location()
    phase_center = ENU(0, 0, 1, obstime=ref_time, location=ref_location).transform_to(ac.ICRS())

    directions = create_spherical_spiral_grid(
        pointing=phase_center,
        num_points=20,
        angular_radius=90 * au.deg
    )
    print(f"Number of directions: {len(directions)}")

    x0_radius = compute_x0_radius(ref_location, ref_time)
    ionosphere = construct_canonical_ionosphere(
        x0_radius=x0_radius,
        turbulent=True,
        dawn=True,
        high_sun_spot=True
    )

    # T = int((times.max() - times.min()) / (1 * au.min)) + 1
    # model_times = times.min() + np.arange(0., T) * au.min
    model_freqs = array.get_channels()[[0, len(array.get_channels()) // 2, -1]]
    gain_model = build_ionosphere_gain_model(
        key=jax.random.PRNGKey(0),
        ionosphere=ionosphere,
        model_freqs=model_freqs,
        antennas=antennas,
        ref_location=ref_location,
        times=times,
        ref_time=ref_time,
        directions=directions,
        phase_centre=phase_center,
        full_stokes=False,
        spatial_resolution=0.5 * au.km,
        save_file='simulated_lwa_dtec.json'
    )
    gain_model.plot_regridded_beam(ant_idx=-1)
