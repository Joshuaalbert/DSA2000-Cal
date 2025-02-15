import os
import time
os.environ['JAX_PLATFORMS'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_common.delay_models import build_far_field_delay_engine
from dsa2000_common.delay_models import build_near_field_delay_engine
from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model
from dsa2000_cal.visibility_model.source_models.celestial.base_point_source_model import build_point_source_model


def build_mock_obs_setup(ant: int, time: int, num_freqs: int):
    array_location = ac.EarthLocation.of_site('vla')
    ref_time = at.Time('2021-01-01T00:00:00', scale='utc')
    obstimes = ref_time + np.arange(time) * au.s
    phase_center = ENU(0, 0, 1, location=array_location, obstime=ref_time).transform_to(ac.ICRS())
    freqs = np.linspace(700, 2000, num_freqs) * au.MHz

    pointing = phase_center
    antennas = ENU(
        east=np.random.uniform(low=-10, high=10, size=ant) * au.km,
        north=np.random.uniform(low=-10, high=10, size=ant) * au.km,
        up=np.random.uniform(low=-10, high=10, size=ant) * au.m,
        location=array_location,
        obstime=ref_time
    ).transform_to(ac.ITRS(location=array_location, obstime=ref_time)).earth_location

    geodesic_model = build_geodesic_model(
        antennas=antennas,
        array_location=array_location,
        phase_center=phase_center,
        obstimes=obstimes,
        ref_time=ref_time,
        pointings=pointing
    )

    far_field_delay_engine = build_far_field_delay_engine(
        antennas=antennas,
        phase_center=phase_center,
        start_time=obstimes.min(),
        end_time=obstimes.max(),
        ref_time=ref_time
    )

    near_field_delay_engine = build_near_field_delay_engine(
        antennas=antennas,
        start_time=obstimes.min(),
        end_time=obstimes.max(),
        ref_time=ref_time
    )

    visibility_coords = far_field_delay_engine.compute_visibility_coords(
        freqs=quantity_to_jnp(freqs),
        times=time_to_jnp(obstimes, ref_time)
    )
    return phase_center, antennas, visibility_coords, geodesic_model, far_field_delay_engine, near_field_delay_engine


def main(num_directions: int, num_ant: int, num_times: int, num_channels: int, num_model_freqs: int):
    model_freqs = np.linspace(0.7, 2, num_model_freqs) * au.GHz
    ra = np.random.uniform(0, 2 * np.pi, num_directions) * au.rad
    dec = np.random.uniform(-np.pi / 2, np.pi / 2, num_directions) * au.rad
    A = np.ones((num_directions, num_model_freqs, 2, 2), dtype=np.float32) * au.Jy
    point_source_model = build_point_source_model(
        model_freqs=model_freqs,
        ra=ra,
        dec=dec,
        A=A
    )

    phase_center, antennas, visibility_coords, geodesic_model, far_field_delay_engine, near_field_delay_engine = build_mock_obs_setup(
        ant=num_ant,
        time=num_times,
        num_freqs=num_channels
    )

    def predict(visibility_coords,
                near_field_delay_engine,
                far_field_delay_engine,
                geodesic_model):
        return point_source_model.predict(
            visibility_coords=visibility_coords,
            gain_model=None,
            near_field_delay_engine=near_field_delay_engine,
            far_field_delay_engine=far_field_delay_engine,
            geodesic_model=geodesic_model
        )

    predict_jit = jax.jit(predict).lower(visibility_coords,
                                         near_field_delay_engine,
                                         far_field_delay_engine,
                                         geodesic_model).compile()

    t0 = time.time()
    jax.block_until_ready(predict_jit(visibility_coords,
                                      near_field_delay_engine,
                                      far_field_delay_engine,
                                      geodesic_model,
                                      ))
    t1 = time.time()
    print(f"Execution time: {t1 - t0}")


if __name__ == '__main__':
    main(
        num_directions=1,
        num_ant=2048,
        num_times=1,
        num_channels=1,
        num_model_freqs=3
    )