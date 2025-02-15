import time

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_cal.common.wgridder import vis_to_image
from dsa2000_common.delay_models.base_far_field_delay_engine import build_far_field_delay_engine


from dsa2000_common.delay_models.base_near_field_delay_engine import build_near_field_delay_engine


from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model


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


def main(num_ant: int, num_times: int, num_channels: int):
    phase_center, antennas, visibility_coords, geodesic_model, far_field_delay_engine, near_field_delay_engine = build_mock_obs_setup(
        ant=num_ant,
        time=num_times,
        num_freqs=num_channels
    )

    uvw = visibility_coords.uvw.reshape((-1, 3))

    num_rows = visibility_coords.uvw.shape[0] * visibility_coords.uvw.shape[1]
    num_freqs = visibility_coords.freqs.shape[0]
    vis = jnp.ones((num_rows, num_freqs), dtype=jnp.complex64)

    def grid(freqs, uvw, visibilities):
        def single_freq(freq, vis):
            return vis_to_image(
                uvw=uvw,
                freqs=freq[None],
                vis=vis[:, None],
                pixsize_l=3.3 / 3600 / 5,
                pixsize_m=3.3 / 3600 / 5,
                center_l=0,
                center_m=0,
                npix_l=17500,
                npix_m=17500
            )

        return jax.vmap(single_freq, in_axes=(0, 1))(freqs, visibilities)

    predict_jit = jax.jit(grid).lower(visibility_coords.freqs,
                                      uvw,
                                      vis).compile()

    t0 = time.time()
    jax.block_until_ready(predict_jit(visibility_coords.freqs,
                                      uvw,
                                      vis
                                      ))
    t1 = time.time()
    print(f"Execution time: {t1 - t0}")


if __name__ == '__main__':
    main(
        num_ant=2048,
        num_times=1,
        num_channels=40
    )
