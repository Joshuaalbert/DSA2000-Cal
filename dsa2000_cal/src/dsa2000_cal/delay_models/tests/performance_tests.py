import itertools
import time

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tomographic_kernel.frames import ENU

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.quantity_utils import time_to_jnp
from dsa2000_cal.delay_models.base_far_field_delay_engine import build_far_field_delay_engine


@pytest.mark.parametrize('n', [2, 10, 100, 1000, 2048])
def test_uvw_performance(n: int):
    # Setup test parameters
    obstime = at.Time("2021-01-01T00:00:00", scale='utc')

    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    antennas = array.get_antennas()
    array_location = antennas[0]
    # array_location = ac.EarthLocation.of_site('vla')
    # antennas = ENU(
    #     east=np.linspace(0., 20, n) * au.km,
    #     north=np.zeros(n) * au.km,
    #     up=np.zeros(n) * au.km,
    #     location=array_location,
    #     obstime=obstime
    # ).transform_to(ac.ITRS(obstime=obstime, location=array_location)).earth_location
    phase_center = ENU(east=0, north=0, up=1, location=array_location, obstime=obstime).transform_to(ac.ICRS())

    engine = build_far_field_delay_engine(
        antennas=antennas,
        phase_center=phase_center,
        start_time=obstime,
        end_time=obstime + (10.3 * 60) * au.s,
        ref_time=obstime,
        verbose=True

    )

    baseline_pairs = np.asarray(list(itertools.combinations_with_replacement(range(n), 2)),
                                dtype=np.int32)
    antenna_1 = baseline_pairs[:, 0]
    antenna_2 = baseline_pairs[:, 1]

    data_dict = dict(
        times=jnp.repeat(time_to_jnp(obstime, obstime)[None], len(antenna_1), axis=0),
        antenna_1=jnp.asarray(antenna_1),
        antenna_2=jnp.asarray(antenna_2)
    )
    data_dict = jax.device_put(data_dict)

    t0 = time.time()
    compute_uvw_jax = jax.jit(engine.compute_uvw).lower(**data_dict).compile()
    compile_time = time.time() - t0
    print(f"Compilation time for n={n}: {compile_time:.4f} seconds")

    t0 = time.time()
    uvw = jax.block_until_ready(compute_uvw_jax(**data_dict))
    compute_time = time.time() - t0
    print(f"Compute time for n={n}: {compute_time:.4f} seconds")
    # [num_baselines, 3]
