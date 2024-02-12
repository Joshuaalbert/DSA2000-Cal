import os
# Force 2 jax  hosts
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import time
from typing import NamedTuple

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pytest
from jax import numpy as jnp
from tomographic_kernel.frames import ENU

from dsa2000_cal.calibration import vec, unvec, kron_product, DFTPredict, ModelData, VisibilityCoords, Calibration, \
    CalibrationData
from dsa2000_cal.coord_utils import earth_location_to_uvw, create_uvw_frame, icrs_to_lmn


def test_vec():
    a = jnp.asarray([[1, 2],
                     [3, 4]])
    assert jnp.alltrue(vec(a) == jnp.asarray([1, 3, 2, 4]))

    assert jnp.alltrue(unvec(vec(a), (2, 2)) == a)
    assert jnp.alltrue(unvec(vec(a)) == a)


def test_kron_product():
    a = jnp.arange(4).reshape((2, 2)).astype(complex)
    b = jnp.arange(4).reshape((2, 2)).astype(complex)
    c = jnp.arange(4).reshape((2, 2)).astype(complex)

    def f(a, b, c):
        return a @ b @ c

    p1 = f(a, b, c)

    p2 = kron_product(a, b, c)

    assert np.alltrue(p2 == p1)

    a1 = jax.jit(f).lower(a, b, c).compile().cost_analysis()[0]
    a2 = jax.jit(kron_product).lower(a, b, c).compile().cost_analysis()[0]
    print()
    print("a @ b @ c")
    print("Naive a.b.c | unvec(kron(c.T, a).vec(b))")
    for key in ['bytes accessed', 'flops', 'utilization operand 0 {}', 'utilization operand 1 {}',
                'utilization operand 2 {}', 'bytes accessed output {}']:
        print(key, ":", a1.get(key, None), a2.get(key, None))

    print()
    print("a @ b @ c.T.conj")
    print("Naive a.b.c | unvec(kron(c.T, a).vec(b))")
    a1 = jax.jit(lambda a, b, c: f(a, b, c.T.conj())).lower(a, b, c).compile().cost_analysis()[0]
    a2 = jax.jit(lambda a, b, c: kron_product(a, b, c.T.conj())).lower(a, b, c).compile().cost_analysis()[0]
    for key in ['bytes accessed', 'flops', 'utilization operand 0 {}', 'utilization operand 1 {}',
                'utilization operand 2 {}', 'bytes accessed output {}']:
        print(key, ":", a1.get(key, None), a2.get(key, None))

    print()
    print("a @ b @ c.conj.T")
    print("Naive a.b.c | unvec(kron(c.T, a).vec(b))")
    a1 = jax.jit(lambda a, b, c: f(a, b, c.conj().T)).lower(a, b, c).compile().cost_analysis()[0]
    a2 = jax.jit(lambda a, b, c: kron_product(a, b, c.conj().T)).lower(a, b, c).compile().cost_analysis()[0]
    for key in ['bytes accessed', 'flops', 'utilization operand 0 {}', 'utilization operand 1 {}',
                'utilization operand 2 {}', 'bytes accessed output {}']:
        print(key, ":", a1.get(key, None), a2.get(key, None))


def test_dft_predict():
    dft_predict = DFTPredict()
    row = 100
    chan = 4
    source = 1
    time = 15
    ant = 24
    model_data = ModelData(
        image=jnp.ones((source, chan, 2, 2), dtype=jnp.complex64),
        gains=jnp.ones((source, time, ant, chan, 2, 2), dtype=jnp.complex64),
        lm=1e-3 * jnp.ones((source, 2))
    )
    visibility_coords = VisibilityCoords(
        uvw=jnp.ones((row, 3)),
        time=jnp.ones((row,)),
        antenna_1=jnp.ones((row,), jnp.int64),
        antenna_2=jnp.ones((row,), jnp.int64),
        time_idx=jnp.ones((row,), jnp.int64)
    )
    freq = jnp.ones((chan,))
    visibilities = dft_predict.predict(model_data=model_data, visibility_coords=visibility_coords, freq=freq)
    assert np.all(np.isfinite(visibilities))


class MockData(NamedTuple):
    """
    Mock data for testing calibration.
    """
    num_source: int
    num_time: int
    num_ant: int
    num_chan: int
    calibration_data: CalibrationData


@pytest.fixture(scope='package')
def mock_data():
    dft_predict = DFTPredict(chunksize=2)
    num_time = 1
    num_ant = 2048
    num_chan = 2
    num_sources = 1

    # row = (ant * (ant - 1) // 2 + ant) * Nt

    array_location = ac.EarthLocation.of_site('vla')
    # array_location = ac.EarthLocation.from_geodetic(lon=0.*au.deg, lat=0.*au.deg, height=0.*au.m)
    t0_mjs = at.Time('2000-01-01T00:00:00', format='isot').mjd * 86400.
    times = at.Time((t0_mjs + np.linspace(0., 15 * 60, num_time)) / 86400., format='mjd')
    enu_frame = ENU(location=ac.EarthLocation.of_site('vla').itrs, obstime=times[0])
    pointing = ac.SkyCoord(east=0., north=0., up=1., frame=enu_frame).transform_to(
        ac.ICRS)  # ac.ICRS(0 * au.deg, 45 * au.deg)

    # print(pointing)

    freq = jnp.linspace(700e6, 800e6, num_chan)
    num_chan = len(freq)

    antennas = ac.SkyCoord(
        east=np.random.uniform(size=(num_ant,), low=-10, high=10) * au.km,
        north=np.random.uniform(size=(num_ant,), low=-10, high=10) * au.km,
        up=np.random.uniform(size=(num_ant,), low=-0.1, high=0.1) * au.km,
        frame=enu_frame
    ).transform_to(ac.ITRS)

    tril_indices = np.tril_indices(num_ant, 0)
    uvw = []
    time_coords = []
    antenna_1 = []
    antenna_2 = []
    time_idx = []
    num_pairs = len(tril_indices[0])
    for i, time in enumerate(times):
        antennas_uvw = earth_location_to_uvw(antennas.earth_location, array_location, time, pointing)

        uvw.append((antennas_uvw[tril_indices[0]] - antennas_uvw[tril_indices[1]]).to('m').value)
        antenna_1.append(tril_indices[0])
        antenna_2.append(tril_indices[1])
        time_idx.append(np.full((num_pairs,), i))
        time_coords.append(np.full((num_pairs,), time.mjd * 86400.))

    uvw = jnp.concatenate(uvw, axis=0)
    time_coords = jnp.concatenate(time_coords)
    antenna_1 = jnp.concatenate(antenna_1)
    antenna_2 = jnp.concatenate(antenna_2)
    time_idx = jnp.concatenate(time_idx)

    sources = ac.ICRS(np.linspace(pointing.ra.deg, pointing.ra.deg + 0.1, num_sources) * au.deg,
                      np.linspace(pointing.dec.deg, pointing.dec.deg + 0.1, num_sources) * au.deg)

    l,m,n = icrs_to_lmn(sources=sources, array_location=array_location, time=times[0], pointing=pointing).T
    lm = jnp.stack([l, m], axis=-1)

    model_data = ModelData(
        image=jnp.ones((num_sources, num_chan, 2, 2), dtype=jnp.complex64),
        gains=jnp.ones((num_sources, num_time, num_ant, num_chan, 2, 2), dtype=jnp.complex64),
        lm=lm
    )

    model_data = model_data._replace(
        gains=0.8*model_data.gains.at[..., 0, 1].set(0.).at[..., 1, 0].set(0.),
        image=model_data.image.at[..., 0, 1].set(0.).at[..., 1, 0].set(0.)
    )

    visibility_coords = VisibilityCoords(
        uvw=uvw,
        time=time_coords,
        antenna_1=antenna_1,
        antenna_2=antenna_2,
        time_idx=time_idx
    )

    visibilities = dft_predict.predict(model_data=model_data, visibility_coords=visibility_coords, freq=freq)
    assert np.all(np.isfinite(visibilities))

    uncert = 0.5 * jnp.mean(jnp.abs(visibilities))

    visibilities += uncert * (jax.random.normal(jax.random.PRNGKey(0), visibilities.shape) + 1j * jax.random.normal(
        jax.random.PRNGKey(1), visibilities.shape)).astype(dft_predict.dtype)

    calibration_data = CalibrationData(
        visibility_coords=visibility_coords,
        image=model_data.image,
        lm=model_data.lm,
        freq=freq,
        obs_vis=visibilities,
        obs_vis_weight=uncert
    )

    return MockData(
        num_source=num_sources,
        num_time=num_time,
        num_ant=num_ant,
        num_chan=num_chan,
        calibration_data=calibration_data
    )


def test_calibration(mock_data: MockData):
    calibration = Calibration(chunksize=1, use_pjit=True)
    init_params = calibration.get_init_params(
        num_source=mock_data.num_source,
        num_time=mock_data.num_time,
        num_ant=mock_data.num_ant,
        num_chan=mock_data.num_chan
    )
    print(mock_data.calibration_data.obs_vis.shape)
    # print(init_params)
    solve_compiled = jax.jit(calibration.solve, donate_argnums=[0, 1]).lower(init_params,
                                                                             mock_data.calibration_data).compile()
    t0 = time.time()
    params, opt_results = solve_compiled(init_params, mock_data.calibration_data)
    params.gains_real.block_until_ready()
    print(f"Time: {time.time() - t0} seconds.")
    print(params, opt_results)
