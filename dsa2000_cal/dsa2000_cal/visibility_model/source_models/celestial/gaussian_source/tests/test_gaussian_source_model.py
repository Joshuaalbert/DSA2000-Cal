import itertools
import time

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.uvw.far_field import VisibilityCoords
from dsa2000_cal.visibility_model.source_models.celestial.gaussian_source.gaussian_source_model import \
    GaussianSourceModel, GaussianPredict, GaussianModelData


def test_gaussian_sources():
    fill_registries()
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    # source_file = source_model_registry.get_instance(source_model_registry.get_match('cas_a')).get_wsclean_source_file()
    # -00:36:28.234,58.50.46.396
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "58d50m46.396s", frame='icrs')
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "78d50m46.396s", frame='icrs')

    source_file = source_model_registry.get_instance(
        source_model_registry.get_match('cyg_a')).get_wsclean_clean_component_file()
    # -04:00:28.608,40.43.33.595
    phase_tracking = ac.ICRS(ra=-4 * au.hour, dec=40 * au.deg)

    freqs = au.Quantity([50e6, 80e6], 'Hz')

    gaussian_source_model = GaussianSourceModel.from_wsclean_model(
        wsclean_clean_component_file=source_file,
        phase_tracking=phase_tracking,
        freqs=freqs,
        lmn_transform_params=True
    )

    assert isinstance(gaussian_source_model, GaussianSourceModel)
    assert gaussian_source_model.num_sources > 0

    gaussian_source_model.plot()


@pytest.mark.parametrize("di_gains", [True, False])
@pytest.mark.parametrize("order_approx", [0, 1])
def test_gaussian_predict(di_gains: bool, order_approx: int):
    gaussian_predict = GaussianPredict(order_approx=order_approx)
    row = 100
    chan = 4
    source = 1
    time = 15
    ant = 24
    lm = 1e-3 * jnp.ones((source, 2))
    n = jnp.sqrt(1. - jnp.sum(lm ** 2, axis=-1))
    lmn = jnp.concatenate([lm, n[:, None]], axis=-1)
    if di_gains:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (source, time, ant, chan, 2, 2)
    freqs = jnp.ones((chan,))
    model_data = GaussianModelData(
        image=jnp.ones((source, chan, 2, 2), dtype=jnp.complex64),
        gains=jnp.ones(gain_shape,
                       dtype=jnp.complex64),
        lmn=lmn,
        ellipse_params=jnp.ones((source, 3)),
        freqs=freqs
    )
    visibility_coords = VisibilityCoords(
        uvw=jnp.ones((row, 3)),
        time_obs=jnp.ones((row,)),
        antenna_1=jnp.ones((row,), jnp.int64),
        antenna_2=jnp.ones((row,), jnp.int64),
        time_idx=jnp.ones((row,), jnp.int64)
    )
    visibilities = gaussian_predict.predict(model_data=model_data, visibility_coords=visibility_coords)
    print(order_approx, visibilities)
    assert np.all(np.isfinite(visibilities))
    assert np.shape(visibilities) == (row, chan, 2, 2)

    # Note: correctness is tested against wgridder


@pytest.mark.parametrize("di_gains", [True, False])
@pytest.mark.parametrize("order_approx", [0])
def test_ensure_gradients_work(di_gains: bool, order_approx: int):
    gaussian_predict = GaussianPredict(order_approx=order_approx)
    row = 100
    chan = 4
    source = 2
    time = 2
    ant = 3
    lm = 0. * jax.random.normal(jax.random.PRNGKey(42), (source, 2))
    n = jnp.sqrt(1. - jnp.sum(lm ** 2, axis=-1))
    lmn = jnp.concatenate([lm, n[:, None]], axis=-1)
    if di_gains:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (source, time, ant, chan, 2, 2)

    freqs = jnp.linspace(700e6, 2000e6, chan)

    antennas = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (ant, 3))
    antenna_1 = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, ant)
    antenna_2 = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, ant)

    uvw = antennas[antenna_2] - antennas[antenna_1]
    uvw = uvw.at[:, 2].mul(1e-3)

    times = jnp.linspace(0, 1, time)
    time_idx = jax.random.randint(jax.random.PRNGKey(42), (row,), 0, time)
    time_obs = times[time_idx]

    image = jnp.zeros((source, chan, 2, 2), dtype=jnp.complex64)
    image = image.at[:, :, 0, 0].set(1.)
    image = image.at[:, :, 1, 1].set(1.)

    gains = jax.random.normal(jax.random.PRNGKey(42), gain_shape) + 1j * jax.random.normal(jax.random.PRNGKey(43),
                                                                                           gain_shape)
    # 50arcsec FWHM
    ellipse_params = jnp.zeros((source, 3))
    ellipse_params = ellipse_params.at[:, 0].set(50 / 3600. * np.pi / 180.)
    ellipse_params = ellipse_params.at[:, 1].set(50 / 3600. * np.pi / 180.)
    ellipse_params = ellipse_params.at[:, 2].set(0.)

    model_data = GaussianModelData(
        image=image,
        gains=gains,
        lmn=lmn,
        ellipse_params=ellipse_params,
        freqs=freqs
    )

    def func(model_data: GaussianModelData, uvw: jax.Array):
        visibility_coords = VisibilityCoords(
            uvw=uvw,
            time_obs=time_obs,
            antenna_1=antenna_1,
            antenna_2=antenna_2,
            time_idx=time_idx
        )
        vis = gaussian_predict.predict(model_data=model_data, visibility_coords=visibility_coords)

        return vis

    grad = jax.grad(lambda *args: jnp.sum(jnp.abs(func(*args)) ** 2), argnums=(0, 1))(model_data, uvw)
    # print(func(freqs, model_data, uvw))
    # print(grad)
    (model_data_grad, uvw_grad) = grad
    if di_gains:
        # gain_shape = (time, ant, chan, 2, 2)
        for t in range(time):
            for a in range(ant):
                print(f"Time: {t}, Ant: {a}")
                print("\tXX", model_data_grad.gains[t, a, :, 0, 0])
                print("\tXY", model_data_grad.gains[t, a, :, 0, 1])
                print("\tYX", model_data_grad.gains[t, a, :, 1, 0])
                print("\tYY", model_data_grad.gains[t, a, :, 1, 1])
                # Ensure gradient is not zero
                assert np.all(np.abs(model_data_grad.gains[t, a, :, :, :]) > 1e-10)

    else:
        # gain_shape = (source, time, ant, chan, 2, 2)
        for s in range(source):
            for t in range(time):
                for a in range(ant):
                    print(f"Source: {s}, Time: {t}, Ant: {a}")
                    print("\tXX", model_data_grad.gains[s, t, a, :, 0, 0])
                    print("\tXY", model_data_grad.gains[s, t, a, :, 0, 1])
                    print("\tYX", model_data_grad.gains[s, t, a, :, 1, 0])
                    print("\tYY", model_data_grad.gains[s, t, a, :, 1, 1])
                    # Ensure gradient is not zero
                    assert np.all(np.abs(model_data_grad.gains[s, t, a, :, :, :]) > 1e-10)


def test_with_sharding():
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec
    from jax.sharding import NamedSharding

    P = PartitionSpec

    devices = mesh_utils.create_device_mesh((len(jax.devices()),))
    mesh = Mesh(devices, axis_names=('chan',))

    def tree_device_put(tree, sharding):
        return jax.tree_map(lambda x: jax.device_put(x, sharding), tree)

    gaussian_predict = GaussianPredict()
    row = 100
    chan = 4
    source = 1
    time = 15
    ant = 24
    lm = 1e-3 * jnp.ones((source, 2))
    n = jnp.sqrt(1. - jnp.sum(lm ** 2, axis=-1))
    lmn = jnp.concatenate([lm, n[:, None]], axis=-1)
    ellipse_params = jnp.ones((source, 3))

    image = jnp.ones((source, chan, 2, 2), dtype=jnp.float64)
    gains = jnp.ones((source, time, ant, chan, 2, 2), dtype=jnp.complex64)

    freqs = jnp.ones((chan,))
    freqs = tree_device_put(freqs, NamedSharding(mesh, P('chan')))

    model_data = GaussianModelData(
        image=tree_device_put(image, NamedSharding(mesh, P(None, 'chan'))),
        gains=tree_device_put(gains, NamedSharding(mesh, P(None, None, None, 'chan'))),
        lmn=tree_device_put(lmn, NamedSharding(mesh, P())),
        ellipse_params=tree_device_put(ellipse_params, NamedSharding(mesh, P())),
        freqs=freqs
    )

    uvw = jnp.ones((row, 3))
    time = jnp.ones((row,))
    antenna_1 = jnp.ones((row,), jnp.int64)
    antenna_2 = jnp.ones((row,), jnp.int64)
    time_idx = jnp.ones((row,), jnp.int64)

    visibility_coords = VisibilityCoords(
        uvw=tree_device_put(uvw, NamedSharding(mesh, P())),
        time_obs=tree_device_put(time, NamedSharding(mesh, P())),
        antenna_1=tree_device_put(antenna_1, NamedSharding(mesh, P())),
        antenna_2=tree_device_put(antenna_2, NamedSharding(mesh, P())),
        time_idx=tree_device_put(time_idx, NamedSharding(mesh, P()))
    )

    visibilities = gaussian_predict.predict(model_data=model_data, visibility_coords=visibility_coords)
    assert np.all(np.isfinite(visibilities))


@pytest.mark.parametrize("di_gains", [True, False, None])
def test_benchmark_performance(di_gains):
    # Benchmark performance
    # di_gains = True Time taken for 2048 antennas, 1 channels, 100 sources: 14.163064 s | 9.324887 s | 8.610048 s | 8.988812 s | 11.194738 s | 9.603148 s
    # di_gains = False Time taken for 2048 antennas, 1 channels, 100 sources: 9.601373 s | 13.285119 s | 11.940015 s | 12.057437 s | 9.140454 s | 9.324197 s
    # di_gains = None Time taken for 2048 antennas, 1 channels, 100 sources: 5.436340 s | 8.067334 s | 6.951573 s | 7.103243 s | 5.576627 s | 4.885742 s
    dft_predict = GaussianPredict()
    num_time = 1
    for num_ant in [2048]:
        for num_chan in [1]:
            for num_source in [100]:
                freqs = jnp.linspace(700e6, 2000e6, num_chan)

                antennas = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (num_ant, 3))
                antenna_1, antenna_2 = jnp.asarray(
                    list(itertools.combinations_with_replacement(range(num_ant), 2))).T

                num_rows = len(antenna_1)

                uvw = antennas[antenna_2] - antennas[antenna_1]
                uvw = uvw.at[:, 2].mul(1e-3)

                times = jnp.arange(num_time) * 1.5
                time_idx = jnp.zeros((num_rows,), jnp.int64)
                time_obs = times[time_idx]

                image = jnp.zeros((num_source, num_chan, 2, 2), dtype=jnp.complex64)
                image = image.at[..., 0, 0].set(0.5)
                image = image.at[..., 1, 1].set(0.5)
                if di_gains is None:
                    gains = None
                else:
                    if di_gains:
                        gain_shape = (num_time, num_ant, num_chan, 2, 2)
                    else:
                        gain_shape = (num_source, num_time, num_ant, num_chan, 2, 2)

                    gains = jnp.ones(gain_shape) + 1j * jnp.zeros(gain_shape)
                    gains = gains.at[..., 1, 0].set(0.)
                    gains = gains.at[..., 0, 1].set(0.)

                lmn = jax.random.normal(jax.random.PRNGKey(42), (num_source, 3))
                lmn /= jnp.linalg.norm(lmn, axis=-1, keepdims=True)

                ellipse_params = jnp.zeros((num_source, 3))

                model_data = GaussianModelData(
                    freqs=freqs,
                    image=image,
                    gains=gains,
                    lmn=lmn,
                    ellipse_params=ellipse_params
                )

                visibility_coords = VisibilityCoords(
                    uvw=uvw,
                    time_obs=time_obs,
                    antenna_1=antenna_1,
                    antenna_2=antenna_2,
                    time_idx=time_idx
                )

                f = jax.jit(dft_predict.predict).lower(model_data=model_data,
                                                       visibility_coords=visibility_coords).compile()

                t0 = time.time()
                visibilities = f(model_data=model_data, visibility_coords=visibility_coords).block_until_ready()
                t1 = time.time()
                print(f"Time taken for {num_ant} antennas, {num_chan} channels, {num_source} sources: {t1 - t0:.6f} s")
