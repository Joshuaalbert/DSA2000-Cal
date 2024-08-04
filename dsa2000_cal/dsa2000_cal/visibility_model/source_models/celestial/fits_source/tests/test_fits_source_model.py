import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.visibility_model.source_models.celestial.fits_source.fits_source_model import FITSSourceModel, \
    FITSPredict, FITSModelData


@pytest.mark.parametrize('source', ['cas_a', 'cyg_a', 'tau_a', 'vir_a'])
def test_fits_sources(source):
    fill_registries()
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    # source_file = source_model_registry.get_instance(source_model_registry.get_match('cas_a')).get_wsclean_source_file()
    # -00:36:28.234,58.50.46.396
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "58d50m46.396s", frame='icrs')
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "78d50m46.396s", frame='icrs')

    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match(source)).get_wsclean_fits_files()
    # -04:00:28.608,40.43.33.595
    phase_tracking = ac.ICRS(ra=-4 * au.hour, dec=40 * au.deg)

    freqs = au.Quantity([65e6, 77e6], 'Hz')

    fits_sources = FITSSourceModel.from_wsclean_model(wsclean_fits_files=wsclean_fits_files,
                                                      phase_tracking=phase_tracking, freqs=freqs, full_stokes=False)
    assert isinstance(fits_sources, FITSSourceModel)

    fits_sources.plot()


@pytest.mark.parametrize("is_gains", [True, False])
@pytest.mark.parametrize("full_stokes", [True, False])
@pytest.mark.parametrize("image_has_chan", [True, False])
def test_faint_predict(is_gains: bool, image_has_chan: bool, full_stokes: bool):
    faint_predict = FITSPredict()
    Nx = 100
    Ny = 100
    chan = 4
    time = 15
    ant = 24
    row = 1000
    if full_stokes:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (time, ant, chan)
    if is_gains:
        gains = jnp.ones(gain_shape, dtype=jnp.complex64)
    else:
        gains = None
    if image_has_chan:
        if full_stokes:
            image_shape = (chan, Nx, Ny, 2, 2)
        else:
            image_shape = (chan, Nx, Ny)
        l0 = jnp.zeros((chan,))
        m0 = jnp.zeros((chan,))
        dl = 0.01 * jnp.ones((chan,))
        dm = 0.01 * jnp.ones((chan,))
    else:
        if full_stokes:
            image_shape = (Nx, Ny, 2, 2)
        else:
            image_shape = (Nx, Ny)
        l0 = jnp.zeros(())
        m0 = jnp.zeros(())
        dl = 0.01 * jnp.ones(())
        dm = 0.01 * jnp.ones(())
    image = jnp.ones(image_shape, dtype=jnp.float32)
    freqs = jnp.ones((chan,))
    model_data = FITSModelData(
        image=image,
        gains=gains,
        l0=l0, m0=m0,
        dl=dl, dm=dm,
        freqs=freqs
    )
    visibility_coords = VisibilityCoords(
        uvw=jnp.ones((row, 3)),
        time_obs=jnp.ones((row,)),
        antenna_1=jnp.ones((row,), jnp.int64),
        antenna_2=jnp.ones((row,), jnp.int64),
        time_idx=jnp.ones((row,), jnp.int64)
    )
    visibilities = faint_predict.predict(model_data=model_data, visibility_coords=visibility_coords)
    assert np.all(np.isfinite(visibilities))
    if full_stokes:
        assert np.shape(visibilities) == (row, chan, 2, 2)
    else:
        assert np.shape(visibilities) == (row, chan)

    # Note: correctness is based on wgridder correctness


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

    faint_predict = FITSPredict()
    Nx = 100
    Ny = 100
    chan = 4
    time = 15
    ant = 24
    row = 1000
    image = jnp.ones((chan, Nx, Ny, 2, 2), dtype=jnp.float32)
    gains = jnp.ones((time, ant, chan, 2, 2), dtype=jnp.complex64)
    l0 = jnp.zeros((chan,))
    m0 = jnp.zeros((chan,))
    dl = 0.1 * jnp.ones((chan,))
    dm = 0.1 * jnp.ones((chan,))
    freqs = jnp.ones((chan,))
    freqs = tree_device_put(freqs, NamedSharding(mesh, P('chan')))
    model_data = FITSModelData(
        image=tree_device_put(image, NamedSharding(mesh, P('chan'))),
        gains=tree_device_put(gains, NamedSharding(mesh, P(None, None, 'chan'))),
        l0=tree_device_put(l0, NamedSharding(mesh, P('chan'))),
        m0=tree_device_put(m0, NamedSharding(mesh, P('chan'))),
        dl=tree_device_put(dl, NamedSharding(mesh, P('chan'))),
        dm=tree_device_put(dm, NamedSharding(mesh, P('chan'))),
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

    visibilities = faint_predict.predict(model_data=model_data, visibility_coords=visibility_coords)
    assert np.all(np.isfinite(visibilities))


@pytest.mark.parametrize("full_stokes", [True, False])
@pytest.mark.parametrize("image_has_chan", [True, False])
def test_grads_work(image_has_chan: bool, full_stokes: bool):
    faint_predict = FITSPredict()
    Nx = 100
    Ny = 100
    chan = 4
    time = 2
    ant = 24
    row = 1000
    if full_stokes:
        gain_shape = (time, ant, chan, 2, 2)
    else:
        gain_shape = (time, ant, chan)

    gains = jax.random.normal(jax.random.PRNGKey(0), gain_shape, dtype=jnp.complex64)

    if image_has_chan:
        if full_stokes:
            image_shape = (chan, Nx, Ny, 2, 2)
        else:
            image_shape = (chan, Nx, Ny)
        l0 = jnp.zeros((chan,))
        m0 = jnp.zeros((chan,))
        dl = 0.01 * jnp.ones((chan,))
        dm = 0.01 * jnp.ones((chan,))
    else:
        if full_stokes:
            image_shape = (Nx, Ny, 2, 2)
        else:
            image_shape = (Nx, Ny)
        l0 = jnp.zeros(())
        m0 = jnp.zeros(())
        dl = 0.01 * jnp.ones(())
        dm = 0.01 * jnp.ones(())
    image = jnp.ones(image_shape, dtype=jnp.float32)
    freqs = jnp.ones((chan,))
    antennas = 10e3 * jax.random.normal(jax.random.PRNGKey(0), (ant, 3))
    antennas = antennas.at[:, 2].set(antennas[:, 2] * 0.001)

    antenna_1 = jax.random.randint(jax.random.PRNGKey(0), (row,), 0, ant)
    antenna_2 = jax.random.randint(jax.random.PRNGKey(0), (row,), 0, ant)
    uvw = antennas[antenna_1] - antennas[antenna_2]
    visibility_coords = VisibilityCoords(
        uvw=uvw,
        time_obs=jnp.zeros((row,)),
        antenna_1=antenna_1,
        antenna_2=antenna_2,
        time_idx=jnp.zeros((row,), jnp.int64)
    )

    def func(gains):
        model_data = FITSModelData(
            image=image,
            gains=gains,
            l0=l0, m0=m0,
            dl=dl, dm=dm,
            freqs=freqs
        )

        vis = faint_predict.predict(model_data=model_data, visibility_coords=visibility_coords)

        return vis

    gains_grad = jax.grad(lambda gains: jnp.sum(jnp.abs(func(gains)) ** 2), argnums=0)(gains)
    # print(func(freqs, model_data, uvw))
    # print(grad)
    # gain_shape = (time, ant, chan, 2, 2)
    for t in range(time):
        for a in range(ant):
            print(f"Time: {t}, Ant: {a}")
            if full_stokes:
                print("\tXX", gains_grad[t, a, ..., 0, 0])
                print("\tXY", gains_grad[t, a, ..., 0, 1])
                print("\tYX", gains_grad[t, a, ..., 1, 0])
                print("\tYY", gains_grad[t, a, ..., 1, 1])
            else:
                print("\tXX", gains_grad[t, a, ...])
            if t == 0:
                # Ensure gradient is not zero
                assert np.all(np.abs(gains_grad[t, a]) > 1e-10)
            else:
                # Ensure gradient is zero
                assert np.all(np.abs(gains_grad[t, a]) < 1e-10)
