import itertools
import time
from functools import partial

import astropy.constants as const
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pylab as plt
import pytest
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.common.ellipse_utils import Gaussian
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.wgridder import dirty2vis
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

# def unwrapped_wkernel(l, w):
#     n = jnp.sqrt(1 - l ** 2)
#     return jnp.exp(-2j * jnp.pi * w * (n - 1.) / 0.5) / n

def test_w_abs_approx():
    def approx(l):
        return 1 + l**2/2 + (3* l**4)/8 + (5* l**6)/16
    def f(l):
        return 1/jnp.sqrt(1 - l**2)

    lvec = jnp.linspace(0., 0.999, 1000)

    import pylab as plt

    plt.plot(lvec, jnp.log10(f(lvec) - approx(lvec)))
    plt.axvline(jnp.interp(0.001, f(lvec) - approx(lvec), lvec), color='r', label='0.1% error')
    plt.title("Approximation error 6th order")
    plt.xlabel('l')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()

    def approx(l):
        return 1 + l**2/2 + (3* l**4)/8 + (5 *l**6)/16 + (35* l**8)/128 + (63 *l**10)/256
    def f(l):
        return 1/jnp.sqrt(1 - l**2)

    lvec = jnp.linspace(0., 0.999, 1000)

    import pylab as plt

    plt.plot(lvec, jnp.log10(f(lvec) - approx(lvec)))
    plt.axvline(jnp.interp(0.001, f(lvec) - approx(lvec), lvec), color='r', label='0.1% error')
    plt.title("Approximation error 10th order")
    plt.xlabel('l')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()

def test_w_angle_approximation():
    def w_angle(l, w):
        n = jnp.sqrt(1 - l ** 2)
        return -2 * jnp.pi * w * (n - 1.)

    def approx(l, w):
        # l ^ 2 \pi  w + (l ^ 4 \pi w) / 4 + (l ^ 6 \pi w) / 8 + (5 l ^ 8 \pi w) / 64 + (7 l ^ {10} \pi w) / 128
        return l * jnp.pi * w + (l ** 3 * jnp.pi * w) / 4 + (l ** 5 * jnp.pi * w) / 8 + (5 * l ** 7 * jnp.pi * w) / 64 + (
                7 * l ** 9 * jnp.pi * w) / 128

    lvec = jnp.linspace(0., 0.999, 1000)
    w = 1e3

    import pylab as plt

    plt.plot(lvec, jnp.abs(w_angle(lvec, w) - approx(lvec, w)))

    plt.title("Approximation error 10th order")
    plt.xlabel('l')
    plt.ylabel('log10(error)')
    plt.show()

def test_wkernel_variation():
    freq = 70e6 * au.Hz
    wavelength = quantity_to_jnp(const.c / freq)
    wvec = jnp.linspace(0., 10e3, 10000)
    lvec = jnp.linspace(0, 0.999, 1000)

    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def wkernel(l, w):
        n = jnp.sqrt(1 - l ** 2)
        return jnp.exp(-2j * jnp.pi * w * (n - 1.) / wavelength) / n

    wterm = wkernel(lvec, wvec)  # [nl, nw]

    import pylab as plt

    wterm_angle = jnp.angle(wterm) # [nl, nw]

    wterm_angle_unwrapped = jnp.unwrap(wterm_angle, axis=0)
    print(jnp.sum(jnp.isnan(wterm_angle_unwrapped)))
    print(wterm_angle_unwrapped)

    np.testing.assert_allclose(jnp.angle(jnp.exp(1j*wterm_angle_unwrapped)), wterm_angle, atol=1e-6)

    diff_unwrapped = jnp.diff(wterm_angle_unwrapped, axis=0)

    plt.imshow(diff_unwrapped.T,
               origin='lower',
               extent=(lvec[0], lvec[-1], wvec[0], wvec[-1]),
               interpolation='nearest',
               aspect='auto',
               cmap='hsv')
    plt.colorbar()
    plt.xlabel('l [proj. rad]')
    plt.ylabel('w [m]')
    plt.show()

    plt.imshow(wterm_angle.T,
               origin='lower',
               extent=(lvec[0], lvec[-1], wvec[0], wvec[-1]),
               interpolation='nearest',
               aspect='auto',
               cmap='hsv')
    plt.colorbar()
    plt.xlabel('l [proj. rad]')
    plt.ylabel('w [m]')
    plt.title(r"${\rm{Arg}W(l,m)}$")
    plt.show()


    plt.imshow(wterm_angle_unwrapped.T,
               origin='lower',
               extent=(lvec[0], lvec[-1], wvec[0], wvec[-1]),
               interpolation='nearest',
               aspect='auto',
               cmap='jet')
    plt.colorbar()
    plt.xlabel('l [proj. rad]')
    plt.ylabel('w [m]')
    plt.title(r"Unwrapped ${\rm{Arg}W(l,m)}$")
    plt.show()

    plt.imshow(jnp.log(jnp.abs(wterm).T),
               origin='lower',
               extent=(lvec[0], lvec[-1], wvec[0], wvec[-1]),
                interpolation='nearest',
                aspect='auto',
                cmap='inferno')
    plt.colorbar()
    plt.xlabel('l [proj. rad]')
    plt.ylabel('w [m]')
    plt.show()


def test_correctness_order_1():
    major_fwhm_arcsec = 4. * 60
    minor_fwhm_arcsec = 2. * 60
    pos_angle_deg = 90.
    total_flux = 1.

    freq = 70e6 * au.Hz
    wavelength = quantity_to_jnp(const.c / freq)
    num_ant = 128
    num_time = 1

    freqs = quantity_to_jnp(freq)[None]

    max_baseline = 20e3

    antennas = max_baseline * jax.random.normal(jax.random.PRNGKey(42), (num_ant, 3))
    # With autocorr
    antenna_1, antenna_2 = jnp.asarray(
        list(itertools.combinations_with_replacement(range(num_ant), 2))).T

    num_rows = len(antenna_1)

    uvw = antennas[antenna_2] - antennas[antenna_1]
    uvw = uvw.at[:, 2].mul(1e-3)

    times = jnp.arange(num_time) * 1.5
    time_idx = jnp.zeros((num_rows,), jnp.int64)
    time_obs = times[time_idx]

    visibility_coords = VisibilityCoords(
        uvw=uvw,
        time_obs=time_obs,
        antenna_1=antenna_1,
        antenna_2=antenna_2,
        time_idx=time_idx
    )

    l0_array = []
    vis_mae_order_0 = []
    vis_mae_order_1 = []
    for l0 in jnp.linspace(-0.99, 0.99, 20):
        l0_array.append(l0)
        m0 = 0.

        # Use wgridder as comparison
        gaussian = Gaussian(
            x0=jnp.asarray([l0, m0]),
            major_fwhm=jnp.asarray(major_fwhm_arcsec / 3600. * np.pi / 180.),
            minor_fwhm=jnp.asarray(minor_fwhm_arcsec / 3600. * np.pi / 180.),
            pos_angle=jnp.asarray(pos_angle_deg / 180. * np.pi),
            total_flux=jnp.asarray(total_flux)
        )

        n = 2096
        pix_size = (wavelength / max_baseline) / 7.
        lvec = pix_size * (-n / 2 + jnp.arange(n)) + l0
        mvec = pix_size * (-n / 2 + jnp.arange(n)) + m0
        L, M = jnp.meshgrid(lvec, mvec, indexing='ij')
        X = jnp.stack([L.flatten(), M.flatten()], axis=-1)
        flux_density = jax.vmap(gaussian.compute_flux_density)(X).reshape(L.shape)
        flux = flux_density * pix_size ** 2

        # plt.imshow(flux.T, origin='lower',
        #            extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]),
        #            cmap='inferno',
        #            interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        vis_wgridder = dirty2vis(
            uvw=uvw,
            freqs=freqs,
            dirty=flux,
            pixsize_l=pix_size,
            pixsize_m=pix_size,
            center_l=l0,
            center_m=m0,
            epsilon=1e-8
        )

        gaussian_data = GaussianModelData(
            freqs=freqs,
            image=gaussian.total_flux[None, None],
            gains=None,
            lmn=jnp.asarray([[l0, m0, jnp.sqrt(1. - l0 ** 2 - m0 ** 2)]]),
            ellipse_params=jnp.asarray([[gaussian.major_fwhm,
                                         gaussian.minor_fwhm,
                                         gaussian.pos_angle]])
        )

        gaussian_predict = GaussianPredict(convention='physical',
                                           dtype=jnp.complex128,
                                           order_approx=0)
        vis_gaussian_order_0 = gaussian_predict.predict(model_data=gaussian_data, visibility_coords=visibility_coords)

        gaussian_predict = GaussianPredict(convention='physical',
                                           dtype=jnp.complex128,
                                           order_approx=1)
        vis_gaussian_order_1 = gaussian_predict.predict(model_data=gaussian_data, visibility_coords=visibility_coords)

        vis_mae_order_0.append(jnp.abs(vis_gaussian_order_0 - vis_wgridder).mean())
        vis_mae_order_1.append(jnp.abs(vis_gaussian_order_1 - vis_wgridder).mean())

        assert (
            vis_mae_order_1[-1] < vis_mae_order_0[-1],
            f"MAE order 0: {vis_mae_order_0[-1]}, MAE order 1: {vis_mae_order_1[-1]}, l0: {l0}"
        )

    plt.plot(l0_array, vis_mae_order_0, label='order 0')
    plt.plot(l0_array, vis_mae_order_1, label='order 1')
    plt.xlabel("l")
    plt.ylabel("MAE [Jy]")
    plt.title(f"Visibility MAE as gaussian center shifted in l: {freq.to('MHz')}.")
    plt.legend()
    plt.show()

    #
    # sc = plt.scatter(uvw[:, 0], uvw[:, 1], c=vis_gaussian.real, alpha=0.5, marker='.')
    # plt.colorbar(sc)
    # plt.title("Real part of Gaussian visibilities")
    # plt.show()
    #
    # sc = plt.scatter(uvw[:, 0], uvw[:, 1], c=vis_wgridder.real, alpha=0.5, marker='.')
    # plt.colorbar(sc)
    # plt.title("Real part of wgridder visibilities")
    # plt.show()
    #
    # sc = plt.scatter(uvw[:, 0], uvw[:, 1], c=vis_gaussian.real - vis_wgridder.real, alpha=0.5, marker='.')
    # plt.colorbar(sc)
    # plt.title("Real part of difference")
    # plt.show()
