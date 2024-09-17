import itertools

import jax
import numpy as np
import pylab as plt
import pytest
from astropy import time as at, coordinates as ac, units as au, constants as const
from jax import numpy as jnp

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.common.ellipse_utils import Gaussian
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.types import complex_type, mp_policy
from dsa2000_cal.common.wgridder import image_to_vis
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.visibility_model.source_models.celestial.fits_source_model import FITSSourceModel, FITSPredict
from dsa2000_cal.visibility_model.source_models.celestial.gaussian_source_model import GaussianModelData, \
    GaussianPredict, GaussianSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.point_source_model import PointSourceModel


@pytest.mark.parametrize('source', ['cas_a', 'cyg_a', 'tau_a', 'vir_a'])
def test_plot_ateam_sources(source):
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

    freqs = au.Quantity(np.linspace(65e6, 77e6, 2), 'Hz')

    fits_sources = FITSSourceModel.from_wsclean_model(wsclean_fits_files=wsclean_fits_files,
                                                      phase_tracking=phase_tracking, freqs=freqs,
                                                      full_stokes=False)
    assert isinstance(fits_sources, FITSSourceModel)

    fits_sources.plot()




def build_mock_visibility_coord(rows: int, ant: int, time: int) -> VisibilityCoords:
    uvw = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (rows, 3))
    uvw = uvw.at[:, 2].mul(1e-3)
    time_obs = jnp.zeros((rows,))
    antenna_1 = jax.random.randint(jax.random.PRNGKey(42), (rows,), 0, ant)
    antenna_2 = jax.random.randint(jax.random.PRNGKey(43), (rows,), 0, ant)
    time_idx = jax.random.randint(jax.random.PRNGKey(44), (rows,), 0, time)

    visibility_coords = VisibilityCoords(
        uvw=mp_policy.cast_to_length(uvw),
        time_obs=mp_policy.cast_to_time(time_obs),
        antenna_1=mp_policy.cast_to_index(antenna_1),
        antenna_2=mp_policy.cast_to_index(antenna_2),
        time_idx=mp_policy.cast_to_index(time_idx)
    )
    return visibility_coords

@pytest.mark.parametrize('source', ['cas_a'])
@pytest.mark.parametrize('full_stokes', [True, False])
@pytest.mark.parametrize('chan', [1, 2])
@pytest.mark.parametrize('is_gains', [True, False])
def test_predict_ateam_sources(source, full_stokes, chan, is_gains: bool):
    fill_registries()

    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match(source)).get_wsclean_fits_files()
    # -04:00:28.608,40.43.33.595
    phase_tracking = ac.ICRS(ra=-4 * au.hour, dec=40 * au.deg)

    freqs = au.Quantity(np.linspace(65e6, 77e6, chan), 'Hz')

    fits_sources = FITSSourceModel.from_wsclean_model(wsclean_fits_files=wsclean_fits_files,
                                                      phase_tracking=phase_tracking, freqs=freqs,
                                                      full_stokes=full_stokes)

    @jax.jit
    def run():
        predict = FITSPredict()
        num_time = 1
        num_ant = 24
        rows = num_ant * (num_ant + 1) // 2

        if full_stokes:
            gain_shape = (num_time, num_ant, chan, 2, 2)
        else:
            gain_shape = (num_time, num_ant, chan)
        if is_gains:
            gains = jnp.ones(gain_shape, dtype=mp_policy.gain_dtype)
        else:
            gains = None

        vis_coords = build_mock_visibility_coord(
            rows=rows, ant=num_ant, time=num_time
        )
        model_data = fits_sources.get_model_data(gains=gains)
        return predict.predict(model_data=model_data, visibility_coords=vis_coords)

    block_until_ready(run())



def test_gaussian_correctness_order_1():
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

        vis_wgridder = image_to_vis(
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
                                           dtype=complex_type,
                                           order_approx=0)
        vis_gaussian_order_0 = gaussian_predict.predict(model_data=gaussian_data, visibility_coords=visibility_coords)

        gaussian_predict = GaussianPredict(convention='physical',
                                           dtype=complex_type,
                                           order_approx=1)
        vis_gaussian_order_1 = gaussian_predict.predict(model_data=gaussian_data, visibility_coords=visibility_coords)

        vis_mae_order_0.append(jnp.abs(vis_gaussian_order_0 - vis_wgridder).mean())
        vis_mae_order_1.append(jnp.abs(vis_gaussian_order_1 - vis_wgridder).mean())

        assert vis_mae_order_1[-1] < vis_mae_order_0[
            -1], f"MAE order 0: {vis_mae_order_0[-1]}, MAE order 1: {vis_mae_order_1[-1]}, l0: {l0}"

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


def test_plot_gaussian_sources():
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
        lmn_transform_params=True,
        full_stokes=False
    )

    assert isinstance(gaussian_source_model, GaussianSourceModel)
    assert gaussian_source_model.num_sources > 0

    gaussian_source_model.plot()


def test_plot_point_sources():
    fill_registries()
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    # -00:36:28.234,58.50.46.396
    source_file = source_model_registry.get_instance(
        source_model_registry.get_match('cas_a')).get_wsclean_clean_component_file()
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "58d50m46.396s", frame='icrs')
    phase_tracking = ac.ICRS(ra=-4 * au.hour, dec=40 * au.deg)

    # -04:00:28.608,40.43.33.595
    # source_file = source_model_registry.get_instance(source_model_registry.get_match('cyg_a')).get_wsclean_source_file()
    # phase_tracking = ac.SkyCoord("-04h00m28.608s", "40d43m33.595s", frame='icrs')

    freqs = au.Quantity([50e6, 80e6], 'Hz')

    point_source_model = PointSourceModel.from_wsclean_model(
        wsclean_clean_component_file=source_file,
        phase_tracking=phase_tracking,
        freqs=freqs,
        full_stokes=False
    )
    assert isinstance(point_source_model, PointSourceModel)
    assert point_source_model.num_sources > 0

    point_source_model.plot()
