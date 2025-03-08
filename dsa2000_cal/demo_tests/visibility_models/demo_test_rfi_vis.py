import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pytest
from jax import numpy as jnp
from dsa2000_common.common.enu_frame import ENU

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.fits_utils import ImageModel, save_image_to_fits
from dsa2000_common.common.wgridder import vis_to_image_np
from dsa2000_common.common.quantity_utils import quantity_to_jnp, time_to_jnp, quantity_to_np
from dsa2000_common.visibility_model.source_models.rfi.base_rfi_emitter_source_model import RFIEmitterSourceModel
from dsa2000_common.visibility_model.source_models.rfi.parametric_rfi_emitter import ParametricDelayACF
from dsa2000_fm.imaging.utils import get_array_image_parameters
from dsa2000_fm.measurement_sets.measurement_set import MeasurementSetMeta, MeasurementSet


@pytest.fixture(scope="function")
def mock_measurement_set_small(tmp_path) -> MeasurementSet:
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W_small'))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    ref_time = at.Time('2021-01-01T00:00:00', scale='utc')
    obstimes = ref_time + np.arange(4) * array.get_integration_time()
    phase_center = ENU(1, 0, 1, location=array_location, obstime=ref_time).transform_to(ac.ICRS())
    # phase_center = ac.ICRS(phase_center.ra, 0 * au.deg)

    meta = MeasurementSetMeta(
        array_name='dsa2000W_small',
        array_location=array_location,
        phase_center=phase_center,
        channel_width=array.get_channel_width(),
        integration_time=array.get_integration_time(),
        coherencies=('I',),
        pointings=phase_center,
        times=obstimes,
        ref_time=ref_time,
        freqs=array.get_channels(),
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=True,
        mount_types='ALT-AZ',
        convention='physical'
    )
    ms = MeasurementSet.create_measurement_set(ms_folder=str(tmp_path / "test_ms"), meta=meta)

    return ms


@pytest.mark.parametrize(
    "mu_Hz, fwhp_Hz, spectral_power_Jy_1km, channel_Hz, channel_width_Hz",
    [
        (700e6, 1e6, 10, 700e6, 130e3),
        (700e6, 100e3, 10, 700e6, 130e3),
        (699e6, 1e6, 10, 700e6, 130e3),
        (699e6, 5e6, 10, 700e6, 130e3),
    ]
)
def test_parametric_delay_acf(mu_Hz, fwhp_Hz, spectral_power_Jy_1km, channel_Hz, channel_width_Hz,
                              mock_measurement_set_small: MeasurementSet):
    mu = jnp.asarray([mu_Hz])
    fwhp = jnp.asarray([fwhp_Hz])
    spectral_power = quantity_to_jnp([spectral_power_Jy_1km] * au.Jy * (1 * au.km) ** 2 / (channel_width_Hz * au.Hz),
                                     'Jy*m^2/Hz')
    delay_acf = ParametricDelayACF(
        mu, fwhp,
        spectral_power=spectral_power,
        channel_width=channel_width_Hz,
        resolution=128
    )
    taus = jnp.linspace(-1e-4, 1e-4, 1000)

    acf_vals = jax.vmap(lambda tau: delay_acf.eval(700e6, tau))(taus)
    import pylab as plt

    plt.plot(taus * 1e6, jnp.abs(acf_vals)[:, 0], label=f'mu={mu_Hz / 1e6}MHz,sigma={fwhp_Hz / 1e6}MHz')
    plt.legend()
    plt.title(rf'Parametric Delay ACF, Channel {channel_Hz / 1e6} MHz $\pm$ {channel_width_Hz / 1e6} MHz')
    plt.xlabel(r'Delay ($\mu$s)')
    plt.ylabel('ACF (Jy km^2)')
    plt.show()

    position_enu = jnp.asarray([1e3, 0, 1e3]).reshape((1, 3))

    rfi_model = RFIEmitterSourceModel(
        position_enu=position_enu,
        delay_acf=delay_acf
    )
    rfi_model.plot()

    # predict and image
    visibility_coords = mock_measurement_set_small.far_field_delay_engine.compute_visibility_coords(
        freqs=quantity_to_jnp(mock_measurement_set_small.meta.freqs, 'Hz'),
        times=time_to_jnp(mock_measurement_set_small.meta.times, mock_measurement_set_small.meta.ref_time),
        with_autocorr=mock_measurement_set_small.meta.with_autocorr,
        convention=mock_measurement_set_small.meta.convention
    )

    visibilities = rfi_model.predict(
        visibility_coords=visibility_coords,
        gain_model=None,
        near_field_delay_engine=mock_measurement_set_small.near_field_delay_engine,
        far_field_delay_engine=mock_measurement_set_small.far_field_delay_engine,
        geodesic_model=mock_measurement_set_small.geodesic_model
    )

    num_pixel, dl, dm, l0, m0 = get_array_image_parameters(
        mock_measurement_set_small.meta.array_name, oversample_factor=5, field_of_view=1 * au.deg
    )

    uvw = np.array(visibility_coords.uvw.reshape((-1, 3)), order='C')
    visibilities = np.array(visibilities.reshape(uvw.shape[:1] + np.shape(visibilities)[2:]), order='F')

    image = vis_to_image_np(
        uvw=uvw,
        freqs=np.array(visibility_coords.freqs),
        vis=visibilities,
        pixsize_m=quantity_to_np(dm),
        pixsize_l=quantity_to_np(dl),
        center_l=quantity_to_np(l0),
        center_m=quantity_to_np(m0),
        npix_l=num_pixel,
        npix_m=num_pixel,
        scale_by_n=True,
        normalise=True,
        num_threads=12
    )

    import pylab as plt
    plt.imshow(image.T, origin='lower')
    plt.colorbar()
    plt.show()

    image_model = ImageModel(
        phase_center=mock_measurement_set_small.meta.phase_center,
        obs_time=mock_measurement_set_small.meta.ref_time,
        dl=dl,
        dm=dm,
        freqs=np.mean(mock_measurement_set_small.meta.freqs)[None],
        bandwidth=channel_width_Hz * au.Hz,
        coherencies=('I',),
        beam_major=np.asarray(3) * au.arcsec,
        beam_minor=np.asarray(3) * au.arcsec,
        beam_pa=np.asarray(0) * au.rad,
        unit='JY/PIXEL',
        object_name=f'RFI_TEST',
        image=image[:, :, None, None] * au.Jy  # [num_l, num_m, 1, 1]
    )
    save_image_to_fits(f"rfi_imaged.fits", image_model=image_model, overwrite=True)
