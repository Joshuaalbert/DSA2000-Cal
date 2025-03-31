import time

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.ellipse_utils import Gaussian
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_common.common.wgridder import vis_to_image
from dsa2000_common.delay_models.base_far_field_delay_engine import build_far_field_delay_engine
from dsa2000_common.delay_models.base_near_field_delay_engine import build_near_field_delay_engine
from dsa2000_common.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model
from dsa2000_fm.imaging.base_imagor import fit_beam, evaluate_beam, divide_out_beam
from dsa2000_fm.measurement_sets.measurement_set import MeasurementSetMeta, MeasurementSet


def build_mock_calibrator_source_models(tmp_path, coherencies):
    fill_registries()
    array_name = 'lwa_mock'
    # Load array
    # array = array_registry.get_instance(array_registry.get_match('dsa2000W_small'))
    array = array_registry.get_instance(array_registry.get_match(array_name))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    # -00:36:29.015,58.45.50.398
    phase_center = ac.SkyCoord("-00h36m29.015s", "58d45m50.398s", frame='icrs')
    phase_center = ac.ICRS(phase_center.ra, phase_center.dec)

    meta = MeasurementSetMeta(
        array_name=array_name,
        array_location=array_location,
        phase_center=phase_center,
        channel_width=array.get_channel_width(),
        integration_time=au.Quantity(1.5, 's'),
        coherencies=coherencies,
        pointings=phase_center,
        times=at.Time("2021-01-01T00:00:00", scale='utc') + 1.5 * np.arange(1) * au.s,
        freqs=au.Quantity([700], unit=au.MHz),
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=True,
        mount_types='ALT-AZ'
    )
    ms = MeasurementSet.create_measurement_set(str(tmp_path), meta)

    return ms


@pytest.mark.parametrize("centre_offset", [0.0, 0.1, 0.2])
@pytest.mark.parametrize("coherencies", [['XX', 'XY', 'YX', 'YY'], ['I', 'Q', 'U', 'V'], ['I']])
def test_evaluate_beam(tmp_path, coherencies, centre_offset: float):
    ms = build_mock_calibrator_source_models(tmp_path, coherencies)
    t0 = time.time()
    beam_gain_model = build_beam_gain_model(array_name=ms.meta.array_name, times=ms.meta.times,
                                            full_stokes=ms.is_full_stokes())
    print(f"Built in {time.time() - t0} seconds.")
    freqs = quantity_to_jnp(ms.meta.freqs)
    times = ms.time_to_jnp(ms.meta.times)
    geodesic_model = ms.geodesic_model
    num_l = 100
    num_m = 100
    dl = 0.001
    dm = 0.001
    center_l = center_m = centre_offset
    beam = evaluate_beam(freqs, times, beam_gain_model, geodesic_model, num_l, num_m, dl, dm, center_l, center_m)

    assert np.all(np.isfinite(beam))
    avg_beam = jnp.mean(beam, axis=(2, 3))

    if len(coherencies) == 1:
        assert beam.shape == (num_l, num_m, len(times), len(freqs))
    else:
        assert beam.shape == (num_l, num_m, len(times), len(freqs), 2, 2)

    image = np.ones((num_l, num_m, len(coherencies)))
    # image[::10, ::10, 0] = 1.
    # image[::10, ::10, 3] = 1.

    pb_cor_image = divide_out_beam(image, avg_beam)
    assert pb_cor_image.shape == image.shape
    assert np.all(np.isfinite(pb_cor_image))
    import pylab as plt

    if len(coherencies) == 4:
        avg_beam = avg_beam[..., 0, 0]

    plt.imshow(
        np.abs(avg_beam).T,
        origin='lower',
        aspect='auto',
        extent=(-0.5 * num_l * dl, 0.5 * num_l * dl, -0.5 * num_m * dm, 0.5 * num_m * dm)
    )
    plt.colorbar()
    plt.title('beam amplitude')
    plt.xlabel('l')
    plt.ylabel('m')
    plt.show()

    plt.imshow(
        np.abs(pb_cor_image[..., 0]).T,
        origin='lower',
        aspect='auto',
        extent=(-0.5 * num_l * dl, 0.5 * num_l * dl, -0.5 * num_m * dm, 0.5 * num_m * dm)
    )
    plt.colorbar()
    plt.title('amplitude')
    plt.xlabel('l')
    plt.ylabel('m')
    plt.show()

    plt.imshow(
        np.angle(pb_cor_image[..., 0]).T,
        origin='lower',
        aspect='auto',
        extent=(-0.5 * num_l * dl, 0.5 * num_l * dl, -0.5 * num_m * dm, 0.5 * num_m * dm)
    )
    plt.colorbar()
    plt.title('phase')
    plt.xlabel('l')
    plt.ylabel('m')
    plt.show()


def test_fit_beam():
    """
    Test the fit_ellipsoid function by verifying it retrieves the correct FWHM
    for both the major and minor axes of a synthetic ellipsoid image.
    """
    dl = dm = 0.01
    n = 256
    lvec = (-0.5 * n + jnp.arange(n)) * dl
    mvec = (-0.5 * n + jnp.arange(n)) * dm
    L, M = jnp.meshgrid(lvec, mvec, indexing='ij')
    lm = jnp.stack([L, M], axis=-1).reshape((-1, 2))

    true_minor_fwhm = 5 * dl
    true_major_fwhm = 10 * dl
    true_posang = 1.

    g = Gaussian(x0=jnp.zeros(2), minor_fwhm=true_minor_fwhm, major_fwhm=true_major_fwhm, pos_angle=true_posang,
                 total_flux=Gaussian.total_flux_from_peak(1, major_fwhm=true_major_fwhm, minor_fwhm=true_minor_fwhm))

    psf = jax.vmap(g.compute_flux_density)(lm)
    psf = jnp.reshape(psf, (n, n))
    major, minor, pos_angle = fit_beam(psf, dl, dm)
    print(major, minor, pos_angle)

    assert jnp.allclose(major, true_major_fwhm, atol=0.01)
    assert jnp.allclose(minor, true_minor_fwhm, atol=0.01)
    assert jnp.allclose(pos_angle, true_posang, atol=0.01)


def test_fit_beam_explore():
    """
    Test the fit_ellipsoid function by verifying it retrieves the correct FWHM
    for both the major and minor axes of a synthetic ellipsoid image.
    """
    x = []
    y = []
    for pixsize in np.linspace(0.25, 5, 100):
        dl = dm = quantity_to_jnp(pixsize * au.arcsec, 'rad')
        n = 256
        lvec = (-0.5 * n + jnp.arange(n)) * dl
        mvec = (-0.5 * n + jnp.arange(n)) * dm
        L, M = jnp.meshgrid(lvec, mvec, indexing='ij')
        lm = jnp.stack([L, M], axis=-1).reshape((-1, 2))

        true_minor_fwhm = quantity_to_jnp(3.3 * au.arcsec, 'rad') / 2
        true_major_fwhm = quantity_to_jnp(3.3 * au.arcsec, 'rad') / 2
        true_posang = 0.

        g = Gaussian(x0=jnp.zeros(2), minor_fwhm=true_minor_fwhm, major_fwhm=true_major_fwhm, pos_angle=true_posang,
                     total_flux=Gaussian.total_flux_from_peak(1, major_fwhm=true_major_fwhm,
                                                              minor_fwhm=true_minor_fwhm))

        psf = jax.vmap(g.compute_flux_density)(lm)
        psf = jnp.reshape(psf, (n, n))
        major, minor, pos_angle = fit_beam(psf, dl, dm)
        error = np.minimum(np.sqrt(((major - true_major_fwhm) ** 2 + (minor - true_minor_fwhm) ** 2) / dl ** 2),
                           np.sqrt(((major - true_minor_fwhm) ** 2 + (minor - true_major_fwhm) ** 2) / dl ** 2))
        print(f"dl={dl}, dm={dm}, major={major}, minor={minor}, pos_angle={pos_angle}, error={error}")

        x.append(dl * 180 / np.pi * 3600)
        y.append(error)
    import pylab as plt
    plt.plot(x, y)
    plt.xlabel('pixel size (arcsec)')
    plt.ylabel('RMS (pixels)')
    plt.title("Fit error in pixels vs pixel size")
    plt.show()


def test_fit_beam_explore_real_PSF():
    """
    Test the fit_ellipsoid function by verifying it retrieves the correct FWHM
    for both the major and minor axes of a synthetic ellipsoid image.
    """
    num_time = 1

    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000_optimal_v1'))
    array_location = array.get_array_location()
    ref_time = at.Time('2021-01-01T00:00:00', scale='utc')
    obstimes = ref_time + array.get_integration_time() * np.arange(num_time)
    phase_center = ENU(0, 0, 1, location=array_location, obstime=ref_time).transform_to(ac.ICRS())
    obsfreqs = array.get_channels()[:1]
    freqs = quantity_to_jnp(obsfreqs, 'Hz')
    times = time_to_jnp(obstimes, ref_time)

    pointing = phase_center
    antennas = array.get_antennas()

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
        freqs=freqs,
        times=times
    )

    uvw = visibility_coords.uvw.reshape((-1, 3))
    vis = jnp.ones(uvw.shape[:1] + np.shape(freqs), jnp.complex64)

    dl = dm = 0.1 * au.arcsec
    psf = vis_to_image(
        uvw=uvw, freqs=freqs, vis=vis,
        pixsize_l=quantity_to_jnp(dl, 'rad'),
        pixsize_m=quantity_to_jnp(dm, 'rad'),
        center_l=0., center_m=0., npix_l=256, npix_m=256
    )
    true_major_fwhm, true_minor_fwhm, true_pos_angle = fit_beam(psf, dl, dm)

    x = []
    y = []
    for pixsize in np.linspace(0.25, 5, 100):
        dl = dm = pixsize * au.arcsec
        psf = vis_to_image(
            uvw=uvw, freqs=freqs, vis=vis,
            pixsize_l=quantity_to_jnp(dl, 'rad'),
            pixsize_m=quantity_to_jnp(dm, 'rad'),
            center_l=0., center_m=0., npix_l=256, npix_m=256
        )
        major, minor, pos_angle = fit_beam(psf, dl, dm)
        error = np.minimum(np.sqrt(((major - true_major_fwhm) ** 2 + (minor - true_minor_fwhm) ** 2) / dl ** 2),
                           np.sqrt(((major - true_minor_fwhm) ** 2 + (minor - true_major_fwhm) ** 2) / dl ** 2))
        print(f"dl={dl}, dm={dm}, major={major}, minor={minor}, pos_angle={pos_angle}, error={error}")

        x.append(dl * 180 / np.pi * 3600)
        y.append(error)
    import pylab as plt
    plt.plot(x, y)
    plt.xlabel('pixel size (arcsec)')
    plt.ylabel('RMS (pixels)')
    plt.title("Fit error in pixels vs pixel size")
    plt.show()
