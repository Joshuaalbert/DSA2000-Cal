import time

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.numpy as jnp
import numpy as np
import pytest

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_cal.imaging.imagor import evaluate_beam, divide_out_beam
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0, MeasurementSet


@pytest.fixture(scope='function')
def mock_calibrator_source_models(tmp_path):
    fill_registries()
    array_name = 'lwa_mock'
    # Load array
    # array = array_registry.get_instance(array_registry.get_match('dsa2000W_small'))
    array = array_registry.get_instance(array_registry.get_match(array_name))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    # -00:36:29.015,58.45.50.398
    phase_tracking = ac.SkyCoord("-00h36m29.015s", "58d45m50.398s", frame='icrs')
    phase_tracking = ac.ICRS(phase_tracking.ra, phase_tracking.dec)

    meta = MeasurementSetMetaV0(
        array_name=array_name,
        array_location=array_location,
        phase_tracking=phase_tracking,
        channel_width=array.get_channel_width(),
        integration_time=au.Quantity(1.5, 's'),
        coherencies=['XX', 'XY', 'YX', 'YY'],
        pointings=phase_tracking,
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


def test_evaluate_beam(mock_calibrator_source_models: MeasurementSet):
    ms = mock_calibrator_source_models
    t0 = time.time()
    beam_gain_model = build_beam_gain_model(
        array_name=ms.meta.array_name,
        full_stokes=ms.is_full_stokes(),
        model_times=ms.meta.times
    )
    print(f"Built in {time.time() - t0} seconds.")
    freqs = quantity_to_jnp(ms.meta.freqs)
    times = ms.time_to_jnp(ms.meta.times)
    geodesic_model = ms.geodesic_model
    num_l = 100
    num_m = 100
    dl = 0.001
    dm = 0.001
    center_l = 0.
    center_m = 0.
    beam = evaluate_beam(freqs, times, beam_gain_model, geodesic_model, num_l, num_m, dl, dm, center_l, center_m)
    assert beam.shape == (num_l, num_m, len(times), len(freqs), 2, 2)
    assert np.all(np.isfinite(beam))

    avg_beam = jnp.mean(beam, axis=(2, 3))

    image = np.ones((num_l, num_m, 4))
    # image[::10, ::10, 0] = 1.
    # image[::10, ::10, 3] = 1.

    pb_cor_image = divide_out_beam(image, avg_beam)
    assert pb_cor_image.shape == image.shape
    assert np.all(np.isfinite(pb_cor_image))
    import pylab as plt

    plt.imshow(
        np.abs(avg_beam[..., 0, 0]).T,
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
