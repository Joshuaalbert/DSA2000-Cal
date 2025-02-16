import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pylab as plt
import pytest
from tomographic_kernel.frames import ENU

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_fm.measurement_sets.measurement_set import MeasurementSetMeta, MeasurementSet


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Hook to run plt.close('all') after each test."""
    yield  # Run the actual test teardown
    plt.close('all')  # Close all plots after each test


@pytest.fixture(scope="function")
def mock_measurement_set_small(tmp_path) -> MeasurementSet:
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W_small'))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    ref_time = at.Time('2021-01-01T00:00:00', scale='utc')
    obstimes = ref_time + np.arange(4) * array.get_integration_time()
    phase_center = ENU(0, 0, 1, location=array_location, obstime=ref_time).transform_to(ac.ICRS())
    phase_center = ac.ICRS(phase_center.ra, 0 * au.deg)

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
        freqs=array.get_channels()[:40],
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=True,
        mount_types='ALT-AZ',
        convention='physical'
    )
    ms = MeasurementSet.create_measurement_set(ms_folder=str(tmp_path / "test_ms"), meta=meta)

    return ms
