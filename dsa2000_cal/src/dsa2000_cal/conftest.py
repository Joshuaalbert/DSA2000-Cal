import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pylab as plt
import pytest

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMeta, MeasurementSet


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Hook to run plt.close('all') after each test."""
    yield  # Run the actual test teardown
    plt.close('all')  # Close all plots after each test


@pytest.fixture(scope="function")
def mock_measurement_set(tmp_path) -> MeasurementSet:
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W_small'))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    meta = MeasurementSetMeta(
        array_name='dsa2000W_small',
        array_location=array_location,
        phase_center=ac.ICRS(0 * au.deg, 0 * au.deg),
        channel_width=array.get_channel_width(),
        integration_time=au.Quantity(1, 's'),
        coherencies=('XX', 'XY', 'YX', 'YY'),
        pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
        times=at.Time("2021-01-01T00:00:00", scale='utc') + np.arange(1) * au.s,
        freqs=au.Quantity([700, 1400, 2000], unit=au.MHz),
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=True,
        mount_types='ALT-AZ',
        convention='physical',
        ref_time=at.Time("2021-01-01T00:00:00", scale='utc')
    )
    ms = MeasurementSet.create_measurement_set(ms_folder=str(tmp_path / "test_ms"), meta=meta)

    return ms
