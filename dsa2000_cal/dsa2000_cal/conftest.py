import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pytest

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0, MeasurementSet


@pytest.fixture(autouse=True)
def skip_if_not_64bit(request):
    if request.node.get_closest_marker("requires_64bit"):
        if not jax.config.read("jax_enable_x64"):
            # Extract test name, file, and line number
            test_name = request.node.name
            test_file = request.node.fspath
            test_line = request.node.location[1]

            # Customize the skip message
            skip_message = (
                f"Skipping test '{test_name}' in {test_file} at line {test_line} "
                "because 64-bit precision is required"
            )
            pytest.skip(skip_message)

@pytest.fixture(autouse=True)
def skip_if_not_32bit(request):
    if request.node.get_closest_marker("requires_32bit"):
        if jax.config.read("jax_enable_x64"):
            # Extract test name, file, and line number
            test_name = request.node.name
            test_file = request.node.fspath
            test_line = request.node.location[1]

            # Customize the skip message
            skip_message = (
                f"Skipping test '{test_name}' in {test_file} at line {test_line} "
                "because 32-bit precision is required"
            )
            pytest.skip(skip_message)


@pytest.fixture(scope="function")
def mock_measurement_set(tmp_path) -> MeasurementSet:
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W_small'))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    meta = MeasurementSetMetaV0(
        array_name='dsa2000W_small',
        array_location=array_location,
        phase_tracking=ac.ICRS(0 * au.deg, 0 * au.deg),
        channel_width=array.get_channel_width(),
        integration_time=au.Quantity(1, 's'),
        coherencies=['XX', 'XY', 'YX', 'YY'],
        pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
        times=at.Time("2021-01-01T00:00:00", scale='utc') + np.arange(1) * au.s,
        freqs=au.Quantity([700, 1400, 2000], unit=au.MHz),
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=True,
        mount_types='ALT-AZ'
    )
    ms = MeasurementSet.create_measurement_set(ms_folder=str(tmp_path / "test_ms"), meta=meta)

    # meta = MeasurementSetMetaV0(
    #     array_name="test_array",
    #     array_location=ac.EarthLocation.from_geodetic(0 * au.deg, 0 * au.deg, 0 * au.m),
    #     phase_tracking=ac.ICRS(0 * au.deg, 0 * au.deg),
    #     channel_width=au.Quantity(1, au.Hz),
    #     integration_time=au.Quantity(1, au.s),
    #     coherencies=['XX', 'XY', 'YX', 'YY'],
    #     pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
    #     times=at.Time.now() + np.arange(10) * au.s,
    #     freqs=au.Quantity([1, 2, 3], au.Hz),
    #     antennas=ac.EarthLocation.from_geodetic(np.arange(5) * au.deg, np.arange(5) * au.deg, np.arange(5) * au.m),
    #     antenna_names=[f"antenna_{i}" for i in range(5)],
    #     antenna_diameters=au.Quantity(np.ones(5), au.m),
    #     with_autocorr=True,
    #     mount_types='ALT-AZ',
    #     system_equivalent_flux_density=au.Quantity(1, au.Jy)
    # )
    # ms = MeasurementSet.create_measurement_set(str(tmp_path / "test_ms"), meta)
    return ms
