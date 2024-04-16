import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pytest

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0, MeasurementSet
from dsa2000_cal.source_models.fits_stokes_I_source_model import FitsStokesISourceModel
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel


@pytest.fixture(scope='function')
def mock_calibrator_source_models(tmp_path):
    fill_registries()
    times = at.Time('2021-01-01T00:00:00', scale='utc') + np.arange(2) * au.s

    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match('cyg_a')).get_wsclean_fits_files()
    # -04:00:28.608,40.43.33.595
    phase_tracking = ac.SkyCoord("-04h00m28.608s", "40d43m33.595s", frame='icrs')

    freqs = au.Quantity([65e6, 77e6], 'Hz')

    fits_sources = FitsStokesISourceModel.from_wsclean_model(
        wsclean_fits_files=wsclean_fits_files,
        time=times[0],
        phase_tracking=phase_tracking,
        freqs=freqs
    )

    # -00:36:28.234,58.50.46.396
    source_file = source_model_registry.get_instance(
        source_model_registry.get_match('cas_a')).get_wsclean_clean_component_file()
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "58d50m46.396s", frame='icrs')
    phase_tracking = ac.SkyCoord("-00h36m28.234s", "78d50m46.396s", frame='icrs')

    # -04:00:28.608,40.43.33.595
    # source_file = source_model_registry.get_instance(source_model_registry.get_match('cyg_a')).get_wsclean_source_file()
    # phase_tracking = ac.SkyCoord("-04h00m28.608s", "40d43m33.595s", frame='icrs')

    wsclean_sources = WSCleanSourceModel.from_wsclean_model(
        wsclean_clean_component_file=source_file,
        time=times[0],
        phase_tracking=phase_tracking,
        freqs=freqs
    )

    #

    meta = MeasurementSetMetaV0(
        array_name="test_array",
        array_location=ac.EarthLocation.from_geodetic(0 * au.deg, 0 * au.deg, 0 * au.m),
        phase_tracking=ac.ICRS(0 * au.deg, 0 * au.deg),
        channel_width=au.Quantity(1, au.Hz),
        integration_time=au.Quantity(1, au.s),
        coherencies=['XX', 'XY', 'YX', 'YY'],
        pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
        times=times,
        freqs=freqs,
        antennas=ac.EarthLocation.from_geodetic(np.arange(5) * au.deg, np.arange(5) * au.deg, np.arange(5) * au.m),
        antenna_names=[f"antenna_{i}" for i in range(5)],
        antenna_diameters=au.Quantity(np.ones(5), au.m),
        with_autocorr=True,
        mount_types='ALT-AZ',
        system_equivalent_flux_density=au.Quantity(1, au.Jy)
    )
    ms = MeasurementSet.create_measurement_set(tmp_path, meta)

    return fits_sources, wsclean_sources, ms


def test_calibration(mock_calibrator_source_models):
    fits_sources, wsclean_sources, ms = mock_calibrator_source_models

    # print(fits_sources, wsclean_sources, ms)

    calibration = Calibration(
        num_iterations=10,
        wsclean_source_models=[wsclean_sources],
        fits_source_models=[fits_sources],
        preapply_gain_model=None,
        inplace_subtract=True
    )
    calibration.calibrate(ms)
