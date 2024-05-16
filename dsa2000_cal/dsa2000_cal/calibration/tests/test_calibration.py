import dataclasses

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pytest

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry, array_registry
from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.forward_model.synthetic_sky_model import SyntheticSkyModelProducer
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0, MeasurementSet
from dsa2000_cal.simulation.simulate_visibilties import SimulateVisibilities
from dsa2000_cal.source_models.fits_stokes_I_source_model import FitsStokesISourceModel
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel


@dataclasses.dataclass(eq=False)
class MockGainModel(GainModel):
    """
    Mock gain model.
    """
    phase: au.Quantity  # [num_antennas, num_freqs, 2, 2]
    amplitude: au.Quantity  # [num_antennas, num_freqs, 2, 2]

    def __post_init__(self):
        if self.phase.shape != self.amplitude.shape:
            raise ValueError("Phase and amplitude must have the same shape.")

        if not self.phase.unit.is_equivalent(au.rad):
            raise ValueError("Phase must be in radians.")
        if not self.amplitude.unit.is_equivalent(au.dimensionless_unscaled):
            raise ValueError("Amplitude must be dimensionless.")

    def compute_gain(self, freqs: au.Quantity, sources: ac.ICRS, phase_tracking: ac.ICRS,
                     array_location: ac.EarthLocation, time: at.Time,
                     **kwargs):
        shape = sources.shape
        sources = sources.reshape((-1,))
        gains = self.amplitude * np.exp(1j * self.phase)  # [num_antennas, num_freqs, 2, 2]
        gains = np.tile(gains[None], (len(sources), 1, 1, 1, 1))  # [num_sources, num_antennas, num_freqs, 2, 2]
        gains = gains.reshape(shape + gains.shape[1:])  # (source_shape) + [num_antennas, num_freqs, 2, 2]
        return gains


@pytest.fixture(scope='function')
def mock_calibrator_source_models(tmp_path):
    fill_registries()

    # Load array
    array = array_registry.get_instance(array_registry.get_match('dsa2000W_small'))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    # -00:36:29.015,58.45.50.398
    phase_tracking = ac.SkyCoord("-00h36m29.015s", "58d45m50.398s", frame='icrs')
    phase_tracking = ac.ICRS(phase_tracking.ra, phase_tracking.dec)

    meta = MeasurementSetMetaV0(
        array_name='dsa2000W_small',
        array_location=array_location,
        phase_tracking=phase_tracking,
        channel_width=array.get_channel_width(),
        integration_time=au.Quantity(1.5, 's'),
        coherencies=['XX', 'XY', 'YX', 'YY'],
        pointings=phase_tracking,
        times=at.Time("2021-01-01T00:00:00", scale='utc') + 1.5 * np.arange(2) * au.s,
        freqs=au.Quantity([50, 70], unit=au.MHz),
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=True,
        mount_types='ALT-AZ'
    )
    ms = MeasurementSet.create_measurement_set(str(tmp_path), meta)

    # Fill MS
    sky_model_producer = SyntheticSkyModelProducer(
        phase_tracking=ms.meta.phase_tracking,
        obs_time=ms.ref_time,
        freqs=ms.meta.freqs,
        num_bright_sources=1,
        num_faint_sources=0,
        field_of_view=2 * au.deg
    )
    sky_model = sky_model_producer.create_sky_model()
    sky_model_calibrators = sky_model_producer.create_sky_model(include_faint=False)

    sky_model_source_models = sky_model.to_wsclean_source_models()
    sky_model_calibrators_source_models = sky_model_calibrators.to_wsclean_source_models()

    gain_model = MockGainModel(

    )

    simulate_visibilities = SimulateVisibilities(
        wsclean_source_models=sky_model_source_models,
        fits_source_models=[]
    )

    # Load source models
    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match('mock')).get_wsclean_fits_files()

    fits_sources = FitsStokesISourceModel.from_wsclean_model(
        wsclean_fits_files=wsclean_fits_files,
        time=ms.ref_time,
        phase_tracking=ms.meta.phase_tracking,
        freqs=ms.meta.freqs
    )

    source_file = source_model_registry.get_instance(
        source_model_registry.get_match('mock')).get_wsclean_clean_component_file()

    wsclean_sources = WSCleanSourceModel.from_wsclean_model(
        wsclean_clean_component_file=source_file,
        time=ms.ref_time,
        phase_tracking=ms.meta.phase_tracking,
        freqs=ms.meta.freqs
    )

    return fits_sources, wsclean_sources, ms


def test_calibration(mock_calibrator_source_models):
    fits_sources, wsclean_sources, ms = mock_calibrator_source_models

    # print(fits_sources, wsclean_sources, ms)

    calibration = Calibration(
        num_iterations=10,
        wsclean_source_models=[wsclean_sources],
        fits_source_models=[fits_sources],
        preapply_gain_model=None,
        inplace_subtract=True,
        verbose=True
    )
    calibration.calibrate(ms)
