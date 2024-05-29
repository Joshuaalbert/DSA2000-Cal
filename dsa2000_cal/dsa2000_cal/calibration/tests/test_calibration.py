import dataclasses
import os

from dsa2000_cal.simulation.rime_model import RIMEModel

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=8"

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pytest

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.calibration.gain_prior_models import DiagonalUnconstrainedGain
from dsa2000_cal.forward_model.synthetic_sky_model import SyntheticSkyModelProducer
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0, MeasurementSet
from dsa2000_cal.simulation.simulate_visibilties import SimulateVisibilities
from dsa2000_cal.types import CalibrationSolutions


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
        gains = self.amplitude.value * np.exp(1j * self.phase.to('rad').value)  # [num_antennas, num_freqs, 2, 2]
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
        times=at.Time("2021-01-01T00:00:00", scale='utc') + 1.5 * np.arange(8) * au.s,
        freqs=au.Quantity([700, 2000], unit=au.MHz),
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=True,
        mount_types='ALT-AZ',
        system_equivalent_flux_density=array.get_system_equivalent_flux_density()
    )
    ms = MeasurementSet.create_measurement_set(str(tmp_path), meta)

    # Fill MS
    sky_model_producer = SyntheticSkyModelProducer(
        phase_tracking=ms.meta.phase_tracking,
        obs_time=ms.ref_time,
        freqs=ms.meta.freqs,
        num_bright_sources=1,
        num_faint_sources=0,
        field_of_view=4 * au.deg
    )
    sky_model = sky_model_producer.create_sky_model(include_bright=True)

    amplitude = 1. + 0.1 * np.asarray(jax.random.normal(jax.random.PRNGKey(0),
                                                        shape=(
                                                            len(ms.meta.antennas), 1, 2,
                                                            2))) * au.dimensionless_unscaled
    amplitude = np.tile(amplitude, (1, len(ms.meta.freqs), 1, 1))
    amplitude[..., 0, 1] = 0.  # Set cross-polarisation to zero
    amplitude[..., 1, 0] = 0.  # Set cross-polarisation to zero
    phase = 10 * np.asarray(jax.random.normal(jax.random.PRNGKey(1),
                                              shape=(len(ms.meta.antennas), 1, 2, 2))) * au.deg
    phase = np.tile(phase, (1, len(ms.meta.freqs), 1, 1))
    phase[..., 0, 1] = 0.  # Set cross-polarisation to zero
    phase[..., 1, 0] = 0.  # Set cross-polarisation to zero
    gain_model = MockGainModel(
        amplitude=amplitude
        ,
        phase=phase
    )

    rime_model = RIMEModel(
        sky_model=sky_model
    )
    simulate_visibilities = SimulateVisibilities(
        rime_model=rime_model,
        sky_model=sky_model,
        plot_folder='plots'
    )
    simulate_visibilities.simulate(ms, gain_model)

    # imagor = DirtyImaging(
    #     plot_folder='plots',
    #     field_of_view=2 * au.deg,
    #     seed=12345,
    #     nthreads=1
    # )
    # imagor.image('dirty', ms)

    return sky_model, ms


def test_calibration(mock_calibrator_source_models):
    sky_model, ms = mock_calibrator_source_models

    # print(fits_sources, wsclean_sources, ms)
    rime_model = RIMEModel(
        sky_model=sky_model
    )

    gain_prior_model = DiagonalUnconstrainedGain()

    calibration = Calibration(
        num_iterations=40,
        solution_interval=ms.meta.integration_time * 8,
        validity_interval=ms.meta.integration_time * 8,
        sky_model=sky_model,
        rime_model=rime_model,
        gain_prior_model=gain_prior_model,
        preapply_gain_model=None,
        inplace_subtract=True,
        verbose=True,
        plot_folder='plots',
        num_shards=2
    )
    calibration.calibrate(ms)


def test_inspect():
    solutions = CalibrationSolutions.parse_file('calibration_solutions.json')  # [source, time, ant, chan, 2, 2]
    print(solutions.gains.shape)  # (1, 2, 62, 2, 2, 2)
    # Only antenna 0 is calibrated
    print(solutions.gains[0, 0, :, :, 0, 0])  # 0.0
