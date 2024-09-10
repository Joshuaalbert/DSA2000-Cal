import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pytest

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import UnconstrainedGain
from dsa2000_cal.calibration.probabilistic_models.gains_per_facet_model import GainsPerFacet
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0, MeasurementSet, VisibilityData
from dsa2000_cal.visibility_model.facet_model import FacetModel
from dsa2000_cal.visibility_model.rime_model import RIMEModel
from dsa2000_cal.visibility_model.source_models.celestial.point_source_model import PointSourceModel


@pytest.fixture(scope='function')
def mock_calibrator_source_models(tmp_path):
    fill_registries()
    array_name = 'dsa2000W_small'
    # Load array
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
        times=at.Time("2021-01-01T00:00:00", scale='utc') + 1.5 * np.arange(5) * au.s,
        freqs=au.Quantity([700], unit=au.MHz),
        antennas=antennas,
        antenna_names=array.get_antenna_names(),
        antenna_diameters=array.get_antenna_diameter(),
        with_autocorr=True,
        mount_types='ALT-AZ'
    )
    ms = MeasurementSet.create_measurement_set(str(tmp_path), meta)
    gen = ms.create_block_generator(vis=True, weights=True, flags=True)
    gen_response = None
    while True:
        try:
            time, visibility_coords, data = gen.send(gen_response)
        except StopIteration:
            break
        gen_response = VisibilityData(
            vis=np.ones_like(data.vis) + 1e-1 * (
                    np.random.normal(size=data.vis.shape) + 1j * np.random.normal(size=data.vis.shape)),
            flags=np.zeros_like(data.flags),
            weights=np.ones_like(data.weights)
        )

    point_source_model = PointSourceModel(
        freqs=ms.meta.freqs,
        l0=[0] * au.dimensionless_unscaled,
        m0=[0] * au.dimensionless_unscaled,
        A=(
              np.ones((1, len(ms.meta.freqs), 2, 2))
              if ms.is_full_stokes() else np.ones((1, len(ms.meta.freqs)))
          ) * au.Jy
    )

    return ms, point_source_model


def test_calibration(mock_calibrator_source_models):
    ms, point_source_model = mock_calibrator_source_models

    probabilistic_models = [
        GainsPerFacet(
            rime_model=RIMEModel(
                facet_models=[
                    FacetModel(
                        point_source_model=point_source_model,
                        geodesic_model=ms.geodesic_model,
                        far_field_delay_engine=ms.far_field_delay_engine,
                        near_field_delay_engine=ms.near_field_delay_engine,
                        gain_model=ms.beam_gain_model,
                        convention=ms.meta.convention
                    )
                ]
            ),
            gain_prior_model=UnconstrainedGain()
        ),
        # HorizonRFIModel(
        #     rfi_prior_model=FullyParameterisedRFIHorizonEmitter(
        #         geodesic_model=ms.geodesic_model,
        #         beam_gain_model=ms.beam_gain_model
        #     ),
        #     rfi_predict=RFIEmitterPredict(
        #         delay_engine=ms.near_field_delay_engine,
        #         convention=ms.meta.convention
        #     )
        # )
    ]
    calibration = Calibration(
        # models to calibrate based on. Each model gets a gain direction in the flux weighted direction.
        probabilistic_models=probabilistic_models,
        num_iterations=1,
        num_approx_steps=0,
        inplace_subtract=True,
        plot_folder='plots',
        solution_folder='solutions',
        validity_interval=None,
        solution_interval=None,
        verbose=True
    )
    subtracted_ms = calibration.calibrate(ms)
