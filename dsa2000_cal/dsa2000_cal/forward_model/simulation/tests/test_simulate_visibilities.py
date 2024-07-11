import time as time_mod

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.numpy as jnp
import numpy as np
import pytest

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry, array_registry
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.visibility_model.sky_model import SkyModel
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0, MeasurementSet
from dsa2000_cal.forward_model.simulation.simulate_visibilties import SimulateVisibilities
from dsa2000_cal.visibility_model.source_models.celestial.fits_source.fits_source_model import FITSSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.wsclean_component_stokes_I_source_model import WSCleanSourceModel


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

    # Load source models
    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match('mock')).get_wsclean_fits_files()

    fits_sources = FITSSourceModel.from_wsclean_model(wsclean_fits_files=wsclean_fits_files,
                                                      phase_tracking=ms.meta.pointing, freqs=ms.meta.freqs)

    source_file = source_model_registry.get_instance(
        source_model_registry.get_match('mock')).get_wsclean_clean_component_file()

    wsclean_sources = WSCleanSourceModel.from_wsclean_model(
        wsclean_clean_component_file=source_file,
        time=ms.ref_time,
        phase_tracking=ms.meta.pointing,
        freqs=ms.meta.freqs
    )

    return fits_sources, wsclean_sources, ms


def test_predict_model_visibilties(mock_calibrator_source_models):
    fits_sources, wsclean_sources, ms = mock_calibrator_source_models

    # print(fits_sources, wsclean_sources, ms)

    simulation = SimulateVisibilities(
        sky_model=SkyModel(
            component_source_models=[wsclean_sources],
            fits_source_models=[fits_sources]
        ),
        verbose=True,
        plot_folder='plots'
    )

    freqs = quantity_to_jnp(ms.meta.freqs)

    apply_gains = jnp.tile(
        jnp.eye(2)[None, None, None, None, :, :],
        (simulation.sky_model.num_sources, len(ms.meta.x), len(ms.meta.antennas), len(freqs), 1, 1)
    ).astype(simulation.dtype)

    gen = ms.create_block_generator(relative_time_idx=True, num_blocks=2)
    gen_response = None
    while True:
        try:
            time, coords, data = gen.send(gen_response)
        except StopIteration:
            break
        t0 = time_mod.time()
        vis = simulation.predict_model_visibilities_jax(freqs=freqs, apply_gains=apply_gains, vis_coords=coords)
        vis.block_until_ready()
        t1 = time_mod.time()
        print(f"Time to simulate: {t1 - t0}")
        assert not np.any(np.isnan(vis))

#
# def test_simulate_visibilties(mock_calibrator_source_models):
#     fits_sources, wsclean_sources, ms = mock_calibrator_source_models
#
#     # print(fits_sources, wsclean_sources, ms)
#
#     simulation = SimulateVisibilities(
#         verbose=True
#     )
#     simulation.simulate(ms=ms,system_gain_model=)