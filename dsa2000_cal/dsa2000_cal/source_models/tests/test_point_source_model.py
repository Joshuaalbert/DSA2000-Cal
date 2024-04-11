from timeit import default_timer

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.measurement_sets.measurement_set import VisibilityData
from dsa2000_cal.source_models.point_source_model import PointSourceModel


def test_point_sources():
    fill_registries()
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    # -00:36:28.234,58.50.46.396
    source_file = source_model_registry.get_instance(source_model_registry.get_match('cas_a')).get_wsclean_source_file()
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "58d50m46.396s", frame='icrs')
    phase_tracking = ac.SkyCoord("-00h36m28.234s", "78d50m46.396s", frame='icrs')

    # -04:00:28.608,40.43.33.595
    # source_file = source_model_registry.get_instance(source_model_registry.get_match('cyg_a')).get_wsclean_source_file()
    # phase_tracking = ac.SkyCoord("-04h00m28.608s", "40d43m33.595s", frame='icrs')

    freqs = au.Quantity([50e6, 80e6], 'Hz')

    point_sources = PointSourceModel.from_wsclean_model(
        wsclean_file=source_file,
        time=time,
        phase_tracking=phase_tracking,
        freqs=freqs
    )

    point_sources.plot()


def test_point_source_predict(mock_measurement_set):
    fill_registries()

    source_file = source_model_registry.get_instance(source_model_registry.get_match('cas_a')).get_wsclean_source_file()

    gen = mock_measurement_set.create_block_generator()
    gen_response = None
    while True:
        try:
            time, coords, data = gen.send(gen_response)

        except StopIteration:
            break
        point_sources = PointSourceModel.from_wsclean_model(
            wsclean_file=source_file,
            time=time,
            phase_tracking=mock_measurement_set.meta.phase_tracking,
            freqs=mock_measurement_set.meta.freqs
        )
        t0 = default_timer()
        vis = point_sources.predict(uvw=coords.uvw * au.m)
        gen_response = VisibilityData(
            vis=np.asarray(vis)
        )
        print(f"Time to predict: {default_timer() - t0}")

