import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
from tomographic_kernel.frames import ENU

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_fm.measurement_sets.measurement_set import MeasurementSetMeta, MeasurementSet, VisibilityData


def build_calibrator_source_models(array_name, tmp_path, full_stokes, num_chan, corrs):
    fill_registries()
    # Load array
    array = array_registry.get_instance(array_registry.get_match(array_name))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    obstime = at.Time("2021-01-01T00:00:00", scale='utc')
    phase_center = zenith = ENU(0, 0, 1, obstime=obstime, location=array_location).transform_to(ac.ICRS())

    meta = MeasurementSetMeta(
        array_name=array_name,
        array_location=array_location,
        phase_center=phase_center,
        channel_width=array.get_channel_width(),
        integration_time=au.Quantity(1.5, 's'),
        coherencies=corrs if full_stokes else corrs[:1],
        pointings=phase_center,
        times=obstime + 1.5 * np.arange(1) * au.s,
        freqs=au.Quantity(np.linspace(700, 2000, num_chan), unit=au.MHz),
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
            vis=np.ones_like(data.vis),
            flags=np.zeros_like(data.flags),
            weights=np.ones_like(data.weights)
        )

    return ms
