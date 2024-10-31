import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pytest
from tomographic_kernel.frames import ENU

from src.dsa2000_cal.assets import fill_registries
from src.dsa2000_cal.assets import array_registry
from dsa2000_cal.imaging.imagor import Imagor
from src.dsa2000_cal.measurement_sets import MeasurementSetMetaV0, MeasurementSet, VisibilityData


def build_calibrator_source_models(array_name, tmp_path, full_stokes, num_chan, corrs):
    fill_registries()
    # Load array
    array = array_registry.get_instance(array_registry.get_match(array_name))
    array_location = array.get_array_location()
    antennas = array.get_antennas()

    obstime = at.Time("2021-01-01T00:00:00", scale='utc')
    phase_tracking = zenith = ENU(0, 0, 1, obstime=obstime, location=array_location).transform_to(ac.ICRS())

    meta = MeasurementSetMetaV0(
        array_name=array_name,
        array_location=array_location,
        phase_tracking=phase_tracking,
        channel_width=array.get_channel_width(),
        integration_time=au.Quantity(1.5, 's'),
        coherencies=corrs if full_stokes else corrs[:1],
        pointings=phase_tracking,
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


@pytest.mark.parametrize("full_stokes", [True, False])
@pytest.mark.parametrize("num_chan", [1, 2])
@pytest.mark.parametrize("corrs", [['XX', 'XY', 'YX', 'YY'], ['I', 'Q', 'U', 'V']])
def test_dirty_imaging(tmp_path, full_stokes, num_chan, corrs):
    ms = build_calibrator_source_models('dsa2000W_small', tmp_path, full_stokes, num_chan, corrs)

    imagor = Imagor(
        plot_folder='plots',
        field_of_view=2 * au.deg
    )
    image_model = imagor.image(image_name='test_dirty', ms=ms, overwrite=True)
    # print(image_model)


@pytest.mark.parametrize("full_stokes", [False])
@pytest.mark.parametrize("num_chan", [40])
@pytest.mark.parametrize("corrs", [['I', 'Q', 'U', 'V']])
def test_demo(tmp_path, full_stokes, num_chan, corrs):
    ms = build_calibrator_source_models('dsa2000W', tmp_path, full_stokes, num_chan, corrs)

    imagor = Imagor(
        plot_folder='plots',
        field_of_view=2 * au.deg
    )
    image_model = imagor.image(image_name='test_dirty', ms=ms, overwrite=True)
    # print(image_model)
