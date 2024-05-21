import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.source_models.gaussian_stokes_I_source_model import GaussianSourceModel


def test_gaussian_sources():
    fill_registries()
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    # source_file = source_model_registry.get_instance(source_model_registry.get_match('cas_a')).get_wsclean_source_file()
    # -00:36:28.234,58.50.46.396
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "58d50m46.396s", frame='icrs')
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "78d50m46.396s", frame='icrs')

    source_file = source_model_registry.get_instance(
        source_model_registry.get_match('cyg_a')).get_wsclean_clean_component_file()
    # -04:00:28.608,40.43.33.595
    phase_tracking = ac.SkyCoord("-04h00m28.608s", "40d43m33.595s", frame='icrs')

    freqs = au.Quantity([50e6, 80e6], 'Hz')

    gaussian_source_model = GaussianSourceModel.from_wsclean_model(
        wsclean_clean_component_file=source_file,
        time=time,
        phase_tracking=phase_tracking,
        freqs=freqs,
        lmn_transform_params=True
    )

    assert isinstance(gaussian_source_model, GaussianSourceModel)
    assert gaussian_source_model.num_sources > 0

    gaussian_source_model.plot()


