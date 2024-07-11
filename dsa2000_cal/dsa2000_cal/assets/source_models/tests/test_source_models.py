import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.visibility_model.source_models.celestial.fits_source import FitsStokesISourceModel


def test_wsclean_component_files():
    fill_registries()
    # Create a sky model for calibration
    for source in ['cas_a', 'cyg_a', 'tau_a', 'vir_a']:
        source_model_asset = source_model_registry.get_instance(source_model_registry.get_match(source))
        time = at.Time('2021-01-01T00:00:00', format='isot', scale='utc')
        freqs = np.linspace(700e6, 2000e6, 1) * au.Hz
        phase_tracking = ac.ICRS(ra=ac.Angle('0h'), dec=ac.Angle('0d'))
        # source_model = WSCleanSourceModel.from_wsclean_model(
        #     wsclean_clean_component_file=source_model_asset.get_wsclean_clean_component_file(),
        #     time=at.Time('2021-01-01T00:00:00', format='isot', scale='utc'),
        #     freqs=np.linspace(700e6, 2000e6, 2) * au.Hz,
        #     phase_tracking=ac.ICRS(ra=ac.Angle('0h'), dec=ac.Angle('0d'))
        # )

        fits_model = FitsStokesISourceModel.from_wsclean_model(source_model_asset.get_wsclean_fits_files(),
                                                               phase_tracking, freqs, ignore_out_of_bounds=True)
        fits_model.plot()
