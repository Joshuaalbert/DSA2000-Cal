import numpy as np
import pytest
from astropy import units as au

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.visibility_model.source_models.celestial.base_fits_source_model import \
    build_fits_source_model_from_wsclean_components


@pytest.mark.parametrize('source', ['cas_a', 'cyg_a', 'tau_a', 'vir_a'])
@pytest.mark.parametrize('crop_box_size', [None, 1*au.arcmin])
def test_plot_ateam_sources(source, crop_box_size):
    fill_registries()

    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match(source)).get_wsclean_fits_files()
    # -04:00:28.608,40.43.33.595

    model_freqs = au.Quantity(np.linspace(65e6, 77e6, 2), 'Hz')

    source_model = build_fits_source_model_from_wsclean_components(
        wsclean_fits_files=wsclean_fits_files,
        model_freqs=model_freqs,
        full_stokes=False,
        crop_box_size=crop_box_size
    )

    source_model.plot()
