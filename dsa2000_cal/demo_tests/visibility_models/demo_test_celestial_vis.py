import numpy as np
import pytest
from astropy import units as au

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import source_model_registry
from dsa2000_common.visibility_model.source_models.celestial.base_fits_source_model import \
    build_fits_source_model_from_wsclean_components, build_calibration_fits_source_models_from_wsclean
from dsa2000_common.visibility_model.source_models.celestial.base_gaussian_source_model import \
    build_gaussian_source_model_from_wsclean_components
from dsa2000_common.visibility_model.source_models.celestial.base_point_source_model import \
    build_point_source_model_from_wsclean_components


@pytest.mark.parametrize('source', ['cas_a', 'cyg_a', 'tau_a', 'vir_a'])
@pytest.mark.parametrize('crop_box_size', [None, 1 * au.arcmin])
@pytest.mark.parametrize('full_stokes', [True, False])
@pytest.mark.parametrize('num_facets_per_side', [1, 2])
def test_plot_ateam_sources(source, crop_box_size, full_stokes, num_facets_per_side):
    fill_registries()

    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match(source)).get_wsclean_fits_files()
    # -04:00:28.608,40.43.33.595

    model_freqs = au.Quantity(np.linspace(65e6, 77e6, 2), 'Hz')

    source_model = build_fits_source_model_from_wsclean_components(
        wsclean_fits_files=wsclean_fits_files,
        model_freqs=model_freqs,
        full_stokes=full_stokes,
        crop_box_size=crop_box_size,
        num_facets_per_side=num_facets_per_side
    )

    source_model.plot()

    wsclean_clean_component_file = source_model_registry.get_instance(
        source_model_registry.get_match(source)).get_wsclean_clean_component_file()

    sky_model = build_gaussian_source_model_from_wsclean_components(
        wsclean_clean_component_file=wsclean_clean_component_file,
        model_freqs=model_freqs,
        full_stokes=full_stokes
    )
    sky_model.plot()

    sky_model = build_point_source_model_from_wsclean_components(
        wsclean_clean_component_file=wsclean_clean_component_file,
        model_freqs=model_freqs,
        full_stokes=full_stokes
    )
    sky_model.plot()


def test_build_fits_calibration_source_model_from_wsclean_components():
    fill_registries()

    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match('cas_a')).get_wsclean_fits_files()
    # -04:00:28.608,40.43.33.595

    model_freqs = au.Quantity(np.linspace(65e6, 77e6, 2), 'Hz')

    sky_models = build_calibration_fits_source_models_from_wsclean(
        wsclean_fits_files=wsclean_fits_files,
        model_freqs=model_freqs,
        full_stokes=True,
        crop_box_size=None,
        num_facets=2
    )
    print(sky_models)
