import jax.numpy as jnp
import numpy as np
from astropy import time as at, coordinates as ac, units as au

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry, rfi_model_registry
from dsa2000_cal.gain_models.geodesic_model import GeodesicModel
from dsa2000_cal.uvw.far_field import FarFieldDelayEngine
from dsa2000_cal.uvw.near_field import NearFieldDelayEngine
from dsa2000_cal.visibility_model.facet_model import FacetModel
from dsa2000_cal.visibility_model.rime_model import RIMEModel
from dsa2000_cal.visibility_model.source_models.celestial.fits_source.fits_source_model import FITSSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.gaussian_source.gaussian_source_model import \
    GaussianSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.point_source.point_source_model import PointSourceModel
from dsa2000_cal.visibility_model.source_models.rfi.lte_source.lte_source_model import LTESourceModel


def test_rime_model():
    fill_registries()

    obstimes = at.Time(['2021-01-01T00:00:00', '2021-01-01T00:00:30'], scale='utc')
    antennas = ac.EarthLocation.from_geocentric([1, 0] * au.km, [0, 0] * au.km, [0, 1] * au.km)
    phase_center = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    array_location = antennas[0]
    ref_time = obstimes[0]
    freqs = au.Quantity([50, 55, 60], 'MHz')

    geodesic_model = GeodesicModel(
        phase_center=phase_center,
        obstimes=obstimes,
        pointings=None,
        antennas=antennas,
        ref_time=ref_time,
        array_location=array_location
    )

    source_file = source_model_registry.get_instance(
        source_model_registry.get_match('cas_a')).get_wsclean_clean_component_file()

    point_source_model = PointSourceModel.from_wsclean_model(
        source_file,
        phase_center,
        freqs
    )
    gaussian_source_model = GaussianSourceModel.from_wsclean_model(
        source_file,
        phase_center,
        freqs
    )

    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match('cas_a')).get_wsclean_fits_files()

    fits_source_model = FITSSourceModel.from_wsclean_model(wsclean_fits_files=wsclean_fits_files,
                                                           phase_tracking=phase_center, freqs=freqs)

    lte_source_model = LTESourceModel.from_rfi_model(
        rfi_model_registry.get_instance(rfi_model_registry.get_match('lte_cell_tower')),
        freqs,
        True
    )

    near_field_delay_engine = NearFieldDelayEngine(
        antennas=antennas,
        start_time=obstimes[0],
        end_time=obstimes[-1],
        verbose=True
    )
    far_field_delay_engine = FarFieldDelayEngine(
        antennas=antennas,
        start_time=obstimes[0],
        end_time=obstimes[-1],
        phase_center=phase_center,
        verbose=True
    )

    facet_model = FacetModel(
        geodesic_model=geodesic_model,
        point_source_model=point_source_model,
        gaussian_source_model=gaussian_source_model,
        fits_source_model=fits_source_model,
        lte_source_model=lte_source_model,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        gain_model=None
    )

    facet_model_data = facet_model.get_model_data(geodesic_model.time_to_jnp(obstimes))

    visibility_coords = far_field_delay_engine.batched_compute_uvw_jax(
        times=geodesic_model.time_to_jnp(obstimes), with_autocorr=False)

    rime_model = RIMEModel(
        facet_models=[facet_model]
    )

    vis = rime_model.predict_visibilities(model_data=[facet_model_data], visibility_coords=visibility_coords)

    _vis, _visibility_coords = rime_model.predict_facets_model_visibilities(times=geodesic_model.time_to_jnp(obstimes),
                                                                            with_autocorr=False)

    assert np.shape(vis) == (1, 2, 3, 2, 2)
    assert np.shape(_vis) == (1, 2, 3, 2, 2)

    gains = jnp.ones((1, len(obstimes), len(antennas), len(freqs), 2, 2), dtype=jnp.complex64)
    summed_vis = rime_model.apply_gains(gains, vis, visibility_coords)
    print(summed_vis)
