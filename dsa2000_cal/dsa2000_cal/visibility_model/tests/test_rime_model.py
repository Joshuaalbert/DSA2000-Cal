import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np
from astropy import time as at, coordinates as ac, units as au

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry, rfi_model_registry
from dsa2000_cal.geodesic_model.geodesic_model import GeodesicModel
from dsa2000_cal.uvw.far_field import FarFieldDelayEngine, VisibilityCoords
from dsa2000_cal.uvw.near_field import NearFieldDelayEngine
from dsa2000_cal.visibility_model.facet_model import FacetModel
from dsa2000_cal.visibility_model.rime_model import RIMEModel
from dsa2000_cal.visibility_model.source_models.celestial.fits_source.fits_source_model import FITSSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.gaussian_source.gaussian_source_model import \
    GaussianSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.point_source.point_source_model import PointSourceModel
from dsa2000_cal.visibility_model.source_models.rfi.rfi_emitter_source.rfi_emitter_source_model import RFIEmitterSourceModel


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

    lte_source_model = RFIEmitterSourceModel.from_rfi_model(
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
        rfi_emitter_source_model=lte_source_model,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        gain_model=None
    )

    facet_model_data = facet_model.get_model_data(geodesic_model.time_to_jnp(obstimes))

    visibility_coords = far_field_delay_engine.compute_visibility_coords(
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


def test_apply_gains_benchmark_performance():
    num_source = 10
    num_chan = 2
    num_ant = 2048
    num_time = 1

    antennas = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (num_ant, 3))
    antenna_1, antenna_2 = jnp.asarray(
        list(itertools.combinations_with_replacement(range(num_ant), 2))).T

    num_rows = len(antenna_1)

    uvw = antennas[antenna_2] - antennas[antenna_1]
    uvw = uvw.at[:, 2].mul(1e-3)

    times = jnp.arange(num_time) * 1.5
    time_idx = jnp.zeros((num_rows,), jnp.int64)
    time_obs = times[time_idx]

    visibility_coords = VisibilityCoords(
        uvw=uvw,
        time_obs=time_obs,
        antenna_1=antenna_1,
        antenna_2=antenna_2,
        time_idx=time_idx
    )

    vis = jnp.zeros((num_source, num_rows, num_chan, 2, 2), dtype=jnp.complex64)
    gains = jnp.zeros((num_source, num_time, num_ant, num_chan, 2, 2), dtype=jnp.complex64)

    f = jax.jit(RIMEModel.apply_gains).lower(gains=gains, vis=vis, visibility_coords=visibility_coords).compile()
    t0 = time.time()
    f(gains=gains, vis=vis, visibility_coords=visibility_coords).block_until_ready()
    t1 = time.time()
    print(f"Apply gains for sources {num_source} ant {num_ant} freqs {num_chan} took {t1 - t0:.2f} s")
    # Apply gains for sources 10 ant 2048 freqs 16 took 7.98 seconds 1.10 s | 1.00 s | 1.28 s | 1.45 s | 1.00 s
