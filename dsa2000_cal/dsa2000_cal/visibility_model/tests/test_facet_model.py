import jax
import numpy as np
import pylab as plt
from astropy import time as at, coordinates as ac, units as au, constants as const

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry, rfi_model_registry, array_registry
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.wgridder import vis2dirty
from dsa2000_cal.gain_models.geodesic_model import GeodesicModel
from dsa2000_cal.uvw.far_field import FarFieldDelayEngine
from dsa2000_cal.uvw.near_field import NearFieldDelayEngine
from dsa2000_cal.visibility_model.facet_model import FacetModel
from dsa2000_cal.visibility_model.source_models.celestial.fits_source.fits_source_model import FITSSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.gaussian_source.gaussian_source_model import \
    GaussianSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.point_source.point_source_model import PointSourceModel
from dsa2000_cal.visibility_model.source_models.rfi.lte_source.lte_source_model import LTESourceModel


def test_facet_model_points():
    fill_registries()

    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    antennas = array.get_antennas()

    obstimes = at.Time(['2021-01-01T00:00:00'], scale='utc')
    # -00:37:22.645,58.30.45.773
    phase_center = ac.SkyCoord("-00h37m22.645s", "58d30m45.773s", frame='icrs').transform_to(ac.ICRS())
    array_location = antennas[0]
    ref_time = obstimes[0]
    freqs = au.Quantity([50, 60], 'MHz')

    geodesic_model = GeodesicModel(
        phase_center=phase_center,
        obstimes=obstimes,
        pointings=phase_center,
        antennas=antennas,
        ref_time=ref_time,
        array_location=array_location
    )

    source_file = source_model_registry.get_instance(
        source_model_registry.get_match('cas_a')).get_wsclean_clean_component_file()

    point_source_model = PointSourceModel.from_wsclean_model(
        source_file,
        phase_center,
        freqs,
        full_stokes=False
    )
    gaussian_source_model = GaussianSourceModel.from_wsclean_model(
        source_file,
        phase_center,
        freqs,
        full_stokes=False
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
        fits_source_model=None,
        lte_source_model=None,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        gain_model=None
    )

    facet_model_data = jax.jit(facet_model.get_model_data)(geodesic_model.time_to_jnp(obstimes))

    print(facet_model_data)

    visibility_coords = far_field_delay_engine.batched_compute_uvw_jax(
        times=geodesic_model.time_to_jnp(obstimes), with_autocorr=True)

    print(visibility_coords)
    plt.plot(visibility_coords.uvw[:, 0], visibility_coords.uvw[:, 1], 'o')
    plt.show()

    vis = jax.jit(facet_model.predict)(model_data=facet_model_data, visibility_coords=visibility_coords)

    print(vis)

    pixsize = quantity_to_jnp(const.c / freqs[0] / (20 * au.km))
    print(pixsize * 180 / np.pi * 3600)
    n = 4096

    dirty = vis2dirty(
        uvw=visibility_coords.uvw,
        freqs=quantity_to_jnp(freqs),
        vis=vis,
        npix_m=n,
        npix_l=n,
        pixsize_l=pixsize,
        pixsize_m=pixsize,
        center_l=0.,
        center_m=0.,
        epsilon=1e-6
    )

    mvec = lvec = pixsize * (-n / 2 + np.arange(n))

    plt.imshow(np.abs(dirty).T, origin='lower',
               extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]),
               cmap='inferno',
               )
    plt.colorbar()
    plt.show()


def test_facet_model():
    fill_registries()

    obstimes = at.Time(['2021-01-01T00:00:00', '2021-01-01T00:00:30'], scale='utc')
    antennas = ac.EarthLocation.from_geocentric([1, 0] * au.km, [0, 0] * au.km, [0, 1] * au.km)
    phase_center = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    array_location = antennas[0]
    ref_time = obstimes[0]
    freqs = au.Quantity([50, 60], 'MHz')

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

    print(facet_model_data)

    visibility_coords = far_field_delay_engine.batched_compute_uvw_jax(
        times=geodesic_model.time_to_jnp(obstimes), with_autocorr=False)

    print(visibility_coords)

    vis = facet_model.predict(model_data=facet_model_data, visibility_coords=visibility_coords)

    print(vis)
    pixsize = quantity_to_jnp(const.c / (freqs[0] * au.MHz) / (20 * au.km))
    print(pixsize * 180 / np.pi * 3600)
    n = 4096
    dirty = vis2dirty(
        uvw=visibility_coords.uvw,
        freqs=quantity_to_jnp(freqs),
        vis=vis[:, :, 0, 0],
        npix_m=n,
        npix_l=n,
        pixsize_l=pixsize,
        pixsize_m=pixsize,
        center_l=0.,
        center_m=0.,
        epsilon=1e-6
    )
    import pylab as plt

    mvec = lvec = pixsize * (-n / 2 + np.arange(n))

    plt.imshow(np.abs(dirty).T, origin='lower',
               extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]),
               cmap='inferno',
               )
    plt.show()
