import itertools

import jax
import numpy as np
import pylab as plt
from astropy import units as au, constants as const, time as at, coordinates as ac
from jax import numpy as jnp
from tomographic_kernel.frames import ENU

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry, source_model_registry, rfi_model_registry

from dsa2000_cal.common.ellipse_utils import Gaussian
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.wgridder import vis2dirty
from dsa2000_cal.delay_models.far_field import VisibilityCoords, FarFieldDelayEngine
from dsa2000_cal.delay_models.near_field import NearFieldDelayEngine
from dsa2000_cal.geodesics.geodesic_model import GeodesicModel
from dsa2000_cal.visibility_model.facet_model import FacetModel
from dsa2000_cal.visibility_model.rime_model import RIMEModel
from dsa2000_cal.visibility_model.source_models.celestial.fits_source_model import FITSSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.gaussian_source_model import GaussianModelData, \
    GaussianPredict, GaussianSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.point_source_model import PointModelData, PointPredict, \
    PointSourceModel
from dsa2000_cal.visibility_model.source_models.rfi.rfi_emitter_source_model import RFIEmitterSourceModel


def test_facet_model_gaussian():
    major_fwhm_arcsec = 4. * 60
    minor_fwhm_arcsec = 2. * 60
    pos_angle_deg = 90.
    total_flux = 1.

    freq = 70e6 * au.Hz
    wavelength = quantity_to_jnp(const.c / freq)
    num_ant = 128
    num_time = 1

    freqs = quantity_to_jnp(freq)[None]

    max_baseline = 20e3

    antennas = max_baseline * jax.random.normal(jax.random.PRNGKey(42), (num_ant, 3))
    # With autocorr
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

    m0 = 0.
    l0 = 0.1

    # Use wgridder as comparison
    gaussian = Gaussian(
        x0=jnp.asarray([l0, m0]),
        major_fwhm=jnp.asarray(major_fwhm_arcsec / 3600. * np.pi / 180.),
        minor_fwhm=jnp.asarray(minor_fwhm_arcsec / 3600. * np.pi / 180.),
        pos_angle=jnp.asarray(pos_angle_deg / 180. * np.pi),
        total_flux=jnp.asarray(total_flux)
    )

    n = 2096
    pix_size = (wavelength / max_baseline) / 7.
    lvec = pix_size * (-n / 2 + jnp.arange(n)) + l0
    mvec = pix_size * (-n / 2 + jnp.arange(n)) + m0
    L, M = jnp.meshgrid(lvec, mvec, indexing='ij')
    X = jnp.stack([L.flatten(), M.flatten()], axis=-1)
    flux_density = jax.vmap(gaussian.compute_flux_density)(X).reshape(L.shape)
    flux = flux_density * pix_size ** 2

    plt.imshow(flux.T, origin='lower',
               extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]),
               cmap='inferno',
               interpolation='nearest')
    plt.colorbar()
    plt.show()

    gaussian_data = GaussianModelData(
        freqs=freqs,
        image=gaussian.total_flux[None, None],
        gains=None,
        lmn=jnp.asarray([[l0, m0, jnp.sqrt(1. - l0 ** 2 - m0 ** 2)]]),
        ellipse_params=jnp.asarray([[gaussian.major_fwhm,
                                     gaussian.minor_fwhm,
                                     gaussian.pos_angle]])
    )

    gaussian_predict = GaussianPredict(convention='physical',
                                       dtype=jnp.complex64,
                                       order_approx=1)
    vis_gaussian_order_1 = gaussian_predict.predict(model_data=gaussian_data, visibility_coords=visibility_coords)

    dirty = vis2dirty(
        uvw=uvw,
        freqs=freqs,
        vis=vis_gaussian_order_1,
        npix_m=n,
        npix_l=n,
        pixsize_l=pix_size,
        pixsize_m=pix_size,
        center_l=l0,
        center_m=m0,
        epsilon=1e-6
    )  # [nl, nm]

    plt.imshow(
        dirty.T,
        origin='lower',
        extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]),
        cmap='inferno',
        aspect='auto',
        interpolation='nearest'
    )
    plt.colorbar()
    plt.show()


def test_facet_model_point():
    # similar as above
    total_flux = 1.

    freq = 70e6 * au.Hz
    wavelength = quantity_to_jnp(const.c / freq)
    num_ant = 128
    num_time = 1

    freqs = quantity_to_jnp(freq)[None]

    max_baseline = 20e3

    antennas = max_baseline * jax.random.normal(jax.random.PRNGKey(42), (num_ant, 3))
    # With autocorr
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

    m0 = 0.
    l0 = 0.1

    # Use wgridder as comparison

    n = 2096
    pix_size = (wavelength / max_baseline) / 7.
    lvec = pix_size * (-n / 2 + jnp.arange(n)) + l0
    mvec = pix_size * (-n / 2 + jnp.arange(n)) + m0
    L, M = jnp.meshgrid(lvec, mvec, indexing='ij')
    X = jnp.stack([L.flatten(), M.flatten()], axis=-1)

    gaussian_data = PointModelData(
        freqs=freqs,
        image=jnp.asarray(total_flux)[None, None],
        gains=None,
        lmn=jnp.asarray([[l0, m0, jnp.sqrt(1. - l0 ** 2 - m0 ** 2)]])
    )

    predict = PointPredict(convention='physical',
                           dtype=jnp.complex64)
    vis_gaussian_order_1 = predict.predict(model_data=gaussian_data, visibility_coords=visibility_coords)

    dirty = vis2dirty(
        uvw=uvw,
        freqs=freqs,
        vis=vis_gaussian_order_1,
        npix_m=n,
        npix_l=n,
        pixsize_l=pix_size,
        pixsize_m=pix_size,
        center_l=l0,
        center_m=m0,
        epsilon=1e-6
    )  # [nl, nm]

    plt.imshow(
        dirty.T,
        origin='lower',
        extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]),
        cmap='inferno',
        aspect='auto',
        interpolation='none'
    )
    plt.colorbar()
    plt.show()


def test_facet_model_fits():
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

    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match('cas_a')).get_wsclean_fits_files()

    fits_source_model = FITSSourceModel.from_wsclean_model(wsclean_fits_files=wsclean_fits_files,
                                                           phase_tracking=phase_center, freqs=freqs,
                                                           full_stokes=False)

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
        point_source_model=None,
        gaussian_source_model=None,
        fits_source_model=fits_source_model,
        rfi_emitter_source_model=None,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        gain_model=None
    )

    facet_model_data = jax.jit(facet_model.get_model_data)(geodesic_model.time_to_jnp(obstimes))

    print(facet_model_data)

    visibility_coords = far_field_delay_engine.compute_visibility_coords(
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

    print(facet_model)

    facet_model_data = facet_model.get_model_data(geodesic_model.time_to_jnp(obstimes))

    print(facet_model_data)

    visibility_coords = far_field_delay_engine.compute_visibility_coords(
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


def test_facet_model_lte():
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('lwa'))

    antennas = array.get_antennas()
    anteanna_xyz = antennas.itrs.cartesian.xyz.to('m').value.T

    obstimes = at.Time(['2021-01-01T00:00:00', '2021-01-01T00:00:30'], scale='utc')
    max_baseline = np.linalg.norm(anteanna_xyz[:, None, :] - anteanna_xyz[None, :, :], axis=-1).max()
    freqs = au.Quantity([55, 59], 'MHz')

    wavelength = quantity_to_jnp(const.c / freqs[0])
    pixsize = (wavelength / max_baseline) / 10
    print('pixel size (arcmim)', pixsize * 180 / np.pi * 60)
    n = 2000
    print('width (deg)', n * pixsize * 180 / np.pi)
    # return

    array_location = antennas[0]
    ref_time = obstimes[0]

    phase_center = ENU(0, 0, 1, obstime=obstimes[0], location=array_location).transform_to(ac.ICRS())

    geodesic_model = GeodesicModel(
        phase_center=phase_center,
        obstimes=obstimes,
        pointings=None,
        antennas=antennas,
        ref_time=ref_time,
        array_location=array_location
    )

    lte_source_model = RFIEmitterSourceModel.from_rfi_model(
        rfi_model_registry.get_instance(rfi_model_registry.get_match('lwa_cell_tower')),
        freqs,
        central_freq=freqs[0],
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
        point_source_model=None,
        gaussian_source_model=None,
        fits_source_model=None,
        rfi_emitter_source_model=lte_source_model,
        near_field_delay_engine=near_field_delay_engine,
        far_field_delay_engine=far_field_delay_engine,
        gain_model=None
    )

    facet_model_data = facet_model.get_model_data(geodesic_model.time_to_jnp(obstimes))

    visibility_coords = far_field_delay_engine.compute_visibility_coords(
        times=geodesic_model.time_to_jnp(obstimes), with_autocorr=True)

    vis = facet_model.predict(model_data=facet_model_data, visibility_coords=visibility_coords)

    non_finite_mask = ~jnp.isfinite(np.abs(vis))
    print(visibility_coords.antenna_1[np.where(non_finite_mask)[0]])
    print(visibility_coords.antenna_2[np.where(non_finite_mask)[0]])

    l0 = 0.
    m0 = 0.
    dirty = vis2dirty(
        uvw=visibility_coords.uvw,
        freqs=quantity_to_jnp(freqs),
        vis=vis,
        npix_m=n,
        npix_l=n,
        pixsize_l=pixsize,
        pixsize_m=pixsize,
        center_l=l0,
        center_m=m0,
        epsilon=1e-6,
        nthreads=12
    )
    import pylab as plt

    lvec = pixsize * (-n / 2 + np.arange(n)) + l0
    mvec = pixsize * (-n / 2 + np.arange(n)) + m0

    # Find the non-finite dirty
    non_finite_mask = ~jnp.isfinite(np.abs(dirty))
    print(lvec[np.where(non_finite_mask)[0]])
    print(mvec[np.where(non_finite_mask)[1]])

    plt.imshow((dirty).T,
               origin='lower',
               extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]),
               cmap='jet',
               interpolation='nearest'
               )
    plt.colorbar()
    plt.xlabel('l (proj. rad)')
    plt.ylabel('m (proj. rad)')
    plt.title('RFI')
    import glob
    idx = len(glob.glob('lte_full_*.png'))
    # plt.savefig(f'lte_full_{idx:03d}.png', dpi=500)
    plt.show()

    plt.imshow((dirty[900:1100, 900:1100]).T,
               origin='lower',
               extent=(lvec[900], lvec[1100], mvec[900], mvec[1100]),
               cmap='jet',
               interpolation='nearest'
               )
    plt.colorbar()
    plt.xlabel('l (proj. rad)')
    plt.ylabel('m (proj. rad)')
    plt.title('RFI 14km East, 80m up')
    import glob
    idx = len(glob.glob('lte_zoom_*.png'))
    # plt.savefig(f'lte_zoom_{idx:03d}.png', dpi=500)
    plt.show()

    assert np.all(np.isfinite(dirty))


def test_rime_model_correct_shapes():
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
        freqs
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
        facet_models=[facet_model, facet_model, facet_model]
    )
    assert rime_model.num_facets == 3

    vis = rime_model.predict_visibilities(model_data=[facet_model_data, facet_model_data, facet_model_data],
                                          visibility_coords=visibility_coords)

    _vis, _visibility_coords = rime_model.predict_facets_model_visibilities(
        times=geodesic_model.time_to_jnp(obstimes),
        with_autocorr=False
    )

    assert np.shape(vis) == (rime_model.num_facets, 2, 3, 2, 2)
    assert np.shape(_vis) == (rime_model.num_facets, 2, 3, 2, 2)

    gains = jnp.ones((rime_model.num_facets, len(obstimes), len(antennas), len(freqs), 2, 2), dtype=jnp.complex64)
    summed_vis = rime_model.apply_gains(gains, vis, visibility_coords)
    print(summed_vis)
    assert np.shape(summed_vis) == (2, 3, 2, 2)
