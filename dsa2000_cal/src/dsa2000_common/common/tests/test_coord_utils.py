import numpy as np
import pylab as plt
import pytest
from astropy import coordinates as ac, time as at, units as au
from astropy.coordinates import offset_by
from astropy.wcs import WCS

from dsa2000_common.common.coord_utils import earth_location_to_uvw_approx, icrs_to_lmn, lmn_to_icrs, \
    earth_location_to_enu, \
    icrs_to_enu, enu_to_icrs, lmn_to_icrs_old
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.quantity_utils import quantity_to_jnp


def test_enu_to_uvw():
    antennas = ac.EarthLocation.of_site('vla')
    array_location = ac.EarthLocation.of_site('vla')
    time = at.Time('2000-01-01T00:00:00', format='isot')
    pointing = ac.ICRS(0 * au.deg, 90 * au.deg)
    uvw = earth_location_to_uvw_approx(antennas, time, pointing)
    assert np.linalg.norm(uvw) < 6400 * au.km

    enu_frame = ENU(location=ac.EarthLocation.of_site('vla'), obstime=time)
    antennas = ac.SkyCoord(
        east=np.random.uniform(size=(10,), low=-5, high=5) * au.km,
        north=np.random.uniform(size=(10,), low=-5, high=5) * au.km,
        up=np.random.uniform(size=(10,), low=-5, high=5) * au.km,
        frame=enu_frame
    ).transform_to(ac.ITRS).earth_location
    uvw = earth_location_to_uvw_approx(antennas, time, pointing)
    assert np.all(np.linalg.norm(uvw, axis=-1) < 6400 * au.km)


def test_lmn_to_icrs():
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    pointing = ac.ICRS(0 * au.deg, 0 * au.deg)
    lmn = icrs_to_lmn(pointing, pointing)
    np.testing.assert_allclose(lmn, au.Quantity([0, 0, 1] * au.rad), atol=1e-6)

    sources = ac.ICRS(4 * au.deg, 2 * au.deg)
    lmn = icrs_to_lmn(sources, pointing)
    reconstructed_sources = lmn_to_icrs(lmn, pointing)
    print(sources)
    print(lmn)
    print(reconstructed_sources)
    assert sources.separation(reconstructed_sources).max() < 1e-6 * au.deg

    sources = ac.ICRS([1, 2, 3, 4] * au.deg, [1, 2, 3, 4] * au.deg).reshape((2, 2))
    lmn = icrs_to_lmn(sources, pointing)
    assert lmn.shape == (2, 2, 3)
    reconstructed_sources = lmn_to_icrs(lmn, pointing)
    assert isinstance(reconstructed_sources, ac.ICRS)
    assert reconstructed_sources.shape == (2, 2)
    assert sources.separation(reconstructed_sources).max() < 1e-6 * au.deg


def test_icrs_to_lmn():
    time = at.Time("2021-01-01T00:00:00", scale='utc')

    pointing = ac.ICRS(0 * au.deg, 0 * au.deg)
    lmn = icrs_to_lmn(pointing, pointing)
    np.testing.assert_allclose(lmn, au.Quantity([0, 0, 1] * au.rad), atol=1e-10)

    pointing = ac.ICRS(0 * au.deg, 0 * au.deg)
    sources = ac.ICRS(4 * au.deg, 2 * au.deg)
    lmn1 = icrs_to_lmn(sources, pointing)

    # No impact due to rotation
    time = at.Time("2022-02-01T00:00:00", scale='utc')
    lmn2 = icrs_to_lmn(sources, pointing)
    assert np.all(lmn1 == lmn2)

    # Test when at zentih
    array_location = ac.EarthLocation.of_site('vla')
    obstime = at.Time('2000-01-01T00:00:00', format='isot')
    frame = ac.AltAz(location=array_location, obstime=obstime)
    zenith = ac.SkyCoord(alt=90 * au.deg, az=0 * au.deg, frame=frame).transform_to(ac.ICRS())
    lmn = icrs_to_lmn(zenith, zenith)
    np.testing.assert_allclose(lmn, np.array([0, 0, 1]) * au.rad, atol=1e-10)

    lmn = icrs_to_lmn(sources, zenith)
    print(lmn)

    # Another test of zenith
    sources = ENU(
        east=0,
        north=0,
        up=1,
        obstime=obstime,
        location=array_location
    ).transform_to(ac.ICRS())
    lmn_sources = quantity_to_jnp(icrs_to_lmn(sources, phase_center=zenith))
    np.testing.assert_allclose(lmn_sources, [0, 0, 1], atol=1e-10)

    # Another test of L
    sources = ENU(
        east=1,
        north=0,
        up=0,
        obstime=obstime,
        location=array_location
    ).transform_to(ac.ICRS())
    lmn_sources = quantity_to_jnp(icrs_to_lmn(sources, phase_center=zenith))
    np.testing.assert_allclose(lmn_sources, [1, 0, 0], atol=1e-3)

    # Another test of M
    sources = ENU(
        east=0,
        north=1,
        up=0,
        obstime=obstime,
        location=array_location
    ).transform_to(ac.ICRS())
    lmn_sources = quantity_to_jnp(icrs_to_lmn(sources, phase_center=zenith))
    np.testing.assert_allclose(lmn_sources, [0, 1, 0], atol=1e-3)


@pytest.mark.parametrize('broadcast_phase_center', [False, True])
@pytest.mark.parametrize('broadcast_lmn', [False, True])
def test_lmn_to_icrs_vectorised(broadcast_phase_center, broadcast_lmn):
    np.random.seed(42)
    if broadcast_phase_center:
        phase_center = ac.ICRS([0, 0, 0, 0] * au.deg, [0, 0, 0, 0] * au.deg).reshape(
            (1, 4, 1)
        )
    else:
        phase_center = ac.ICRS(0 * au.deg, 0 * au.deg)
    if broadcast_lmn:
        lmn = np.random.normal(size=(5, 1, 1, 3)) * au.rad
    else:
        lmn = np.random.normal(size=(3,)) * au.rad
    lmn /= np.linalg.norm(lmn, axis=-1, keepdims=True)

    expected_shape = np.broadcast_shapes(lmn.shape[:-1], phase_center.shape)

    sources = lmn_to_icrs(lmn, phase_center)
    assert sources.shape == expected_shape


@pytest.mark.parametrize('broadcast_phase_center', [False, True])
@pytest.mark.parametrize('broadcast_lmn', [False, True])
def test_icrs_to_lmn_vectorised(broadcast_phase_center, broadcast_lmn):
    np.random.seed(42)
    if broadcast_phase_center:
        phase_center = ac.ICRS([0, 0, 0, 0] * au.deg, [0, 0, 0, 0] * au.deg).reshape(
            (1, 4, 1)
        )
    else:
        phase_center = ac.ICRS(0 * au.deg, 0 * au.deg)
    if broadcast_lmn:
        sources = ac.ICRS([0, 0, 0, 0, 0] * au.deg, [0, 0, 0, 0, 0] * au.deg).reshape((5, 1, 1))
    else:
        sources = ac.ICRS(0 * au.deg, 0 * au.deg)

    expected_shape = np.broadcast_shapes(sources.shape, phase_center.shape) + (3,)

    sources = icrs_to_lmn(sources, phase_center)
    assert sources.shape == expected_shape


def test_lmn_to_icrs_near_poles():
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    lmn = au.Quantity(
        [
            [0.05, 0.0, np.sqrt(1 - 0.05 ** 2)],
            [0.0, 0.05, np.sqrt(1 - 0.05 ** 2)],
            [0.0, 0.0, 1],
        ],
        'rad'
    )

    pointing_north_pole = ac.ICRS(0 * au.deg, 90 * au.deg)
    sources = lmn_to_icrs(lmn, pointing_north_pole)
    print(sources)

    lmn_reconstructed = icrs_to_lmn(sources, pointing_north_pole)
    print(lmn_reconstructed)

    np.testing.assert_allclose(lmn, lmn_reconstructed, atol=1e-6)

    # Near south pole
    pointing_south_pole = ac.ICRS(0 * au.deg, -90 * au.deg)
    sources = lmn_to_icrs(lmn, pointing_south_pole)
    print(sources)

    lmn_reconstructed = icrs_to_lmn(sources, pointing_south_pole)
    print(lmn_reconstructed)

    np.testing.assert_allclose(lmn, lmn_reconstructed, atol=1e-6)


def test_earth_location_to_enu():
    antennas = ac.EarthLocation.of_site('vla')
    array_location = ac.EarthLocation.of_site('vla')
    time = at.Time('2000-01-01T00:00:00', format='isot')
    enu = earth_location_to_enu(antennas, array_location, time)
    assert np.linalg.norm(enu.cartesian.xyz.T) < 6400 * au.km

    enu_frame = ENU(location=ac.EarthLocation.of_site('vla'), obstime=time)
    n = 500
    antennas = ac.SkyCoord(
        east=np.random.uniform(size=(n,), low=-5, high=5) * au.km,
        north=np.random.uniform(size=(n,), low=-5, high=5) * au.km,
        up=np.random.uniform(size=(n,), low=-5, high=5) * au.km,
        frame=enu_frame
    ).transform_to(ac.ITRS).earth_location
    enu = earth_location_to_enu(antennas, array_location, time).cartesian.xyz.T
    # print(enu)

    dist = np.linalg.norm(enu[:, None, :] - enu[None, :, :], axis=-1)
    assert np.all(dist < np.sqrt(3) * 10 * au.km)


def test_icrs_to_enu():
    sources = ac.ICRS(0 * au.deg, 90 * au.deg)
    array_location = ac.EarthLocation.of_site('vla')
    time = at.Time('2000-01-01T00:00:00', format='isot')
    enu = icrs_to_enu(sources, array_location, time)
    print(enu)
    np.testing.assert_allclose(np.linalg.norm(enu.cartesian.xyz.T), 1.)

    reconstruct_sources = enu_to_icrs(enu)
    print(reconstruct_sources)
    np.testing.assert_allclose(sources.separation(reconstruct_sources).deg, 0., atol=1e-6)


def test_enu_to_icrs():
    enu_coords = np.array([[0, 1, 0], [0, 0, 1]]) * au.km
    array_location = ac.EarthLocation.of_site('vla')
    time = at.Time('2000-01-01T00:00:00', format='isot')
    enu = ENU(east=enu_coords[:, 0], north=enu_coords[:, 1], up=enu_coords[:, 2], location=array_location, obstime=time)
    sources = enu_to_icrs(enu)
    print(sources)
    # np.testing.assert_allclose(np.linalg.norm(sources.cartesian.xyz, axis=-1), 1.)
    reconstruct_enu = icrs_to_enu(sources, array_location, time)
    print(reconstruct_enu)
    np.testing.assert_allclose(enu.cartesian.xyz, reconstruct_enu.cartesian.xyz, atol=1e-6)


# @pytest.mark.parametrize('offset_dt', [1 * au.hour,
#                                        1 * au.day,
#                                        2 * au.day,
#                                        1 * au.year])
def test_lmn_to_icrs_over_year():
    mvec = lvec = np.linspace(-0.99, 0.99, 100) * au.rad

    M, L = np.meshgrid(mvec, lvec, indexing='ij')
    R = np.sqrt(L ** 2 + M ** 2)

    lmn = np.stack([L, M, np.sqrt(1 - L ** 2 - M ** 2)], axis=-1)

    phase_center = ac.ICRS(0 * au.deg, 0 * au.deg)
    time0 = at.Time("2021-01-01T00:00:00", scale='utc')
    icrs0 = lmn_to_icrs(lmn, phase_center)
    for offset_dt in np.linspace(0., 1., 10) * au.year:
        time1 = time0 + offset_dt
        icrs1 = lmn_to_icrs_old(lmn, time1, phase_center)
        sep = icrs0.separation(icrs1)
        print(offset_dt.to(au.year).value)
        color = offset_dt.to(au.year).value * np.ones(lvec.size)
        bins = np.linspace(0., 1., 101)
        freqs, _ = np.histogram(R.flatten(), bins=bins, weights=sep.to(au.arcsec).value.flatten())
        counts, _ = np.histogram(R.flatten(), bins=bins)
        avg_sep = freqs / counts

        print(lvec.size)
        print(avg_sep.size)
        print(color.size)

        plt.scatter(bins[:-1] + 0.5 * (bins[1] - bins[0]),
                    avg_sep,
                    c=color,
                    cmap='hsv',
                    vmin=0.,
                    vmax=1.)
    plt.xlabel(r'$\sqrt{l^2 + m^2}$')
    plt.ylabel('Astrometry error (arcsec)')
    plt.colorbar(label='Years', alpha=1)
    plt.show()
    # im = plt.imshow(sep.to(au.arcsec).value,
    #                 origin='lower',
    #                 extent=(lvec[0].value, lvec[-1].value, mvec[0].value, mvec[-1].value))
    # plt.xlabel('l')
    # plt.ylabel('m')
    # plt.colorbar(im)
    # plt.show()
    # assert np.all(np.where(np.isnan(sep), True, sep < 0.03 * au.arcsec))


def test_lmn_to_icrs_over_time():
    mvec = lvec = np.linspace(-0.99, 0.99, 100) * au.rad

    M, L = np.meshgrid(mvec, lvec, indexing='ij')

    lmn = au.Quantity(np.stack([L, M, np.sqrt(1 - L ** 2 - M ** 2)], axis=-1))

    phase_center = ac.ICRS(0 * au.deg, 0 * au.deg)
    time0 = at.Time("2021-01-01T00:00:00", scale='utc')
    icrs0 = lmn_to_icrs(lmn, phase_center)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    for i, offset_dt in enumerate([1 * au.hour, 10 * au.day, 180 * au.day, 1 * au.year]):
        ax = axs.flatten()[i]
        time1 = time0 + offset_dt
        icrs1 = lmn_to_icrs_old(lmn, time1, phase_center)
        sep = icrs0.separation(icrs1)
        im = ax.imshow(sep.to(au.arcsec).value,
                       origin='lower',
                       extent=(lvec[0].value, lvec[-1].value, mvec[0].value, mvec[-1].value))
        ax.set_xlabel('l')
        ax.set_ylabel('m')
        ax.set_title(f'Offset: {offset_dt}')
        # Make smaller colorbar
        fig.colorbar(im, ax=ax, label='Error (arcsec)',
                     fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()


def test_lmn_coords():
    phase_center = ac.ICRS(0 * au.deg, 0 * au.deg)
    distance = 0.1 * au.deg
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    # Offset north
    dnorth_icrs = ac.ICRS(
        *offset_by(lon=phase_center.ra, lat=phase_center.dec, posang=0 * au.deg, distance=distance))
    lmn_dnorth = icrs_to_lmn(dnorth_icrs, phase_center)
    m_dnorth = lmn_dnorth[1]
    print('North', dnorth_icrs)
    print(lmn_dnorth)
    assert m_dnorth > 0
    # Offset east
    deast_icrs = ac.ICRS(
        *offset_by(lon=phase_center.ra, lat=phase_center.dec, posang=90 * au.deg, distance=distance))
    lmn_deast = icrs_to_lmn(deast_icrs, phase_center)
    l_deast = lmn_deast[0]
    print('East', deast_icrs)
    print(lmn_deast)
    assert l_deast > 0
    # Offset south
    dsouth_icrs = ac.ICRS(
        *offset_by(lon=phase_center.ra, lat=phase_center.dec, posang=180 * au.deg, distance=distance))
    lmn_dsouth = icrs_to_lmn(dsouth_icrs, phase_center)
    m_dsouth = lmn_dsouth[1]
    print('South', dsouth_icrs)
    print(lmn_dsouth)
    assert m_dsouth < 0
    # Offset west
    dwest_icrs = ac.ICRS(
        *offset_by(lon=phase_center.ra, lat=phase_center.dec, posang=270 * au.deg, distance=distance))
    lmn_dwest = icrs_to_lmn(dwest_icrs, phase_center)
    l_dwest = lmn_dwest[0]
    print('West', dwest_icrs)
    print(lmn_dwest)
    assert l_dwest < 0


def test_lmn_ellipse_to_sky():
    # Create an ellipse in LM-coords
    major = 0.1 * au.rad
    minor = 0.05 * au.rad
    theta = 45 * au.deg

    l0 = 0. * au.rad
    m0 = 0. * au.rad

    phi = np.linspace(0, 2 * np.pi, 100) * au.rad
    m_circle = np.cos(phi)
    l_circle = np.sin(phi)

    # Convert to ellipse using rotation @ scale @ circ_coords
    l_scaled = l_circle * minor
    m_scaled = m_circle * major

    # Rotate by theta, m aligns with major
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    circ_vec = np.stack([l_scaled, m_scaled], axis=0)
    rot_vec = np.dot(R, circ_vec)
    l_rot = rot_vec[0]
    m_rot = rot_vec[1]

    # Translate
    l = l_rot + l0
    m = m_rot + m0

    # Convert to sky
    phase_center = ac.ICRS(15 * au.deg, 75 * au.deg)
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    lmn = au.Quantity(np.stack([l, m, np.sqrt(1 - l ** 2 - m ** 2)], axis=-1))
    icrs = lmn_to_icrs(lmn, phase_center)

    # plot

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---AIT', 'DEC--AIT']  # AITOFF projection
    wcs.wcs.crval = [0, 0]  # Center of the projection
    wcs.wcs.crpix = [0, 0]
    wcs.wcs.cdelt = [-1, 1]

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5), subplot_kw=dict(projection=wcs))
    ax[0][0].plot(icrs.ra.deg, icrs.dec.deg, marker='o',
                  transform=ax[0][0].get_transform('world'))
    ax[0][0].set_xlabel('Right Ascension')
    ax[0][0].set_ylabel('Declination')
    ax[0][0].set_title("Ellipse on the sky")
    ax[0][0].grid()
    fig.tight_layout()
    fig.savefig('ellipse_on_sky.png')
    plt.show()

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))

    ax[0][0].plot(l, m)
    ax[0][0].set_xlabel('l')
    ax[0][0].set_ylabel('m')
    ax[0][0].set_title("Ellipse in LM-coords")
    ax[0][0].grid()
    fig.tight_layout()
    fig.savefig('ellipse_in_plane_of_sky.png')
    plt.show()


def test_earth_location_to_uvw():
    phase_center = ac.ICRS(0 * au.deg, 0 * au.deg)
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    antennas = ac.EarthLocation.of_site('vla')
    uvw = earth_location_to_uvw_approx(antennas=antennas, obs_time=time, phase_center=phase_center)
    print(uvw)

    # TODO: should compare to known values.
