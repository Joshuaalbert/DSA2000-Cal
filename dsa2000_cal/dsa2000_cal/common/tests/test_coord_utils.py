import numpy as np
import pylab as plt
from astropy import coordinates as ac, time as at, units as au
from astropy.coordinates import offset_by
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.coord_utils import earth_location_to_uvw, icrs_to_lmn, lmn_to_icrs, earth_location_to_enu, \
    icrs_to_enu, enu_to_icrs


def test_enu_to_uvw():
    antennas = ac.EarthLocation.of_site('vla')
    array_location = ac.EarthLocation.of_site('vla')
    time = at.Time('2000-01-01T00:00:00', format='isot')
    pointing = ac.ICRS(0 * au.deg, 90 * au.deg)
    uvw = earth_location_to_uvw(antennas, time, pointing)
    assert np.linalg.norm(uvw) < 6400 * au.km

    enu_frame = ENU(location=ac.EarthLocation.of_site('vla'), obstime=time)
    antennas = ac.SkyCoord(
        east=np.random.uniform(size=(10,), low=-5, high=5) * au.km,
        north=np.random.uniform(size=(10,), low=-5, high=5) * au.km,
        up=np.random.uniform(size=(10,), low=-5, high=5) * au.km,
        frame=enu_frame
    ).transform_to(ac.ITRS).earth_location
    uvw = earth_location_to_uvw(antennas, time, pointing)
    assert np.all(np.linalg.norm(uvw, axis=-1) < 6400 * au.km)


def test_lmn_to_icrs():
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    pointing = ac.ICRS(0 * au.deg, 0 * au.deg)
    sources = ac.ICRS(4 * au.deg, 2 * au.deg)
    lmn = icrs_to_lmn(sources, time, pointing)
    reconstructed_sources = lmn_to_icrs(lmn, time, pointing)
    print(sources)
    print(lmn)
    print(reconstructed_sources)
    assert sources.separation(reconstructed_sources).max() < 1e-10 * au.deg

    sources = ac.ICRS([1, 2, 3, 4] * au.deg, [1, 2, 3, 4] * au.deg).reshape((2, 2))
    lmn = icrs_to_lmn(sources, time, pointing)
    assert lmn.shape == (2, 2, 3)
    reconstructed_sources = lmn_to_icrs(lmn, time, pointing)
    assert reconstructed_sources.shape == (2, 2)
    assert sources.separation(reconstructed_sources).max() < 1e-10 * au.deg


def test_icrs_to_lmn():
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    pointing = ac.ICRS(0 * au.deg, 0 * au.deg)
    sources = ac.ICRS(4 * au.deg, 2 * au.deg)
    lmn1 = icrs_to_lmn(sources, time, pointing)

    # Different at different times due to Earth's rotation
    time = at.Time("2022-02-01T00:00:00", scale='utc')
    lmn2 = icrs_to_lmn(sources, time, pointing)
    assert np.all(lmn1 != lmn2)


def test_lmn_to_icrs_near_poles():
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    lmn = au.Quantity(
        [
            [0.05, 0.0, np.sqrt(1 - 0.05 ** 2)],
            [0.0, 0.05, np.sqrt(1 - 0.05 ** 2)],
            [0.0, 0.0, 1],
        ]
    )

    pointing_north_pole = ac.ICRS(0 * au.deg, 90 * au.deg)
    sources = lmn_to_icrs(lmn, time, pointing_north_pole)
    print(sources)

    lmn_reconstructed = icrs_to_lmn(sources, time, pointing_north_pole)
    print(lmn_reconstructed)

    np.testing.assert_allclose(lmn, lmn_reconstructed, atol=1e-10)

    # Near south pole
    pointing_south_pole = ac.ICRS(0 * au.deg, -90 * au.deg)
    sources = lmn_to_icrs(lmn, time, pointing_south_pole)
    print(sources)

    lmn_reconstructed = icrs_to_lmn(sources, time, pointing_south_pole)
    print(lmn_reconstructed)

    np.testing.assert_allclose(lmn, lmn_reconstructed, atol=1e-10)


def test_earth_location_to_enu():
    antennas = ac.EarthLocation.of_site('vla')
    array_location = ac.EarthLocation.of_site('vla')
    time = at.Time('2000-01-01T00:00:00', format='isot')
    enu = earth_location_to_enu(antennas, array_location, time)
    assert np.linalg.norm(enu) < 6400 * au.km

    enu_frame = ENU(location=ac.EarthLocation.of_site('vla'), obstime=time)
    n = 500
    antennas = ac.SkyCoord(
        east=np.random.uniform(size=(n,), low=-5, high=5) * au.km,
        north=np.random.uniform(size=(n,), low=-5, high=5) * au.km,
        up=np.random.uniform(size=(n,), low=-5, high=5) * au.km,
        frame=enu_frame
    ).transform_to(ac.ITRS).earth_location
    enu = earth_location_to_enu(antennas, array_location, time)
    # print(enu)

    dist = np.linalg.norm(enu[:, None, :] - enu[None, :, :], axis=-1)
    assert np.all(dist < np.sqrt(3) * 10 * au.km)

    # Test earth cetnre
    earth_centre = ac.GCRS(0 * au.deg, 0 * au.deg, 0 * au.km).transform_to(ac.ITRS()).earth_location
    enu = earth_location_to_enu(earth_centre, array_location, time)
    assert np.all(np.abs(enu) < 2e-5 * au.m)


def test_icrs_to_enu():
    sources = ac.ICRS(0 * au.deg, 90 * au.deg)
    array_location = ac.EarthLocation.of_site('vla')
    time = at.Time('2000-01-01T00:00:00', format='isot')
    enu = icrs_to_enu(sources, array_location, time)
    print(enu)
    np.testing.assert_allclose(np.linalg.norm(enu), 1.)

    reconstruct_sources = enu_to_icrs(enu, array_location, time)
    print(reconstruct_sources)
    np.testing.assert_allclose(sources.separation(reconstruct_sources).deg, 0., atol=1e-6)


def test_enu_to_icrs():
    enu = np.array([[0, 1, 0], [0, 0, 1]]) * au.km
    array_location = ac.EarthLocation.of_site('vla')
    time = at.Time('2000-01-01T00:00:00', format='isot')
    sources = enu_to_icrs(enu, array_location, time)
    print(sources)
    # np.testing.assert_allclose(np.linalg.norm(sources.cartesian.xyz, axis=-1), 1.)
    reconstruct_enu = icrs_to_enu(sources, array_location, time)
    print(reconstruct_enu)
    np.testing.assert_allclose(enu, reconstruct_enu, atol=1e-6)


# @pytest.mark.parametrize('offset_dt', [1 * au.hour,
#                                        1 * au.day,
#                                        2 * au.day,
#                                        1 * au.year])
def test_lmn_to_icrs_over_year():
    mvec = lvec = np.linspace(-0.99, 0.99, 100) * au.dimensionless_unscaled

    M, L = np.meshgrid(mvec, lvec, indexing='ij')
    R = np.sqrt(L ** 2 + M ** 2)

    lmn = np.stack([L, M, np.sqrt(1 - L ** 2 - M ** 2)], axis=-1)

    phase_tracking = ac.ICRS(0 * au.deg, 0 * au.deg)
    time0 = at.Time("2021-01-01T00:00:00", scale='utc')
    icrs0 = lmn_to_icrs(lmn, time0, phase_tracking)
    for offset_dt in np.linspace(0., 1., 10) * au.year:
        time1 = time0 + offset_dt
        icrs1 = lmn_to_icrs(lmn, time1, phase_tracking)
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
    mvec = lvec = np.linspace(-0.99, 0.99, 100) * au.dimensionless_unscaled

    M, L = np.meshgrid(mvec, lvec, indexing='ij')

    lmn = np.stack([L, M, np.sqrt(1 - L ** 2 - M ** 2)], axis=-1)

    phase_tracking = ac.ICRS(0 * au.deg, 0 * au.deg)
    time0 = at.Time("2021-01-01T00:00:00", scale='utc')
    icrs0 = lmn_to_icrs(lmn, time0, phase_tracking)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    for i, offset_dt in enumerate([1 * au.hour, 10 * au.day, 180 * au.day, 1 * au.year]):
        ax = axs.flatten()[i]
        time1 = time0 + offset_dt
        icrs1 = lmn_to_icrs(lmn, time1, phase_tracking)
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
    phase_tracking = ac.ICRS(0 * au.deg, 0 * au.deg)
    distance = 0.1 * au.deg
    time = at.Time("2021-01-01T00:00:00", scale='utc')
    # Offset north
    dnorth_icrs = ac.ICRS(
        *offset_by(lon=phase_tracking.ra, lat=phase_tracking.dec, posang=0 * au.deg, distance=distance))
    lmn_dnorth = icrs_to_lmn(dnorth_icrs, time, phase_tracking)
    m_dnorth = lmn_dnorth[1]
    print('North', dnorth_icrs)
    print(lmn_dnorth)
    assert m_dnorth > 0
    # Offset east
    deast_icrs = ac.ICRS(
        *offset_by(lon=phase_tracking.ra, lat=phase_tracking.dec, posang=90 * au.deg, distance=distance))
    lmn_deast = icrs_to_lmn(deast_icrs, time, phase_tracking)
    l_deast = lmn_deast[0]
    print('East', deast_icrs)
    print(lmn_deast)
    assert l_deast > 0
    # Offset south
    dsouth_icrs = ac.ICRS(
        *offset_by(lon=phase_tracking.ra, lat=phase_tracking.dec, posang=180 * au.deg, distance=distance))
    lmn_dsouth = icrs_to_lmn(dsouth_icrs, time, phase_tracking)
    m_dsouth = lmn_dsouth[1]
    print('South', dsouth_icrs)
    print(lmn_dsouth)
    assert m_dsouth < 0
    # Offset west
    dwest_icrs = ac.ICRS(
        *offset_by(lon=phase_tracking.ra, lat=phase_tracking.dec, posang=270 * au.deg, distance=distance))
    lmn_dwest = icrs_to_lmn(dwest_icrs, time, phase_tracking)
    l_dwest = lmn_dwest[0]
    print('West', dwest_icrs)
    print(lmn_dwest)
    assert l_dwest < 0
