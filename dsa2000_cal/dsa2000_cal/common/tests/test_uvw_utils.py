import itertools

import numpy as np
import pytest
from astropy import coordinates as ac, units as au, time as at
from jax import numpy as jnp
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.coord_utils import earth_location_to_uvw, icrs_to_lmn, lmn_to_icrs
from dsa2000_cal.common.uvw_utils import perley_lmn_from_icrs, perley_icrs_from_lmn, celestial_to_cartesian, compute_uvw


@pytest.mark.parametrize('ra0', [0, np.pi / 2, np.pi])
@pytest.mark.parametrize('dec0', [0, np.pi / 2, -np.pi / 2])
@pytest.mark.parametrize('ra', [0, np.pi / 2, np.pi])
@pytest.mark.parametrize('dec', [0, np.pi / 2, -np.pi / 2])
def test_perley_icrs_from_lmn(ra, dec, ra0, dec0):
    l, m, n = perley_lmn_from_icrs(ra, dec, ra0, dec0)

    _ra, _dec = perley_icrs_from_lmn(l, m, n, ra0, dec0)
    np.testing.assert_allclose(ra, _ra)
    np.testing.assert_allclose(dec, _dec)


@pytest.mark.parametrize('ra', [0, 90, 180])
@pytest.mark.parametrize('dec', [0, 90, -90])
def test_celestial_to_cartesian(ra, dec):
    sources = ac.ICRS(ra * au.deg, dec * au.deg)
    np.testing.assert_allclose(celestial_to_cartesian(sources.ra.rad, sources.dec.rad), sources.cartesian.xyz.value.T,
                               atol=1e-7)


@pytest.mark.parametrize('ra0', [0, 90, 180])
@pytest.mark.parametrize('dec0', [0, 90, -90])
def test_lm_to_k_bcrs(ra0, dec0):
    l = m = jnp.asarray(0.)
    n = jnp.sqrt(1. - jnp.square(l) - jnp.square(m))
    ra, dec = perley_icrs_from_lmn(l=l, m=m, n=n, ra0=ra0, dec0=dec0)
    K_bcrs = celestial_to_cartesian(ra, dec)
    x = ac.ICRS(ra=ra * au.rad, dec=dec * au.rad)
    np.testing.assert_allclose(x.cartesian.xyz.value, K_bcrs, atol=1e-8)


@pytest.mark.parametrize('with_autocorr', [True, False])
def test_uvw(with_autocorr):
    times = at.Time.now().reshape((1,))
    array_location = ac.EarthLocation.of_site('vla')
    antennas = ENU(
        east=[0, 10] * au.km,
        north=[0, 0] * au.km,
        up=[0, 0] * au.km,
        location=array_location,
        obstime=times[0]
    )
    antennas = antennas.transform_to(ac.ITRS(obstime=times[0])).earth_location

    phase_centre = ENU(east=0, north=0, up=1, location=array_location, obstime=times[0]).transform_to(ac.ICRS())
    uvw = compute_uvw(
        antennas=antennas,
        times=times,
        phase_center=phase_centre,
        verbose=True,
        with_autocorr=with_autocorr
    )

    uvw_other = earth_location_to_uvw(
        antennas=antennas[None, :],
        obs_time=times[:, None],
        phase_tracking=phase_centre
    )
    if with_autocorr:
        antenna_1, antenna_2 = jnp.asarray(list(itertools.combinations_with_replacement(range(len(antennas)), 2))).T
    else:
        antenna_1, antenna_2 = jnp.asarray(list(itertools.combinations(range(len(antennas)), 2))).T
    uvw_other = uvw_other[:, antenna_2, :] - uvw_other[:, antenna_1, :]

    if with_autocorr:
        np.testing.assert_allclose(uvw_other[0, 1, 0], 10 * au.km, atol=1 * au.m)
    else:
        np.testing.assert_allclose(uvw_other[0, 0, 0], 10 * au.km, atol=1 * au.m)
    np.testing.assert_allclose(uvw, uvw_other, atol=0.9 * au.m)


def test_icrs_to_lmn_against_perley():
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    time = at.Time("2021-01-01T00:00:00", scale='utc')

    source = ac.ICRS(ra=[0., 1., 10] * au.deg, dec=[0., 1., 10] * au.deg)
    lmn_ours = icrs_to_lmn(source, phase_tracking=phase_tracking)
    lmn_perley = np.stack(
        perley_lmn_from_icrs(source.ra.rad, source.dec.rad, phase_tracking.ra.rad, phase_tracking.dec.rad), axis=-1)
    np.testing.assert_allclose(lmn_ours, lmn_perley, atol=2e-5)


def test_lmn_to_icrs_against_perley():
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    time = at.Time("2021-01-01T00:00:00", scale='utc')

    lmn = np.asarray([
        [0., 0., 1.],
        [0.1, 0.1, np.sqrt(1. - 0.1 ** 2 - 0.1 ** 2)],
        [0.2, 0.2, np.sqrt(1. - 0.2 ** 2 - 0.2 ** 2)],
        [0.5, 0.5, np.sqrt(1. - 0.5 ** 2 - 0.5 ** 2)],
    ]) * au.dimensionless_unscaled
    icrs_ours = lmn_to_icrs(lmn, phase_tracking=phase_tracking)
    ra, dec = perley_icrs_from_lmn(lmn[..., 0], lmn[..., 1], lmn[..., 2], phase_tracking.ra.rad, phase_tracking.dec.rad)
    icrs_perley = ac.ICRS(ra * au.rad, dec * au.rad)

    def wrap(angle):
        return np.angle(np.exp(1j * angle.to('rad').value)) * au.rad

    np.testing.assert_allclose(wrap(icrs_ours.ra), wrap(icrs_perley.ra), atol=1e-4)
    np.testing.assert_allclose(icrs_ours.dec, icrs_perley.dec, atol=3e-3)
