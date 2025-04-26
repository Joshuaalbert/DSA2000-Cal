import jax.random
import numpy as np
import pytest
from astropy import coordinates as ac, units as au, time as at
from jax import numpy as jnp

from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.coord_utils import icrs_to_lmn, lmn_to_icrs
from dsa2000_common.delay_models.uvw_utils import perley_lmn_from_icrs, perley_icrs_from_lmn, celestial_to_cartesian, \
    cartesian_to_celestial, geometric_uvw_from_gcrs, gcrs_from_geometric_uvw


@pytest.mark.parametrize('ra0', [0, np.pi / 2, np.pi])
@pytest.mark.parametrize('dec0', [0, np.pi / 2, -np.pi / 2])
@pytest.mark.parametrize('ra', [0, np.pi / 2, np.pi])
@pytest.mark.parametrize('dec', [0, np.pi / 2, -np.pi / 2])
def test_perley_icrs_from_lmn(ra, dec, ra0, dec0):
    l, m, n = perley_lmn_from_icrs(ra, dec, ra0, dec0)

    _ra, _dec = perley_icrs_from_lmn(l, m, n, ra0, dec0)
    np.testing.assert_allclose(ra, _ra)
    np.testing.assert_allclose(dec, _dec)


@pytest.mark.parametrize('ra', [0, 45, 90, 180, -45, 270, -270])
@pytest.mark.parametrize('dec', [0, 45, 90, -90, -45])
def test_celestial_to_cartesian(ra, dec):
    sources = ac.ICRS(ra * au.deg, dec * au.deg)
    np.testing.assert_allclose(celestial_to_cartesian(sources.ra.rad, sources.dec.rad), sources.cartesian.xyz.value.T,
                               atol=1e-7)


@pytest.mark.parametrize('x', [1, 0, 0])
@pytest.mark.parametrize('y', [0, 1, 0])
@pytest.mark.parametrize('z', [0, 0, 1])
def test_cartesian_to_celestial(x, y, z):
    source = ac.ICRS().realize_frame(data=ac.CartesianRepresentation(x, y, z, unit=au.dimensionless_unscaled))
    ra, dec, dist = cartesian_to_celestial(x, y, z)
    np.testing.assert_allclose(ra, source.ra.rad)
    np.testing.assert_allclose(dec, source.dec.rad)
    np.testing.assert_allclose(dist, source.distance.value)


@pytest.mark.parametrize('ra0', [0, 90, 180])
@pytest.mark.parametrize('dec0', [0, 90, -90])
def test_lm_to_k_bcrs(ra0, dec0):
    l = m = jnp.asarray(0.)
    n = jnp.sqrt(1. - jnp.square(l) - jnp.square(m))
    ra, dec = perley_icrs_from_lmn(l=l, m=m, n=n, ra0=ra0, dec0=dec0)
    K_bcrs = celestial_to_cartesian(ra, dec)
    x = ac.ICRS(ra=ra * au.rad, dec=dec * au.rad)
    np.testing.assert_allclose(x.cartesian.xyz.value, K_bcrs, atol=1e-5)


def test_icrs_to_lmn_against_perley():
    phase_center = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    time = at.Time("2021-01-01T00:00:00", scale='utc')

    source = ac.ICRS(ra=[0., 1., 10] * au.deg, dec=[0., 1., 10] * au.deg)
    lmn_ours = icrs_to_lmn(source, phase_center=phase_center)
    lmn_perley = np.stack(
        perley_lmn_from_icrs(source.ra.rad, source.dec.rad, phase_center.ra.rad, phase_center.dec.rad), axis=-1)
    np.testing.assert_allclose(lmn_ours, lmn_perley, atol=2e-5)


def test_lmn_to_icrs_against_perley():
    phase_center = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    time = at.Time("2021-01-01T00:00:00", scale='utc')

    lmn = np.asarray([
        [0., 0., 1.],
        [0.1, 0.1, np.sqrt(1. - 0.1 ** 2 - 0.1 ** 2)],
        [0.2, 0.2, np.sqrt(1. - 0.2 ** 2 - 0.2 ** 2)],
        [0.5, 0.5, np.sqrt(1. - 0.5 ** 2 - 0.5 ** 2)],
    ]) * au.dimensionless_unscaled
    icrs_ours = lmn_to_icrs(lmn, phase_center=phase_center)
    ra, dec = perley_icrs_from_lmn(lmn[..., 0], lmn[..., 1], lmn[..., 2], phase_center.ra.rad, phase_center.dec.rad)
    icrs_perley = ac.ICRS(ra * au.rad, dec * au.rad)

    def wrap(angle):
        return np.angle(np.exp(1j * angle.to('rad').value)) * au.rad

    np.testing.assert_allclose(wrap(icrs_ours.ra), wrap(icrs_perley.ra), atol=1e-4)
    np.testing.assert_allclose(icrs_ours.dec, icrs_perley.dec, atol=3e-3)


def test_gcrs_to_icrs():
    import astropy.time as at
    import astropy.coordinates as ac
    import astropy.units as au

    k = ac.ICRS(ra=0 * au.deg, dec=50 * au.deg)
    k_cartesian = k.cartesian.xyz

    t = at.Time('9125-06-10T15:00:00', scale='utc')
    k_gcrs = k.transform_to(ac.GCRS(obstime=t))
    k_gcrs_cartesian = k_gcrs.cartesian.xyz

    print(k_cartesian, k_gcrs_cartesian)


def test_geometric_uvw_from_gcrs():
    ra0, dec0 = 0, 50 * np.pi / 180
    x_gcrs = jax.random.normal(jax.random.PRNGKey(0), shape=(3,))
    np.testing.assert_allclose(geometric_uvw_from_gcrs(x_gcrs, ra0, dec0),
                               _brute_force_geometric_uvw_from_gcrs(x_gcrs, ra0, dec0),
                               atol=1e-6)

    np.testing.assert_allclose(
        gcrs_from_geometric_uvw(geometric_uvw_from_gcrs(x_gcrs, ra0, dec0), ra0, dec0),
        x_gcrs,
        atol=1e-6
    )

    ra0, dec0 = 10 * np.pi / 180, 50 * np.pi / 180
    x_gcrs = 100 * jax.random.normal(jax.random.PRNGKey(0), shape=(3,))
    np.testing.assert_allclose(geometric_uvw_from_gcrs(x_gcrs, ra0, dec0),
                               _brute_force_geometric_uvw_from_gcrs(x_gcrs, ra0, dec0),
                               atol=1e-6)

    np.testing.assert_allclose(
        gcrs_from_geometric_uvw(geometric_uvw_from_gcrs(x_gcrs, ra0, dec0), ra0, dec0),
        x_gcrs,
        atol=1e-6
    )


def _brute_force_geometric_uvw_from_gcrs(x_gcrs: FloatArray, ra0, dec0):
    """
    Convert GCRS coordinates to UVW coordinates assuming no relativistic effects.

    Args:
        x_gcrs: [3] GCRS coordinates
        ra0: tracking center right ascension in radians
        dec0: tracking center declination in radians

    Returns:
        [3] UVW coordinates
    """
    # In strict geometric case GCRS and BCRS are the same, i.e. no relativity effects
    if np.shape(x_gcrs)[-1] != 3:
        raise ValueError("x_gcrs must have shape [..., 3]")

    def delay_fn(l, m, n):
        ra, dec = perley_icrs_from_lmn(l, m, n, ra0, dec0)
        K_bcrs = celestial_to_cartesian(ra, dec)
        return x_gcrs @ K_bcrs

    l = m = jnp.asarray(0.)
    n = jnp.sqrt(1 - l ** 2 - m ** 2)
    # tau = (-?) c * delay = u l + v m + w sqrt(1 - l^2 - m^2) ==> w = tau(l=0, m=0)
    # d/dl tau = u + w l / sqrt(1 - l^2 - m^2) ==> u = d/dl tau(l=0, m=0)
    # d/dm tau = v + w m / sqrt(1 - l^2 - m^2) ==> v = d/dm tau(l=0, m=0)
    w, (u, v) = jax.value_and_grad(delay_fn, argnums=(0, 1))(l, m, n)
    return jnp.stack([u, v, w], axis=-1)  # [3]
