import jax.lax
import jax.numpy as jnp
import numpy as np

from dsa2000_common.common.array_types import FloatArray


# from astropy.coordinates import solar_system_ephemeris
# solar_system_ephemeris.set('jpl')


def norm(x, axis=-1, keepdims: bool = False):
    return jnp.sqrt(norm2(x, axis, keepdims))


def norm2(x, axis=-1, keepdims: bool = False):
    return jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims)


def perley_icrs_from_lmn(l, m, n, ra0, dec0):
    dec = jnp.arcsin(jnp.clip(m * jnp.cos(dec0) + n * jnp.sin(dec0), -1, 1))
    ra = ra0 + jnp.arctan2(l, n * jnp.cos(dec0) - m * jnp.sin(dec0))
    return ra, dec


def perley_lmn_from_icrs(alpha, dec, alpha0, dec0):
    dra = alpha - alpha0

    l = jnp.cos(dec) * jnp.sin(dra)
    m = jnp.sin(dec) * jnp.cos(dec0) - jnp.cos(dec) * jnp.sin(dec0) * jnp.cos(dra)
    n = jnp.sin(dec) * jnp.sin(dec0) + jnp.cos(dec) * jnp.cos(dec0) * jnp.cos(dra)
    return l, m, n


def celestial_to_cartesian(ra, dec, distance=None):
    x = jnp.cos(ra) * jnp.cos(dec)
    y = jnp.sin(ra) * jnp.cos(dec)
    z = jnp.sin(dec)
    if distance is not None:
        x *= distance
        y *= distance
        z *= distance
    return jnp.stack([x, y, z], axis=-1)


def cartesian_to_celestial(x, y, z):
    norm = jnp.sqrt(x * x + y * y + z * z)
    ra = jnp.arctan2(y, x)
    dec = jnp.arcsin(jnp.clip(jax.lax.select(norm == 0, norm, z / norm), -1, 1))
    return ra, dec, norm


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


def geometric_uvw_from_gcrs(x_gcrs: FloatArray, ra0, dec0):
    x, y, z = x_gcrs[..., 0], x_gcrs[..., 1], x_gcrs[..., 2]
    u = -x * jnp.sin(ra0) + y * jnp.cos(ra0)
    v = -x * jnp.sin(dec0) * jnp.cos(ra0) - y * jnp.sin(dec0) * jnp.sin(ra0) + z * jnp.cos(dec0)
    w = x * jnp.cos(dec0) * jnp.cos(ra0) + y * jnp.cos(dec0) * jnp.sin(ra0) + z * jnp.sin(dec0)
    return jnp.stack([u, v, w], axis=-1)  # [..., 3]


def test_geometric_uvw_from_gcrs():
    ra0, dec0 = 0, 50
    x_gcrs = np.random.normal(size=3)
    np.testing.assert_allclose(geometric_uvw_from_gcrs(x_gcrs, ra0, dec0),
                               _brute_force_geometric_uvw_from_gcrs(x_gcrs, ra0, dec0))
