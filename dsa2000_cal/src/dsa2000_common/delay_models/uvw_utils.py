import jax.lax
import jax.numpy as jnp

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


def geometric_uvw_from_gcrs(x_gcrs: FloatArray, ra0, dec0):
    x, y, z = x_gcrs[..., 0], x_gcrs[..., 1], x_gcrs[..., 2]
    u = -x * jnp.sin(ra0) + y * jnp.cos(ra0)
    v = -x * jnp.sin(dec0) * jnp.cos(ra0) - y * jnp.sin(dec0) * jnp.sin(ra0) + z * jnp.cos(dec0)
    w = x * jnp.cos(dec0) * jnp.cos(ra0) + y * jnp.cos(dec0) * jnp.sin(ra0) + z * jnp.sin(dec0)
    return jnp.stack([u, v, w], axis=-1)  # [..., 3]


def gcrs_from_geometric_uvw(uvw: FloatArray, ra0, dec0):
    u, v, w = uvw[..., 0], uvw[..., 1], uvw[..., 2]

    x = (
            -jnp.sin(ra0) * u
            - jnp.sin(dec0) * jnp.cos(ra0) * v
            + jnp.cos(dec0) * jnp.cos(ra0) * w
    )
    y = (
            jnp.cos(ra0) * u
            - jnp.sin(dec0) * jnp.sin(ra0) * v
            + jnp.cos(dec0) * jnp.sin(ra0) * w
    )
    z = (
            # 0 * u
            jnp.cos(dec0) * v
            + jnp.sin(dec0) * w
    )

    return jnp.stack([x, y, z], axis=-1)
