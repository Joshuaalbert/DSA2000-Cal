import numpy as np
from jax import numpy as jnp

from dsa2000_common.common.parallactic_utils import parallactic_angle, standard_parallactic


def test_parallactic_angle():
    # Example:  lat = 35°, dec = 20°, H = +30°
    deg = jnp.pi / 180
    lat, dec, H = 35 * deg, 20 * deg, 30 * deg
    R = 1.0  # scale irrelevant
    θ = H  # RA = 0 for simplicity
    x = R * jnp.cos(lat) * jnp.cos(θ)
    y = R * jnp.cos(lat) * jnp.sin(θ)
    z = R * jnp.sin(lat)

    ra = 0.0  # RA of source
    u = -x * jnp.sin(ra) + y * jnp.cos(ra)
    v = -x * jnp.cos(ra) * jnp.sin(dec) - y * jnp.sin(ra) * jnp.sin(dec) + z * jnp.cos(dec)
    w = x * jnp.cos(ra) * jnp.cos(dec) + y * jnp.sin(ra) * jnp.cos(dec) + z * jnp.sin(dec)

    chi = parallactic_angle(u, v, w, dec, lat)
    chi_expected = standard_parallactic(H, lat, dec)
    np.testing.assert_allclose(chi, chi_expected, atol=1e-6)
