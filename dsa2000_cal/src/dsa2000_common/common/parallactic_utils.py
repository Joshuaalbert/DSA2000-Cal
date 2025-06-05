import jax.numpy as jnp


def parallactic_angle(u, v, w, dec, lat):
    """
    Parallactic angle χ (radians)

    Args:
        u: U component of the antenna position vector (in the tangent plane)
        v: V component of the antenna position vector (in the tangent plane)
        w: W component of the antenna position vector (in the tangent plane)
        dec: Declination of the source (radians)
        lat: Latitude of the observer (radians)

    Returns:
        χ: Parallactic angle (radians) in the range (-π, π]
    """

    r = w / u  # handy ratio
    tanlat = jnp.tan(lat)
    cosdec = jnp.cos(dec)
    sindec = jnp.sin(dec)

    # Solve quadratic A s² + B s + C = 0 for s = sin H
    A = cosdec * cosdec + r * r
    B = -2.0 * r * tanlat * sindec
    C = tanlat * tanlat * sindec * sindec - cosdec * cosdec
    D = B * B - 4.0 * A * C  # discriminant
    if D < 0:
        D = 0.0  # numerical safety
    root = jnp.sqrt(D)

    # pick the root whose sign matches that of u  (because u ∝ sin H)
    s1 = (-B + root) / (2.0 * A)
    s2 = (-B - root) / (2.0 * A)
    s = jnp.where(jnp.sign(s1) == jnp.sign(u), s1, s2)
    s = jnp.clip(s, -1., 1.)  # clamp to [-1, 1] for safety

    # corresponding cos H
    c = (r * s - tanlat * sindec) / cosdec
    c = jnp.clip(c, -1., 1.)  # clamp to [-1, 1] for safety

    # finally χ
    denom = tanlat * cosdec - sindec * c
    chi = jnp.arctan2(s, denom)  # full-quadrant result

    on_meridean = jnp.abs(u) < 1e-12
    chi = jnp.where(
        on_meridean,
        jnp.where(v >= 0.0, 0.0, jnp.pi),  # meridian case)
        chi
    )  # otherwise, use the computed value
    return chi


def standard_parallactic(H, lat, dec):
    """
    Standard parallactic angle calculation.

    Args:
        H: Hour angle (radians)
        lat: Latitude of the observer (radians)
        dec: Declination of the source (radians)

    Returns:
        the parallactic angle (radians) in the range (-π, π]
    """
    return jnp.atan2(
        jnp.sin(H),
        jnp.tan(lat) * jnp.cos(dec) - jnp.sin(dec) * jnp.cos(H)
    )
