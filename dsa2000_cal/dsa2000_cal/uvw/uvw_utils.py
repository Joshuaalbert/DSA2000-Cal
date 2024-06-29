import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights, apply_interp


# from astropy.coordinates import solar_system_ephemeris
# solar_system_ephemeris.set('jpl')

@dataclasses.dataclass(eq=False)
class InterpolatedArray:
    times: jax.Array  # [N]
    values: jax.Array  # [..., N, ...] `axis` has N elements

    axis: int = 0
    regular_grid: bool = False

    def __post_init__(self):

        if len(np.shape(self.times)) != 1:
            raise ValueError(f"Times must be 1D, got {np.shape(self.times)}.")

        def _assert_shape(x):
            if np.shape(x)[self.axis] != np.size(self.times):
                raise ValueError(f"Input values must have time length on `axis` dimension, got {np.shape(x)}.")

        jax.tree.map(_assert_shape, self.values)

        self.times, self.values = jax.tree.map(jnp.asarray, (self.times, self.values))

    @property
    def shape(self):
        return jax.tree.map(lambda x: np.shape(x)[:self.axis] + np.shape(x)[self.axis + 1:], self.values)

    def __call__(self, time: jax.Array) -> jax.Array:
        """
        Interpolate at time based on input times.

        Args:
            time: time to evaluate at.

        Returns:
            value at given time
        """
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(time, self.times, regular_grid=self.regular_grid)
        return jax.tree.map(lambda x: apply_interp(x, i0, alpha0, i1, alpha1, axis=self.axis), self.values)


@pytest.mark.parametrize('regular_grid', [True, False])
def test_interpolated_array(regular_grid: bool):
    # scalar time
    times = jnp.linspace(0, 10, 100)
    values = jnp.sin(times)
    interp = InterpolatedArray(times, values, regular_grid=regular_grid)
    assert interp(5.).shape == ()
    np.testing.assert_allclose(interp(5.), jnp.sin(5), atol=2e-3)

    # vector time
    assert interp(jnp.array([5., 6.])).shape == (2,)
    np.testing.assert_allclose(interp(jnp.array([5., 6.])), jnp.sin(jnp.array([5., 6.])), atol=2e-3)

    # Now with axis = 1
    times = jnp.linspace(0, 10, 100)
    values = jnp.stack([jnp.sin(times), jnp.cos(times)], axis=0)  # [2, 100]
    interp = InterpolatedArray(times, values, axis=1, regular_grid=regular_grid)
    assert interp(5.).shape == (2,)
    np.testing.assert_allclose(interp(5.), jnp.array([jnp.sin(5), jnp.cos(5)]), atol=2e-3)

    # Vector
    assert interp(jnp.array([5., 6., 7.])).shape == (2, 3)
    np.testing.assert_allclose(interp(jnp.array([5., 6., 7.])),
                               jnp.stack([jnp.sin(jnp.array([5., 6., 7.])), jnp.cos(jnp.array([5., 6., 7.]))],
                                         axis=0),
                               atol=2e-3)


def norm(x, axis=-1, keepdims: bool = False):
    return jnp.sqrt(norm2(x, axis, keepdims))


def norm2(x, axis=-1, keepdims: bool = False):
    return jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims)


def perley_icrs_from_lmn(l, m, n, ra0, dec0):
    dec = jnp.arcsin(m * jnp.cos(dec0) + n * jnp.sin(dec0))
    ra = ra0 + jnp.arctan2(l, n * jnp.cos(dec0) - m * jnp.sin(dec0))
    return ra, dec


def perley_lmn_from_icrs(alpha, dec, alpha0, dec0):
    dra = alpha - alpha0

    l = jnp.cos(dec) * jnp.sin(dra)
    m = jnp.sin(dec) * jnp.cos(dec0) - jnp.cos(dec) * jnp.sin(dec0) * jnp.cos(dra)
    n = jnp.sin(dec) * jnp.sin(dec0) + jnp.cos(dec) * jnp.cos(dec0) * jnp.cos(dra)
    return l, m, n


def celestial_to_cartesian(ra, dec):
    x = jnp.cos(ra) * jnp.cos(dec)
    y = jnp.sin(ra) * jnp.cos(dec)
    z = jnp.sin(dec)
    return jnp.stack([x, y, z], axis=-1)
