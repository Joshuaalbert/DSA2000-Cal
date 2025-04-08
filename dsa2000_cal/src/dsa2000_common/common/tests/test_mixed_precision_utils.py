import jax
import jax.numpy as jnp
import numpy as np
from jax import numpy as jnp

from dsa2000_common.common.mixed_precision_utils import mp_policy, ComplexMP, kron_product_2x2_complex_mp
from dsa2000_common.common.vec_utils import kron_product_2x2


def test_mp_policy():
    print(mp_policy)
    assert mp_policy.length_dtype == jnp.float64
    assert mp_policy.length_dtype == np.float64


def test_mp_policy_np_stays_np():
    x = np.array([0], np.complex64)
    assert isinstance(mp_policy.cast_to_vis(x), np.ndarray)


def test_complex32_dunders():
    def assert_allclose(x, y):
        np.testing.assert_allclose(x.real, y.real, atol=1e-6)
        np.testing.assert_allclose(x.imag, y.imag, atol=1e-6)

    a = b = jax.lax.complex(jnp.arange(5).astype(np.float32), 5 + jnp.arange(5).astype(np.float32))
    _a = ComplexMP(a.real, a.imag)
    _b = ComplexMP(b.real, b.imag)

    assert_allclose(a + b, (_a + _b).complex())
    assert_allclose(a - b, (_a - _b).complex())
    assert_allclose(a * b, (_a * _b).complex())
    assert_allclose(a / b, (_a / _b).complex())

    # mixed
    assert_allclose((_a + b).complex(), (_a + _b).complex())
    assert_allclose((_a - b).complex(), (_a - _b).complex())
    assert_allclose((_a * b).complex(), (_a * _b).complex())
    assert_allclose((_a / b).complex(), (_a / _b).complex())

    # reversed
    assert_allclose((a + _b).complex(), (_a + _b).complex())
    assert_allclose((a - _b).complex(), (_a - _b).complex())
    assert_allclose((a * _b).complex(), (_a * _b).complex())
    assert_allclose((a / _b).complex(), (_a / _b).complex())

    a = jax.lax.complex(jnp.arange(5).astype(np.float32), 5 + jnp.arange(5).astype(np.float32))
    b = jnp.arange(5).astype(np.float32)
    _a = ComplexMP(a.real, a.imag)
    _b = ComplexMP(b.real, 0 * b.imag)

    assert_allclose(a + b, (_a + _b).complex())
    assert_allclose(a - b, (_a - _b).complex())
    assert_allclose(a * b, (_a * _b).complex())

    # mixed
    assert_allclose((_a + b).complex(), (_a + _b).complex())
    assert_allclose((_a - b).complex(), (_a - _b).complex())
    assert_allclose((_a * b).complex(), (_a * _b).complex())

    # reversed
    assert_allclose((a + _b).complex(), (_a + _b).complex())
    assert_allclose((a - _b).complex(), (_a - _b).complex())
    assert_allclose((a * _b).complex(), (_a * _b).complex())

    assert_allclose(a / b, (_a / _b).complex())
    assert_allclose((_a / b).complex(), (_a / _b).complex())
    assert_allclose((a / _b).complex(), (_a / _b).complex())


def test_kron_product():
    def _get_array(key, shape):
        key1, key2 = jax.random.split(key)
        return ComplexMP(real=jax.random.normal(key1, shape), imag=jax.random.normal(key2, shape))

    g1 = _get_array(jax.random.PRNGKey(0), (3, 4, 2, 2))
    g2 = _get_array(jax.random.PRNGKey(1), (3, 4, 2, 2))
    vis = _get_array(jax.random.PRNGKey(2), (4, 2, 2))

    np.testing.assert_allclose(
        kron_product_2x2(g1.complex(), vis.complex(), g2.complex()),
        kron_product_2x2_complex_mp(g1, vis, g2).complex(),
        atol=0.0098
    )

    g1 *= 0.
    g2 *= 0.
    vis *= 0.

    np.testing.assert_allclose(
        kron_product_2x2(g1.complex(), vis.complex(), g2.complex()),
        kron_product_2x2_complex_mp(g1, vis, g2).complex(),
        atol=0.0
    )
