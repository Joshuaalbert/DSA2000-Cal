import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_cal.common.pure_callback_utils import construct_threaded_pure_callback, _build_batch_shape_determiner, \
    construct_threaded_callback


def test_construct_threaded_pure_callback():
    def add_kernel(x, y, z):
        assert x.shape == ()
        assert y.shape == ()
        assert z.shape == ()
        return x + y + z

    x = jnp.ones((4,), dtype=jnp.float32)
    y = jnp.ones((5,), dtype=jnp.float32)
    z = jnp.ones((), dtype=jnp.float32)

    cb = construct_threaded_pure_callback(
        add_kernel,
        jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        0, 0, 0
    )

    cb_vmap = jax.vmap(jax.vmap(cb, in_axes=(None, 0, None)), in_axes=(0, None, None))
    assert cb_vmap(x, y, z).shape == (4, 5)

    def add_kernel(x, y, z):
        if z is None:
            return x + y
        return x + y + z

    cb = construct_threaded_pure_callback(
        add_kernel,
        jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        0, 0, 0
    )

    cb_vmap = jax.vmap(jax.vmap(cb, in_axes=(None, 0, None)), in_axes=(0, None, None))
    cb_jit = jax.jit(cb_vmap)
    assert cb_jit(x, y, z).shape == (4, 5)
    assert cb_jit(x, y, None).shape == (4, 5)


def test_build_batch_shape_determiner():
    x = np.ones((4, 5, 1))
    y = np.ones((4, 5, 19, 3))
    z = np.ones((4, 5))
    batch_shape_determiner = _build_batch_shape_determiner(1, 2, 0)
    assert batch_shape_determiner(x, y, z) == (4, 5)


def test_construct_threaded_callback():
    def cb_kernel(x, y):
        assert np.shape(x) == ()
        assert np.shape(y) == ()
        return np.sin(x) + np.cos(y)

    cb = construct_threaded_callback(cb_kernel, 0, 0)
    x = np.ones((2, 3))
    y = np.ones((2, 3))

    res = cb(x, y)

    np.testing.assert_allclose(res, np.sin(x) + np.cos(y))


def test_construct_threaded_callback_nested():
    def cb_kernel(x, y):
        def cb_inner(a):
            assert np.shape(a) == ()
            return a * 2

        assert np.shape(x) == (3,)
        assert np.shape(y) == (3,)
        a = np.sin(x) + np.cos(y)
        cb = construct_threaded_callback(cb_inner, 0)
        return cb(a)


    cb = construct_threaded_callback(cb_kernel, 1, 1)
    x = np.ones((2, 3))
    y = np.ones((2, 3))

    res = cb(x, y)

    np.testing.assert_allclose(res, (np.sin(x) + np.cos(y))*2)
