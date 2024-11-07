from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.vmap, in_axes=(0, None, None))
@partial(jax.vmap, in_axes=(None, 0, None))
def add_vmapped(x, y, z):
    return x + y + z


@partial(jax.vmap, in_axes=(0, None, None))
@partial(jax.vmap, in_axes=(None, 0, None))
def cb_no_vec(x, y, z):
    print(x.shape)
    def add(x, y, z):
        assert x.shape == ()
        assert y.shape == ()
        assert z.shape == ()
        return x + y + z

    return jax.pure_callback(add, jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype), x, y, z,vmap_method='broadcast_all')


def convert_to_ufunc(f, tile: bool = True):
    f = jax.custom_batching.custom_vmap(f)

    @f.def_vmap
    def rule(axis_size, in_batched, *args):
        batched_args = jax.tree.map(
            lambda x, b: x if b else jax.lax.broadcast(x, ((axis_size if tile else 1),)), args,
            tuple(in_batched))
        out = f(*batched_args)
        out_batched = jax.tree.map(lambda _: True, out)
        return out, out_batched

    return f


@partial(jax.vmap, in_axes=(0, None, None))
@partial(jax.vmap, in_axes=(None, 0, None))
@convert_to_ufunc
def cb_vec_tiled(x, y, z):
    def add(x, y, z):
        assert x.shape == (4, 5)
        assert y.shape == (4, 5)
        assert z.shape == (4, 5)
        return x + y + z

    return jax.pure_callback(add, jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype), x, y, z, vectorized=True)


@partial(jax.vmap, in_axes=(0, None, None))
@partial(jax.vmap, in_axes=(None, 0, None))
@partial(convert_to_ufunc, tile=False)
def cb_vec_untiled(x, y, z):
    def add(x, y, z):
        assert x.shape == (4, 1)
        assert y.shape == (1, 5)
        assert z.shape == (1, 1)
        return x + y + z

    return jax.pure_callback(add, jax.ShapeDtypeStruct(shape=jnp.broadcast_shapes(x.shape, y.shape), dtype=x.dtype), x,
                             y, z, vectorized=True)




def cb(x, y, z):
    def add(x, y, z):
        assert x.shape == (4, 5)
        assert y.shape == (4, 5)
        assert z.shape == ()
        return x + y + z

    return jax.pure_callback(add, jax.ShapeDtypeStruct(shape=jnp.broadcast_shapes(x.shape, y.shape), dtype=x.dtype), x,
                             y, z, vmap_method='broadcast_all')


if __name__ == '__main__':
    x = jnp.arange(4, dtype=jnp.float32)
    y = jnp.arange(5, dtype=jnp.float32)
    z = jnp.array(1, dtype=jnp.float32)

    assert add_vmapped(x, y, z).shape == (4, 5)
    assert cb_no_vec(x, y, z).shape == (4, 5)
    assert cb_vec_tiled(x, y, z).shape == (4, 5)
    assert cb_vec_untiled(x, y, z).shape == (4, 5)

    assert jax.vmap(jax.vmap(convert_to_ufunc(partial(cb, z=z)), in_axes=(None, 0)), in_axes=(0, None))(x, y).shape == (4, 5)


