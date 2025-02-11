import jax
import jax.numpy as jnp


@jax.jit
@jax.vmap
def callback(*args):
    def cb(*args):
        return args

    return jax.pure_callback(cb,
                             jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), args), *args,
                             vmap_method='broadcast_all')


def main():
    print(callback(jnp.ones(2), jnp.ones(2)))


if __name__ == '__main__':
    main()
