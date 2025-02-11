import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class Pytree:
    x: jax.Array
    values: jax.Array
    axis: int

    def __post_init__(self):
        assert np.shape(self.values)[self.axis] == np.size(self.x)


def pytree_flatten(p: Pytree):
    return (
        [p.x, p.values],
        (p.axis,)
    )


def pytree_unflatten(aux_data, children):
    x, values = children
    axis, = aux_data
    return Pytree(x, values, axis)


jax.tree_util.register_pytree_node(
    Pytree,
    pytree_flatten,
    pytree_unflatten
)


def f(p: Pytree):
    return p


def main():
    x = jnp.ones(5)
    values = jnp.ones((10, 5))
    axis = -1
    p = Pytree(x, values, axis)
    jax.block_until_ready(f(p))
    jax.block_until_ready(jax.jit(f)(p))

    jax.jit(f).lower(p).compile()


if __name__ == '__main__':
    main()
