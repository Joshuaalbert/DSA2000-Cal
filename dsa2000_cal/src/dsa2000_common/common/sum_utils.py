import jax
from jax import numpy as jnp


def kahan_sum(accum_fn, init_accumulate, xs):
    def body_fn(carry, x):
        accumulate, error_accumulate = carry
        delta = accum_fn(x)
        y = jax.tree.map(jax.lax.sub, delta, error_accumulate)
        t = jax.tree.map(jax.lax.add, accumulate, y)
        error_accumulate = jax.tree.map(jax.lax.sub, jax.tree.map(jax.lax.sub, t, accumulate), y)
        accumulate = t
        return (accumulate, error_accumulate), None

    init_error_accumulate = jax.tree.map(jnp.zeros_like, init_accumulate)
    (accumulate, error_accumulate), _ = jax.lax.scan(body_fn, (init_accumulate, init_error_accumulate), xs)
    return accumulate, error_accumulate
