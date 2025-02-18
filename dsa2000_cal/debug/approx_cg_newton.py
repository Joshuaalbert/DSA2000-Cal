import time

import jax
import jax.numpy as jnp


def hvp_linearize(f, params):
    # Compute the gradient function and linearize it at params
    grad_f = jax.grad(f)
    _, lin_fun = jax.linearize(grad_f, params)
    # lin_fun is a function that computes the JVP of grad_f at params
    return lin_fun  # This function computes HVPs for different v


def hvp_forward_over_reverse(f, params):
    def hvp(v):
        return jax.jvp(jax.grad(f), (params,), (v,))[1]

    return hvp


def hvp_reverse_over_reverse(f, params):
    def hvp(v):
        return jax.grad(lambda y: jnp.vdot(jax.grad(f)(y), v))(params)

    return hvp


def hvp_reverse_over_forward(f, params):
    def hvp(v):
        jvp_fun = lambda params: jax.jvp(f, (params,), (v,))[1]
        return jax.grad(jvp_fun)(params)

    return hvp


# compare performnace of different methods

def test_performance():
    def f(x):
        return jnp.sum(x ** 2) + jnp.sum(jnp.roll(x, 1) ** 2) + jnp.cos(x[0]) * jnp.sum(x[1:])

    n = 10
    params = jnp.ones(n)

    v = jnp.ones(n)  # vector for Hessian-vector product

    for method in [hvp_linearize, hvp_forward_over_reverse, hvp_reverse_over_reverse, hvp_reverse_over_forward]:
        print(f"Method: {method.__name__}")

        def multi_apply(params, v):
            hvp = method(f, params)
            results = [v]
            for i in range(100):
                results.append(hvp(v * results[-1]))
            return results[-1]

        hvp_jit = jax.jit(multi_apply).lower(params, v).compile()
        t0 = time.time()
        for _ in range(1):
            jax.block_until_ready(hvp_jit(params, v))
        t1 = time.time()
        print(f"Time: {t1 - t0}")
