import jax
import jax.numpy as jnp

from dsa2000_cal.solvers.multi_step_lm import lm_solver


@jax.jit
def fit_timings(n, t):
    """
    Fit timings from benchmarks to: t(n) = a*n^b + c

    Args:
        n: [N] size >= 1
        t: [N] timings > 0

    Returns:
        a,b,c
    """

    def transform_params(params):
        log_a, log_b, log_c = params
        return jnp.exp(log_a), jnp.exp(log_b), jnp.exp(log_c)

    def residual_fn(params, n, t):
        a, b, c = transform_params(params)
        t_est = a * n ** b + c
        return t - t_est

    x0 = (jnp.log(jnp.mean(t / n)), jnp.log(jnp.asarray(1.)), jnp.log(jnp.min(t)))
    params, diagnostics = lm_solver(residual_fn, x0, args=(n, t))
    return transform_params(params)
