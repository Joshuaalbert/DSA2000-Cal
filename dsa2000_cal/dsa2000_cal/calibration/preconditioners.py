import jax
import jax.numpy as jnp


def hutchisons_diag_estimator(key, matvec, x0, num_samples: int, rv_type: str = "normal"):
    """
    Estimate the diagonal of a linear operator using Hutchinson method.

    Args:
        key: the random key
        matvec: the linear operator
        x0: a pytree of the same structure as the output of matvec, only needs shape and dtype info
        num_samples: the number of samples to use for the estimation
        rv_type: the type of random variable to use, one of "normal", "uniform", "rademacher"

    Returns:
        the estimated diagonal
    """

    def single_sample(key):
        leaves, tree_def = jax.tree.flatten(x0)
        sample_keys = jax.random.split(key, len(leaves))
        sample_keys = jax.tree.unflatten(tree_def, sample_keys)

        def sample(key, shape, dtype):
            if rv_type == "normal":
                return jax.random.normal(key, shape=shape, dtype=dtype)
            elif rv_type == "uniform":
                rv_scale = jnp.sqrt(1. / 3.)
                return jax.random.uniform(key, shape=shape, dtype=dtype, minval=-1., maxval=1.) / rv_scale
            elif rv_type == "rademacher":
                return jnp.where(jax.random.uniform(key, shape=shape) < 0.5, -1., 1.)
            else:
                raise ValueError(f"Unknown rv_type: {rv_type}")

        v = jax.tree.map(lambda key, x: sample(key, x.shape, x.dtype), sample_keys, x0)
        Av = matvec(v)
        return jax.tree.map(lambda x, y: x * y, v, Av)

    keys = jax.random.split(key, num_samples)
    results = jax.vmap(single_sample)(keys)
    return jax.tree.map(lambda y: jnp.mean(y, axis=0), results)


def jacobi_preconditioner(key, matvec, x0, num_samples: int):
    """
    Compute the Jacobi preconditioner for a linear operator.

    Args:
        key: the random key
        matvec: the linear operator
        x0: a pytree of the same structure as the output of matvec, only needs shape and dtype info
        num_samples: the number of samples to use for the estimation

    Returns:
        a function that applies the preconditioner
    """
    diag_est = hutchisons_diag_estimator(key, matvec, x0, num_samples, rv_type='normal')
    diag_reciprocal = jax.tree.map(lambda x: jnp.where(x > 1e-10, jnp.reciprocal(x), jnp.zeros_like(x)), diag_est)

    def preconditioner(v):
        return jax.tree.map(lambda x, y: x * y, v, diag_reciprocal)

    return preconditioner
