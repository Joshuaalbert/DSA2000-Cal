import time

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.probabilistic_models.gain_prior_models import GainPriorModel


@pytest.mark.parametrize(
    'di_type', ['unconstrained', 'phase', 'amplitude', 'clock', 'dtec', 'amplitude+phase', 'amplitude+clock', 'clock+dtec']
)
@pytest.mark.parametrize(
    'dd_type', ['unconstrained', 'phase', 'amplitude', 'clock', 'dtec', 'amplitude+phase', 'amplitude+clock', 'clock+dtec']
)
@pytest.mark.parametrize(
    'full_stokes,dd_dof,double_differential,di_dof', [
        (True, 1, True, 1),
        (True, 1, True, 2),
        (True, 1, True, 4),
        (True, 2, True, 1),
        (True, 2, True, 2),
        (True, 2, True, 4),
        (True, 4, True, 1),
        (True, 4, True, 2),
        (True, 4, True, 4),
        (False, 1, False, 1)
    ]
)
def test_gain_prior_model(full_stokes, dd_type, dd_dof, double_differential, di_dof, di_type):
    D = 4
    num_ant = 3
    T = 5
    C = 6

    freqs = jnp.array([1.4e9] * C)
    times = jnp.array([0.] * T)

    gain_probabilistic_model = GainPriorModel(
        num_source=D,
        num_ant=num_ant,
        freqs=freqs,
        times=times,
        gain_stddev=1.,
        full_stokes=full_stokes,
        dd_type=dd_type,
        dd_dof=dd_dof,
        double_differential=double_differential,
        di_dof=di_dof,
        di_type=di_type
    )

    init_params = gain_probabilistic_model.get_init_params(jax.random.PRNGKey(0))
    for leaf in jax.tree.leaves(init_params):
        assert np.all(np.isfinite(leaf))

    gains = gain_probabilistic_model.compute_gains(init_params)

    if full_stokes:
        assert gains.shape == (D, T, num_ant, C, 2, 2)
    else:
        assert gains.shape == (D, T, num_ant, C)

    assert np.all(np.isfinite(gains))

    # Performance test compute_gains
    compute_gains = jax.jit(gain_probabilistic_model.compute_gains).lower(init_params).compile()

    t0 = time.time()
    for _ in range(10):
        jax.block_until_ready(compute_gains(init_params))
    t1 = time.time()
    print(f"Took {(t1-t0)/10} seconds.")

