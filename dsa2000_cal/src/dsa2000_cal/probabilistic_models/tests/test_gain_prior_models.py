import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jaxns.framework import context as ctx
from jaxns.framework.ops import simulate_prior_model

from dsa2000_cal.probabilistic_models.gain_prior_models import GainPriorModel


@pytest.mark.parametrize(
    'full_stokes,dd_type,dd_dof,double_differential,di_dof,di_type', [
        (True, 'unconstrained', 1, True, 1, 'unconstrained'),
        (True, 'rice', 1, True, 1, 'unconstrained'),
        (True, 'unconstrained', 2, True, 2, 'unconstrained'),
        (True, 'rice', 2, True, 2, 'unconstrained'),
        (True, 'unconstrained', 4, True, 4, 'unconstrained'),
        (False, 'unconstrained', 1, False, 1, 'unconstrained'),
        (False, 'rice', 1, False, 1, 'unconstrained'),
    ]
)
def test_gain_prior_model(full_stokes, dd_type, dd_dof, double_differential, di_dof, di_type):
    gain_probabilistic_model = GainPriorModel(
        gain_stddev=1.,
        full_stokes=full_stokes,
        dd_type=dd_type,
        dd_dof=dd_dof,
        double_differential=double_differential,
        di_dof=di_dof,
        di_type=di_type
    )

    D = 4
    num_ant = 3
    freqs = jnp.array([1.4e9] * 3)
    times = jnp.array([0.] * 4)

    def get_gains():
        # TODO: pass in probabilitic model
        prior_model = gain_probabilistic_model.build_prior_model(
            num_source=D,
            num_ant=num_ant,
            freqs=freqs,
            times=times
        )
        (gains,), _ = simulate_prior_model(jax.random.PRNGKey(0), prior_model)  # [D, Tm, A, Cm[,2,2]]
        return gains  # [D, Tm, A, Cm[,2,2]]

    get_gains_transformed = ctx.transform(get_gains)

    params = get_gains_transformed.init(jax.random.PRNGKey(0)).params

    gains = get_gains_transformed.apply(params, jax.random.PRNGKey(0)).fn_val
    if full_stokes:
        assert gains.shape == (D, 4, num_ant, 3, 2, 2)
    else:
        assert gains.shape == (D, 4, num_ant, 3)

    assert np.all(np.isfinite(gains))
