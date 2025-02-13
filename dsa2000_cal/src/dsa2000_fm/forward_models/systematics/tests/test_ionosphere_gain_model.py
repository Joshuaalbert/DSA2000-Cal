import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_cal.common.linalg_utils import msqrt
from dsa2000_fm.forward_models.systematics.ionosphere_gain_model import interpolate_antennas


def test_msqrt():
    M = jax.random.normal(jax.random.PRNGKey(42), (100, 100))
    A = M @ M.T
    max_eig, min_eig, L = msqrt(A)
    np.testing.assert_allclose(A, L @ L.T, atol=2e-4)


def test_interpolate_antennas():
    N = 4
    M = 5
    num_time = 6
    num_dir = 7
    dtec_interp = interpolate_antennas(
        antennas_enu=jnp.ones((N, 3)),
        model_antennas_enu=jnp.ones((M, 3)),
        dtec=jnp.ones((num_time, num_dir, M))
    )
    assert dtec_interp.shape == (num_time, num_dir, N)
