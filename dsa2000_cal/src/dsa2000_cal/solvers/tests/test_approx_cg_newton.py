import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.solvers.approx_cg_newton import newton_cg_solver
from dsa2000_common.common.array_types import FloatArray


@pytest.mark.parametrize("n", [2, 10, 100, 1000])
def test_approx_cg_newton(n):
    def rosenbrock_nd(x: FloatArray) -> FloatArray:
        a = 1.
        b = 100.
        return jnp.sum((a - x[:-1]) ** 2 + b * (x[1:] - x[:-1] ** 2) ** 2)

    x0 = 10 * jnp.ones(n)

    solution, diagnostics = newton_cg_solver(rosenbrock_nd, x0, verbose=True)
    np.testing.assert_allclose(solution, jnp.ones(n), atol=1e-4)
    import pylab as plt
    plt.plot(np.log(diagnostics.g_norm), label="error")
    plt.plot(np.log(diagnostics.ddelta_x_norm), label="delta_norm")
    plt.plot(np.log(diagnostics.g_norm / diagnostics.ddelta_x_norm), label="error/delta_norm")
    plt.plot(np.log(diagnostics.damping), label="damping")
    plt.plot(np.log(diagnostics.mu), label="mu")
    plt.legend()
    plt.show()
