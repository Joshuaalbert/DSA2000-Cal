import numpy as np
import pytest
from jax import numpy as jnp

from dsa2000_cal.calibration.solvers.approx_cg_newton import ApproxCGNewton
from dsa2000_common.common.array_types import FloatArray


@pytest.mark.parametrize("n", [2, 10, 100, 1000])
def test_approx_cg_newton(n):
    def rosenbrock_nd(x: FloatArray) -> FloatArray:
        a = 1.
        b = 100.
        return jnp.sum((a - x[:-1]) ** 2 + b * (x[1:] - x[:-1] ** 2) ** 2)

    x0 = 10 * jnp.ones(n)

    solver = ApproxCGNewton(rosenbrock_nd, num_approx_steps=2, num_iterations=40, verbose=True)
    state = solver.create_initial_state(x0)
    state, diagnostics = solver.solve(state)
    print(state)
    import pylab as plt
    plt.plot(np.log(diagnostics.error), label="error")
    plt.plot(np.log(diagnostics.delta_norm), label="delta_norm")
    plt.plot(np.log(diagnostics.error/diagnostics.delta_norm), label="error/delta_norm")
    plt.plot(np.log(diagnostics.damping), label="damping")
    plt.plot(np.log(diagnostics.mu), label="mu")
    plt.legend()
    plt.show()
