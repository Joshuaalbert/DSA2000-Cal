import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_cal.common.vec_utils import vec, unvec, kron_product, kron_product_2x2


def test_vec():
    a = jnp.asarray([[1, 2],
                     [3, 4]])
    assert jnp.alltrue(vec(a) == jnp.asarray([1, 3, 2, 4]))

    assert jnp.alltrue(unvec(vec(a), (2, 2)) == a)
    assert jnp.alltrue(unvec(vec(a)) == a)


def test_kron_product():
    g1 = jax.random.normal(jax.random.PRNGKey(0), (2, 2)) + 1j * jax.random.normal(jax.random.PRNGKey(1), (2, 2))
    g2 = jax.random.normal(jax.random.PRNGKey(2), (2, 2)) + 1j * jax.random.normal(jax.random.PRNGKey(3), (2, 2))
    coherencies = jax.random.normal(jax.random.PRNGKey(4), (2, 2)) + 1j * jax.random.normal(jax.random.PRNGKey(5),
                                                                                            (2, 2))

    output = kron_product(g1, coherencies, g2.conj().T)
    expected = g1 @ coherencies @ g2.conj().T
    np.testing.assert_allclose(output, expected, atol=1e-6)

    output = kron_product(g1, coherencies, g1.conj().T)
    expected = g1 @ coherencies @ g1.conj().T
    np.testing.assert_allclose(output, expected, atol=1e-6)


def test_kron_product_cost():
    a = jnp.arange(4).reshape((2, 2)).astype(complex)
    b = jnp.arange(4).reshape((2, 2)).astype(complex)
    c = jnp.arange(4).reshape((2, 2)).astype(complex)

    def f(a, b, c):
        return a @ b @ c

    p1 = f(a, b, c)

    p2 = kron_product(a, b, c)

    assert np.alltrue(p2 == p1)

    a1 = jax.jit(f).lower(a, b, c).compile().cost_analysis()[0]
    a2 = jax.jit(kron_product).lower(a, b, c).compile().cost_analysis()[0]
    print()
    print("a @ b @ c")
    print("Naive a.b.c | unvec(kron(c.T, a).vec(b))")
    for key in ['bytes accessed', 'flops', 'utilization operand 0 {}', 'utilization operand 1 {}',
                'utilization operand 2 {}', 'bytes accessed output {}']:
        print(key, ":", a1.get(key, None), a2.get(key, None))

    print()
    print("a @ b @ c.T.conj")
    print("Naive a.b.c | unvec(kron(c.T, a).vec(b))")
    a1 = jax.jit(lambda a, b, c: f(a, b, c.conj().T)).lower(a, b, c).compile().cost_analysis()[0]
    a2 = jax.jit(lambda a, b, c: kron_product(a, b, c.conj().T)).lower(a, b, c).compile().cost_analysis()[0]
    for key in ['bytes accessed', 'flops', 'utilization operand 0 {}', 'utilization operand 1 {}',
                'utilization operand 2 {}', 'bytes accessed output {}']:
        print(key, ":", a1.get(key, None), a2.get(key, None))

    print()
    print("a @ b @ c.conj.T")
    print("Naive a.b.c | unvec(kron(c.T, a).vec(b))")
    a1 = jax.jit(lambda a, b, c: f(a, b, c.conj().T)).lower(a, b, c).compile().cost_analysis()[0]
    a2 = jax.jit(lambda a, b, c: kron_product(a, b, c.conj().T)).lower(a, b, c).compile().cost_analysis()[0]
    for key in ['bytes accessed', 'flops', 'utilization operand 0 {}', 'utilization operand 1 {}',
                'utilization operand 2 {}', 'bytes accessed output {}']:
        print(key, ":", a1.get(key, None), a2.get(key, None))


def test_kron_product_2x2():
    M0 = jnp.array([[1, 2], [3, 4]])
    M1 = jnp.array([[5, 6], [7, 8]])
    M2 = jnp.array([[9, 10], [11, 12]])
    np.testing.assert_allclose(kron_product_2x2(M0, M1, M2), M0 @ M1 @ M2)


def test_derive_best_2x2():
    import sympy as sp
    a0, b0, c0, d0 = sp.symbols('a0 b0 c0 d0')
    a1, b1, c1, d1 = sp.symbols('a1 b1 c1 d1')
    a2, b2, c2, d2 = sp.symbols('a2 b2 c2 d2')

    M0 = sp.Matrix([[a0, b0], [c0, d0]])
    M1 = sp.Matrix([[a1, b1], [c1, d1]])
    M2 = sp.Matrix([[a2, b2], [c2, d2]])

    M = M0 @ M1 @ M2
    # M = M.simplify()

    print(M)

    # common substring optimal evaluation
    # Compute (M0 @ M1) @ M2
    M0M1 = M0 @ M1
    M = M0M1 @ M2
    M.simplify()
    print("Order (M0 @ M1) @ M2:")
    print(M)
    print(sp.count_ops(M))

    # Compute M0 @ (M1 @ M2)
    M1M2 = M1 @ M2
    M = M0 @ M1M2
    M.simplify()
    print("Order M0 @ (M1 @ M2):")
    print(M)

    # Compare the number of operations for each case
    print("Operations for (M0 @ M1) @ M2:")
    print(sp.count_ops(M0M1 @ M2))

    print("Operations for M0 @ (M1 @ M2):")
    print(sp.count_ops(M0 @ M1M2))

    M = sp.simplify(M, ratio=1.0, inverse=True)
    print(M)
    print(sp.count_ops(M))

    print(sp.cse(M))