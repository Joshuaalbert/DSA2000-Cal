import jax.numpy as jnp

from dsa2000_cal.common.jvp_linear_op import JVPLinearOp


def test_jvp_linear_op():
    n = 4
    k = 10
    m = 2

    def fn(x):
        return jnp.asarray([jnp.sum(jnp.sin(x) ** i) for i in range(m)])

    x = jnp.arange(n).astype(jnp.float32)

    jvp_op = JVPLinearOp(fn, primals=x)

    x_space = jnp.ones((n, k))
    f_space = jnp.ones((m, k))

    assert jvp_op.matvec(x_space[:, 0]).shape == (m,)
    assert jnp.allclose(jvp_op.matvec(x_space[:, 0]), jvp_op.to_dense() @ x_space[:, 0])
    assert jnp.allclose(jvp_op @ x_space[:, 0], jvp_op.to_dense() @ x_space[:, 0])

    assert jvp_op.matvec(f_space[:, 0], adjoint=True).shape == (n,)
    assert jnp.allclose(jvp_op.matvec(f_space[:, 0], adjoint=True), jvp_op.to_dense().T @ f_space[:, 0])
    assert jnp.allclose(jvp_op.T @ f_space[:, 0], jvp_op.to_dense().T @ f_space[:, 0])

    assert jvp_op.matmul(x_space).shape == (m, k)
    assert jnp.allclose(jvp_op.matmul(x_space), jvp_op.to_dense() @ x_space)
    assert jnp.allclose(jvp_op @ x_space, jvp_op.to_dense() @ x_space)

    assert jvp_op.matmul(x_space.T, left_multiply=False).shape == (k, m)
    assert jnp.allclose(jvp_op.matmul(x_space.T, left_multiply=False), x_space.T @ jvp_op.to_dense().T)
    assert jnp.allclose(x_space.T @ jvp_op.T, x_space.T @ jvp_op.to_dense().T)

    assert jvp_op.matmul(f_space, adjoint=True).shape == (n, k)
    assert jnp.allclose(jvp_op.matmul(f_space, adjoint=True), jvp_op.to_dense().T @ f_space)
    assert jnp.allclose(jvp_op.T @ f_space, jvp_op.to_dense().T @ f_space)

    assert jvp_op.matmul(f_space.T, adjoint=True, left_multiply=False).shape == (k, n)
    assert jnp.allclose(jvp_op.matmul(f_space.T, adjoint=True, left_multiply=False), f_space.T @ jvp_op.to_dense())
    assert jnp.allclose(f_space.T @ jvp_op, f_space.T @ jvp_op.to_dense())

    # test setting primals
    assert jnp.allclose(jvp_op(x).matvec(x_space[:, 0]), jvp_op.matvec(x_space[:, 0]))

    # Test neg
    assert jnp.allclose((-jvp_op).to_dense(), -jvp_op.to_dense())

    # Test when f: R^n -> scalar
    def fn(x):
        return jnp.sum(jnp.sin(x))

    jvp_op = JVPLinearOp(fn, primals=x)
    assert jvp_op.matvec(x_space[:, 0]).shape == ()
    assert jnp.allclose(jvp_op.matvec(x_space[:, 0]), jvp_op.to_dense() @ x_space[:, 0])
    assert jnp.allclose(jvp_op @ x_space[:, 0], jvp_op.to_dense() @ x_space[:, 0])

