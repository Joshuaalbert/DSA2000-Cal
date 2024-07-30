from typing import Tuple

import jax
import numpy as np
from jax import numpy as jnp, lax


def vec(a: jnp.ndarray, transpose: bool = False) -> jnp.ndarray:
    """
    Vectorize a matrix.

    Args:
        a: [n, m] array

    Returns:
        [n*m] array
    """
    if len(a.shape) != 2:
        raise ValueError(f"a should be a matrix, got shape {a.shape}")
    n, m = a.shape
    # a.T.ravel()
    if transpose:
        return lax.reshape(a, (n * m,))
    return lax.reshape(a, (n * m,), (1, 0))


def unvec(a: jnp.ndarray, shape: Tuple[int, ...] | None = None, transpose: bool = False) -> jnp.ndarray:
    """
    Unvectorize a matrix.

    Args:
        a: [n*m] array
        shape: shape of the unvectorized array

    Returns:
        [n, m] array
    """
    if shape is None:
        # assume square
        n = int(np.sqrt(a.shape[-1]))
        if n * n != a.shape[-1]:
            raise ValueError(f"a is not square. Can't infer unvec shape.")
        shape = (n, n)
    if len(shape) != 2:
        raise ValueError(f"shape should be length 2, got {len(shape)}")
    if transpose:
        # jnp.reshape(a, shape).T
        return lax.reshape(a, shape)
    # jnp.reshape(a, shape).T
    return lax.transpose(lax.reshape(a, shape), (1, 0))


def kron(a, b):
    """
    Compute the Kronecker product of two arrays.

    Args:
        a: [n, m]
        b: [p, q]

    Returns:
        [n*p, m*q]
    """
    if len(np.shape(a)) < len(np.shape(b)):
        a = lax.expand_dims(a, range(np.ndim(b) - np.ndim(a)))
    elif np.ndim(b) < np.ndim(a):
        b = lax.expand_dims(b, range(np.ndim(a) - np.ndim(b)))
    a_reshaped = lax.expand_dims(a, range(1, 2 * np.ndim(a), 2))
    b_reshaped = lax.expand_dims(b, range(0, 2 * np.ndim(b), 2))
    out_shape = tuple(np.multiply(np.shape(a), np.shape(b)))
    return lax.reshape(lax.mul(a_reshaped, b_reshaped), out_shape)


def kron_product(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    """
    Compute the matrix product of three matrices using Kronecker product.

    a @ b @ c

    Args:
        a: [n, m]
        b: [m, p]
        c: [p, q]

    Returns:
        [n, q]
    """

    def kron_product_1(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
        return a @ b @ c

    def kron_product_2(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
        return unvec(kron(c.T, a) @ vec(b), (a.shape[0], c.shape[1]))

    def kron_product_3(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
        return unvec(jnp.sum(kron(c.T, a) * vec(b), axis=-1), (a.shape[0], c.shape[1]))

    def kron_product_2x2(M0: jax.Array, M1: jax.Array, M2: jax.Array) -> jax.Array:
        # Matrix([[a0*(a1*a2 + b1*c2) + b0*(a2*c1 + c2*d1), a0*(a1*b2 + b1*d2) + b0*(b2*c1 + d1*d2)], [c0*(a1*a2 + b1*c2) + d0*(a2*c1 + c2*d1), c0*(a1*b2 + b1*d2) + d0*(b2*c1 + d1*d2)]])
        # 36
        # ([(x0, a1*a2 + b1*c2), (x1, a2*c1 + c2*d1), (x2, a1*b2 + b1*d2), (x3, b2*c1 + d1*d2)], [Matrix([
        # [a0*x0 + b0*x1, a0*x2 + b0*x3],
        # [c0*x0 + d0*x1, c0*x2 + d0*x3]])])
        a0, b0, c0, d0 = M0[0, 0], M0[0, 1], M0[1, 0], M0[1, 1]
        a1, b1, c1, d1 = M1[0, 0], M1[0, 1], M1[1, 0], M1[1, 1]
        a2, b2, c2, d2 = M2[0, 0], M2[0, 1], M2[1, 0], M2[1, 1]
        x0 = a1 * a2 + b1 * c2
        x1 = a2 * c1 + c2 * d1
        x2 = a1 * b2 + b1 * d2
        x3 = b2 * c1 + d1 * d2

        # flat = jnp.stack([a0 * x0 + b0 * x1, c0 * x0 + d0 * x1, a0 * x2 + b0 * x3, c0 * x2 + d0 * x3], axis=-1)
        # return unvec(flat, (2, 2))
        flat = jnp.stack([a0 * x0 + b0 * x1, a0 * x2 + b0 * x3, c0 * x0 + d0 * x1, c0 * x2 + d0 * x3], axis=-1)
        return lax.reshape(flat, (2, 2))

    n, m = np.shape(a)
    m, p = np.shape(b)
    p, q = np.shape(c)
    s = n
    if (n == m and m == p and p == q) and (s == 2):
        # s=2:
        # 	kron_product_1: max error=0.0 run_time=0.04960222244262695 BA=96.0, F=32.0 BA(B)=-2.0, F(B)=-2.0
        # 	kron_product_2: max error=3.814697265625e-06 run_time=0.00281221866607666 BA=96.0, F=48.0 BA(B)=25600000.0, F(B)=4800000.0
        # 	kron_product_3: max error=3.814697265625e-06 run_time=0.0005743980407714843 BA=64.0, F=44.0 BA(B)=6400000.0, F(B)=4400000.0
        # 	kron_product_2x2: max error=3.814697265625e-06 run_time=0.0004351615905761719 BA=64.0, F=24.0 BA(B)=6400000.0, F(B)=2400000.0
        return kron_product_2x2(a, b, c)
    if s == 2:
        # s=2:
        # 	kron_product_1: max error=0.0 run_time=0.04960222244262695 BA=96.0, F=32.0 BA(B)=-2.0, F(B)=-2.0
        # 	kron_product_2: max error=3.814697265625e-06 run_time=0.00281221866607666 BA=96.0, F=48.0 BA(B)=25600000.0, F(B)=4800000.0
        # 	kron_product_3: max error=3.814697265625e-06 run_time=0.0005743980407714843 BA=64.0, F=44.0 BA(B)=6400000.0, F(B)=4400000.0
        return kron_product_3(a, b, c)
    elif s == 3:
        # s=3:
        # 	kron_product_1: max error=0.0 run_time=0.05097482204437256 BA=216.0, F=108.0 BA(B)=-2.0, F(B)=-2.0
        # 	kron_product_2: max error=5.7220458984375e-06 run_time=0.015997862815856932 BA=216.0, F=243.0 BA(B)=93600000.0, F(B)=24300000.0
        # 	kron_product_3: max error=7.62939453125e-06 run_time=0.002956199645996094 BA=144.0, F=234.0 BA(B)=14400000.0, F(B)=23400000.0
        return kron_product_3(a, b, c)
    elif s == 4:
        # s=4:
        # 	kron_product_1: max error=0.0 run_time=0.052040839195251466 BA=384.0, F=256.0 BA(B)=-2.0, F(B)=-2.0
        # 	kron_product_2: max error=7.62939453125e-06 run_time=0.059854960441589354 BA=384.0, F=768.0 BA(B)=140800000.0, F(B)=25600000.0
        # 	kron_product_3: max error=7.62939453125e-06 run_time=0.006914520263671875 BA=256.0, F=752.0 BA(B)=25600000.0, F(B)=75200000.0
        return kron_product_3(a, b, c)
    elif s == 5:
        # s=5:
        # 	kron_product_1: max error=0.0 run_time=0.054116344451904295 BA=600.0, F=500.0 BA(B)=-2.0, F(B)=-2.0
        # 	kron_product_2: max error=1.1444091796875e-05 run_time=0.1432589292526245 BA=600.0, F=1875.0 BA(B)=310000000.0, F(B)=62500000.0
        # 	kron_product_3: max error=7.62939453125e-06 run_time=0.05458433628082275 BA=400.0, F=1850.0 BA(B)=40000000.0, F(B)=185000000.0
        return kron_product_3(a, b, c)  # Almost 1
    elif s == 6:
        # s=6:
        # 	kron_product_1: max error=0.0 run_time=0.05298464298248291 BA=864.0, F=864.0 BA(B)=-2.0, F(B)=-2.0
        # 	kron_product_2: max error=7.62939453125e-06 run_time=0.25304379463195803 BA=864.0, F=3888.0 BA(B)=604800000.0, F(B)=129600000.0
        # 	kron_product_3: max error=7.62939453125e-06 run_time=0.2962526321411133 BA=11524.0, F=4860.0 BA(B)=1152000000.0, F(B)=486000000.0
        return kron_product_1(a, b, c)
    else:
        return kron_product_1(a, b, c)


def kron_inv(a: jax.Array, K: jax.Array, c: jax.Array) -> jax.Array:
    """
    Compute the matrix product of three matrices using Kronecker product.

    a^-1 @ K @ c^-1 -> b

    where K = a @ b @ c

    Args:
        a: [n, m]
        b: [m, p]
        c: [p, q]

    Returns:
        [n, q]
    """

    def kron_inv_1(a: jax.Array, K: jax.Array, c: jax.Array) -> jax.Array:
        return kron_product(jnp.linalg.pinv(a), K, jnp.linalg.pinv(c))

    def kron_inv_2(a: jax.Array, K: jax.Array, c: jax.Array) -> jax.Array:
        return kron_product(jnp.linalg.inv(a), K, jnp.linalg.inv(c))

    def kron_inv_3(a: jax.Array, K: jax.Array, c: jax.Array) -> jax.Array:
        return unvec(jnp.sum(jnp.linalg.inv(kron(c.T, a)) * vec(K), axis=-1), (a.shape[0], c.shape[1]))

    def kron_inv_4(a: jax.Array, K: jax.Array, c: jax.Array) -> jax.Array:
        return unvec(jnp.linalg.solve(kron(c.T, a), vec(K)), (a.shape[0], c.shape[1]))

    def kron_inv_2x2(a: jax.Array, K: jax.Array, c: jax.Array) -> jax.Array:
        # Matrix([[(-c10*(K01*a11 - K11*a01) + c11*(K00*a11 - K10*a01))/((a00*a11 - a01*a10)*(c00*c11 - c01*c10)), (c00*(K01*a11 - K11*a01) - c01*(K00*a11 - K10*a01))/((a00*a11 - a01*a10)*(c00*c11 - c01*c10))], [(c10*(K01*a10 - K11*a00) - c11*(K00*a10 - K10*a00))/((a00*a11 - a01*a10)*(c00*c11 - c01*c10)), (-c00*(K01*a10 - K11*a00) + c01*(K00*a10 - K10*a00))/((a00*a11 - a01*a10)*(c00*c11 - c01*c10))]])
        # CSE:
        # ([(x0, K01*a11 - K11*a01), (x1, K00*a11 - K10*a01), (x2, 1/((a00*a11 - a01*a10)*(c00*c11 - c01*c10))), (x3, K01*a10 - K11*a00), (x4, K00*a10 - K10*a00)], [Matrix([
        # [x2*(-c10*x0 + c11*x1),  x2*(c00*x0 - c01*x1)],
        # [ x2*(c10*x3 - c11*x4), x2*(-c00*x3 + c01*x4)]])])
        a00, a01, a10, a11 = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
        K00, K01, K10, K11 = K[0, 0], K[0, 1], K[1, 0], K[1, 1]
        c00, c01, c10, c11 = c[0, 0], c[0, 1], c[1, 0], c[1, 1]
        x0 = K01 * a11 - K11 * a01
        x1 = K00 * a11 - K10 * a01
        x2 = jnp.reciprocal((a00 * a11 - a01 * a10) * (c00 * c11 - c01 * c10))
        x3 = K01 * a10 - K11 * a00
        x4 = K00 * a10 - K10 * a00

        # flat = jnp.stack([x2 * (-c10 * x0 + c11 * x1), x2 * (c10 * x3 - c11 * x4), x2 * (c00 * x0 - c01 * x1),  x2 * (-c00 * x3 + c01 * x4)], axis=-1)
        # return unvec(flat, (2, 2))
        flat = jnp.stack([x2 * (-c10 * x0 + c11 * x1), x2 * (c00 * x0 - c01 * x1), x2 * (c10 * x3 - c11 * x4),
                          x2 * (-c00 * x3 + c01 * x4)], axis=-1)
        return lax.reshape(flat, (2, 2))

    n, m = np.shape(a)
    m, p = np.shape(K)
    p, q = np.shape(c)
    s = n
    if not (n == m and m == p and p == q):
        raise ValueError(f"Square matrices expected, got shapes {np.shape(a)}, {np.shape(K)}, {np.shape(c)}")

    if s == 2:
        # s=2:
        # 	kron_inv_1: max error=0.0017872750759124756 run_time=0.002995920181274414 BA=382.0, F=96.0 BA(B)=383998.0, F(B)=97998.0
        # 	kron_inv_2: max error=0.0037404000759124756 run_time=0.00038604736328125 BA=724.0, F=92.0 BA(B)=872590.0, F(B)=88058.0
        # 	kron_inv_3: max error=0.0027638375759124756 run_time=0.0004916667938232421 BA=958.0, F=122.0 BA(B)=2880335.0, F(B)=377055.0
        # 	kron_inv_4: max error=0.003809601068496704 run_time=0.00042150020599365237 BA=602.0, F=62.0 BA(B)=1344327.0, F(B)=224023.0
        return kron_inv_2x2(a, K, c)
    elif s == 3:
        # s=3:
        # 	kron_inv_1: max error=0.002649068832397461 run_time=0.005621647834777832 BA=830.0, F=418.0 BA(B)=831998.0, F(B)=419998.0
        # 	kron_inv_2: max error=0.0014542192220687866 run_time=0.0007739543914794921 BA=1148.0, F=340.0 BA(B)=2176646.0, F(B)=406080.0
        # 	kron_inv_3: max error=0.0024307817220687866 run_time=0.0021479129791259766 BA=3578.0, F=527.0 BA(B)=29320356.0, F(B)=3902185.0
        # 	kron_inv_4: max error=0.0012392476201057434 run_time=0.001464080810546875 BA=1702.0, F=212.0 BA(B)=10604327.0, F(B)=2259023.0
        return kron_inv_2(a, K, c)
    elif s == 4:
        # s=4:
        # 	kron_inv_1: max error=0.005160659551620483 run_time=0.010452651977539062 BA=1454.0, F=1132.0 BA(B)=1455998.0, F(B)=1133998.0
        # 	kron_inv_2: max error=0.0024751126766204834 run_time=0.0011742830276489258 BA=1716.0, F=908.0 BA(B)=4704654.0, F(B)=1112108.0
        # 	kron_inv_3: max error=0.0076525211334228516 run_time=0.008309316635131837 BA=10270.0, F=1598.0 BA(B)=157392320.0, F(B)=21269536.0
        # 	kron_inv_4: max error=0.00547172874212265 run_time=0.0036844491958618166 BA=4250.0, F=590.0 BA(B)=53808328.0, F(B)=12416023.0
        return kron_inv_2(a, K, c)
    elif s == 5:
        # s=5:
        # 	kron_inv_1: max error=0.007080078125 run_time=0.017490625381469727 BA=2254.0, F=2534.0 BA(B)=2255998.0, F(B)=2535998.0
        # 	kron_inv_2: max error=0.00439453125 run_time=0.0018147468566894532 BA=2428.0, F=2068.0 BA(B)=8624662.0, F(B)=2506144.0
        # 	kron_inv_3: max error=0.0168379545211792 run_time=0.02645268440246582 BA=24058.0, F=3839.0 BA(B)=587016320.0, F(B)=80031272.0
        # 	kron_inv_4: max error=0.005302906036376953 run_time=0.008314085006713868 BA=9254.0, F=1364.0 BA(B)=197484320.0, F(B)=47075024.0
        return kron_inv_2(a, K, c)
    else:
        return kron_inv_2(a, K, c)
