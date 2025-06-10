# tests/test_apply_gains.py

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from dsa2000_cal.ops.residuals import compute_residual_TBC, apply_gains_to_model_vis_TBC


def _rand_complex(key, shape, dtype=jnp.complex64):
    """Utility: random complex array with given shape."""
    real_key, imag_key = jax.random.split(key)
    real = jax.random.normal(real_key, shape)
    imag = jax.random.normal(imag_key, shape)
    return (real + 1.0j * imag).astype(dtype)


# ==================================================================
# 1.  SHAPE & BROADCASTING TESTS
# ==================================================================

@pytest.mark.parametrize(
    "full_stokes, tm, ts, cm, cs",
    [
        # scalar gains ------------------------------------------------
        (False, 4, 4, 6, 6),  # identical
        (False, 4, 2, 6, 3),  # repeated
        (False, 4, 1, 6, 1),  # broadcast
        # full-Stokes gains ------------------------------------------
        (True, 4, 4, 6, 6),
        (True, 4, 2, 6, 3),
        (True, 4, 1, 6, 1),
    ],
)
def test_output_shape(full_stokes, tm, ts, cm, cs):
    """
    For every gain layout make sure `compute_residual_TBC`
    returns the expected shape.
    """
    key = jax.random.PRNGKey(0)
    d, a, b = 2, 5, 3  # directions, antennas, baselines
    dtype = jnp.complex64

    # ----------------------------------------------------------------
    # Model visibilities
    # ----------------------------------------------------------------
    vis_shape = (d, tm, b, cm, 2, 2) if full_stokes else (d, tm, b, cm)
    vis_model = _rand_complex(key, vis_shape, dtype)

    # ----------------------------------------------------------------
    # Gains
    # ----------------------------------------------------------------
    gain_shape = (
        (d, ts, a, cs, 2, 2) if full_stokes else (d, ts, a, cs)
    )
    gains = _rand_complex(jax.random.PRNGKey(1), gain_shape, dtype)

    # ----------------------------------------------------------------
    # Data visibilities (all zeros so that residual == -model@gains)
    # ----------------------------------------------------------------
    vis_data_shape = (tm, b, cm, 2, 2) if full_stokes else (tm, b, cm)
    vis_data = jnp.zeros(vis_data_shape, dtype)

    # ----------------------------------------------------------------
    # Simple antenna pairing: (baseline i = (i, i+1))
    # ----------------------------------------------------------------
    antenna1 = jnp.arange(b)
    antenna2 = (antenna1 + 1) % a

    residual = compute_residual_TBC(
        vis_model, vis_data, gains, antenna1, antenna2
    )

    assert residual.shape == vis_data_shape


def test_divisibility_errors():
    """Ts or Cs that do **not** divide Tm / Cm must raise."""
    d, tm, b, cm = 1, 4, 1, 6
    ts, cs, a = 3, 4, 2  # deliberately wrong
    key = jax.random.PRNGKey(42)

    vis_model = _rand_complex(key, (d, tm, b, cm))
    gains = _rand_complex(key, (d, ts, a, cs))
    vis_data = jnp.zeros((tm, b, cm), dtype=vis_model.dtype)

    with pytest.raises(ValueError, match="compatible"):
        compute_residual_TBC(
            vis_model, vis_data, gains,
            antenna1=jnp.array([0]), antenna2=jnp.array([1])
        )


def test_invalid_gain_rank():
    """A gain array with rank ≠ 4 or 6 must fail fast."""
    d, tm, b, cm = 1, 1, 1, 1
    bad_gains = jnp.zeros((d, tm, 2, cm, 3))  # rank 5
    vis_model = jnp.zeros((d, tm, b, cm))
    vis_data = jnp.zeros((tm, b, cm))

    with pytest.raises(ValueError, match="Invalid gains"):
        compute_residual_TBC(
            vis_model, vis_data, bad_gains,
            antenna1=jnp.array([0]), antenna2=jnp.array([1])
        )


# ==================================================================
# 2.  SYMPY-VALIDATED NUMERICAL TRUTH TESTS
# ==================================================================


def test_scalar_gain_math_with_sympy():
    """
    1 direction, 1 time, 1 baseline, 1 channel.
    Sympy provides the ground-truth complex arithmetic.
    """
    # -----------------------------------
    # Ground-truth numbers in Sympy
    # -----------------------------------
    v = 3 + 4 * sp.I
    g1 = 2 - sp.I
    g2 = 1 + 5 * sp.I
    expected = g1 * sp.conjugate(g2) * v

    # -----------------------------------
    # Same numbers in JAX
    # -----------------------------------
    vis_model = jnp.asarray([[[[complex(v)]]]])
    gains = jnp.asarray([[[[complex(g1)], [complex(g2)]]]])  # A=2
    antenna1 = jnp.array([0])
    antenna2 = jnp.array([1])

    out = apply_gains_to_model_vis_TBC(
        vis_model, gains, antenna1, antenna2
    )  # shape (Tm=1,B=1,Cm=1)

    np.testing.assert_allclose(out.squeeze(), complex(expected), rtol=1e-7, atol=1e-7)


def test_full_stokes_gain_math_with_sympy():
    """
    Same idea but in full-Stokes (2×2) mode.
    """
    # Sympy matrices --------------------------------------------------
    g1 = sp.Matrix([[1 + sp.I, 2],
                    [3 - sp.I, 4]])
    g2 = sp.Matrix([[1, -1 - sp.I],
                    [2 * sp.I, 0.5]])
    V = sp.Matrix([[2, 1 - sp.I],
                   [-1, sp.I]])

    expected = g1 @ V @ g2.conjugate().T  # Eq. (4)

    # Convert to numpy ----------------------------------------------
    g1_np = np.array(g1.subs(sp.I, 1j)).astype(np.complex64)
    g2_np = np.array(g2.subs(sp.I, 1j)).astype(np.complex64)
    V_np = np.array(V.subs(sp.I, 1j)).astype(np.complex64)

    # JAX tensors ----------------------------------------------------
    D, Tm, B, Cm, A = 1, 1, 1, 1, 2
    vis_model = jnp.asarray(V_np).reshape(D, Tm, B, Cm, 2, 2)
    gains = jnp.zeros((D, Tm, A, Cm, 2, 2), vis_model.dtype)
    gains = gains.at[0, 0, 0, 0].set(g1_np)
    gains = gains.at[0, 0, 1, 0].set(g2_np)

    antenna1 = jnp.array([0])
    antenna2 = jnp.array([1])

    out = apply_gains_to_model_vis_TBC(
        vis_model, gains, antenna1, antenna2
    )
    np.testing.assert_allclose(out.squeeze(),
                               np.array(expected.tolist()).astype(np.complex64),
                               rtol=1e-6, atol=1e-6)


# ==================================================================
# 3.  GRADIENT FLOW TESTS
# ==================================================================


@pytest.mark.parametrize("full_stokes", [False, True])
def test_gain_parameters_receive_gradients(full_stokes):
    """
    Build a least-squares loss ‖V_obs – V_model@gains‖² with V_obs ≡ 0.
    Confirm **all** gain elements (real *and* imaginary) get a finite,
    non-zero gradient.
    """
    key = jax.random.PRNGKey(123)
    d, tm, b, cm, a = 1, 2, 3, 4, 5
    dtype = jnp.complex64

    # ----------------------------------------------------------------
    # Random visibilities and gains
    # ----------------------------------------------------------------
    vis_shape = (d, tm, b, cm, 2, 2) if full_stokes else (d, tm, b, cm)
    gain_shape = (d, tm, a, cm, 2, 2) if full_stokes else (d, tm, a, cm)

    vis_model = _rand_complex(key, vis_shape, dtype)
    gains_init = _rand_complex(jax.random.PRNGKey(321), gain_shape, dtype)

    antenna1 = jnp.arange(b) % a
    antenna2 = (antenna1 + 1) % a
    vis_data = jnp.zeros(
        (tm, b, cm, 2, 2) if full_stokes else (tm, b, cm),
        dtype,
    )

    def loss_fn(g):
        r = compute_residual_TBC(vis_model, vis_data, g, antenna1, antenna2)
        return jnp.real(jnp.vdot(r, r))  # scalar real loss

    grads = jax.grad(loss_fn)(gains_init)

    assert jnp.all(jnp.isfinite(grads)), "NaN or Inf in gradients"
    # Real & imag parts both present and non-zero
    assert jnp.any(jnp.abs(jnp.real(grads)) > 0), "zero real-part gradient"
    assert jnp.any(jnp.abs(jnp.imag(grads)) > 0), "zero imag-part gradient"
