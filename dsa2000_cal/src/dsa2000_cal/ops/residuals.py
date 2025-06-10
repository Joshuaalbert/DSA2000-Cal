import jax
import numpy as np
from jax import numpy as jnp

from dsa2000_common.common.mixed_precision_utils import mp_policy
from dsa2000_common.common.sum_utils import scan_sum
from dsa2000_common.common.vec_utils import kron_product_2x2



# TBC ordering is [..., Tm, B, Cm, ...]
# BTC ordering is [..., B, Tm, Cm, ...]

def conj_transpose(a):
    perm = np.arange(np.ndim(a))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return jax.lax.transpose(jax.lax.conj(a), list(perm))


def apply_gains_to_model_vis_TBC(vis_model, gains, antenna1, antenna2):
    """
    Compute the residual between the model visibilities and the observed visibilities.

    TBC refers to the ordering.

    Args:
        vis_model: [D, Tm, B, Cm[,2,2]] the model visibilities per direction
        gains: [D, Tm/1, A, Cm/1[,2,2]] the gains
        antenna1: [B] the antenna1
        antenna2: [B] the antenna2

    Returns:
        [Tm, B, Cm[, 2, 2]] the residuals
    """

    if len(np.shape(gains)) == 6 and np.shape(gains)[-2:] == (2, 2):
        full_stokes = True
    elif len(np.shape(gains)) == 4:
        full_stokes = False
    else:
        raise ValueError(f"Invalid gains shape {np.shape(gains)}. "
                         f"Expected 4 or 6 dimensions with last two being (2, 2) for full stokes.")

    def accum_fn(x):
        vis_model, gains = x

        g1 = gains[:, antenna1, :, ...]  # [D, Tm, B, Cm[, 2, 2]]
        g2 = gains[:, antenna2, :, ...]  # [D, Tm, B, Cm[, 2, 2]]

        if full_stokes:
            delta_vis = mp_policy.cast_to_vis(
                kron_product_2x2(g1, vis_model, conj_transpose(g2)))  # [Tm, B, Cm[, 2, 2]]
        else:
            delta_vis = mp_policy.cast_to_vis((g1 * jax.lax.conj(g2)) * vis_model)  # [Tm, B, Cm]

        return delta_vis

    if np.shape(vis_model)[0] != np.shape(gains)[0]:
        raise ValueError(
            f"Model visibilities and gains must have the same number of directions, "
            f"got {np.shape(vis_model)[0]} and {np.shape(gains)[0]}")

    zeros = jnp.zeros(np.shape(vis_model)[1:], dtype=mp_policy.vis_dtype)
    accumulate = scan_sum(accum_fn, zeros, (vis_model, gains), unroll=2)
    return accumulate


def compute_residual_TBC(vis_model, vis_data, gains, antenna1, antenna2):
    """
    Compute the residual between the model visibilities and the observed visibilities.

    Args:
        vis_model: [D, Tm, B, Cm[,2,2]] the model visibilities per direction
        vis_data: [Tm, B, Cm[,2,2]] the data visibilities, Ts = 0 mod Tm, Cs = 0 mod Cm i.e. Ts % Tm = 0, Cs % Cm = 0
        gains: [D, Ts, A, Cs[,2,2]] the gains
        antenna1: [B] the antenna1
        antenna2: [B] the antenna2

    Returns:
        [Ts, B, Cs[, 2, 2]] the residuals
    """
    if np.shape(vis_model)[1:] != np.shape(vis_data):
        raise ValueError("The model visibilities and data must have the same shape.")
    D, Tm, B, Cm = np.shape(vis_model)[:4]
    D_, Ts, A, Cs = np.shape(gains)[:4]
    if D != D_:
        raise ValueError(f"Number of model and gain directions mismatch {D} and {D_}")
    if Tm % Ts != 0 or Cm % Cs != 0:
        raise ValueError(f"Model dimensions ({Tm}, {Cm}) not compatible with gain dimensions ({Ts}, {Cs}).")

    # Replicate gains if necessary. Broadcasting is used otherwise if Ts=1 and Cs=1
    if Ts > 1 and Ts != Tm:
        time_reps = Tm // Ts
        gains = jnp.repeat(gains, time_reps, axis=1)

    if Cs > 1 and Cs != Cm:
        freq_reps = Cm // Cs
        gains = jnp.repeat(gains, freq_reps, axis=3)

    accumulate = apply_gains_to_model_vis_TBC(vis_model, gains, antenna1, antenna2)  # [Tm, B, Cm[, 2, 2]]

    if np.shape(accumulate) != np.shape(vis_data):
        raise ValueError(
            f"Accumulate {np.shape(accumulate)} and vis_data {np.shape(vis_data)} must have the same shape.")
    return vis_data - accumulate


def apply_gains_to_model_vis_BTC(vis_model, gains, antenna1, antenna2):
    """
    Compute the residual between the model visibilities and the observed visibilities.

    Args:
        vis_model: [D, B, Tm, Cm[,2,2]] the model visibilities per direction
        gains: [D, Tm, A, Cm[,2,2]] the gains
        antenna1: [B] the antenna1
        antenna2: [B] the antenna2

    Returns:
        [B, Tm, Cm[, 2, 2]] the residuals
    """
    # Note: non-transposed is 1/3 faster
    full_stokes = len(np.shape(gains)) == 6 and np.shape(gains)[-2:] == (2, 2)

    # Note: I found a scan is same or faster than:
    # g1 = gains[:, antenna1, :, ...]
    # g2 = gains[:, antenna2, :, ...]
    # return kron_product_2x2(g1, vis_model, conj_transpose(g2))
    # Hence for better memory bounding we use scan.

    def body_fn(accumulate, x):
        vis_model, gains = x

        # Note: I found it is faster to index inside the scan
        g1 = gains[antenna1, ...]  # [D, B, Tm, Cm[, 2, 2]]
        g2 = gains[antenna2, ...]  # [D, B, Tm, Cm[, 2, 2]]

        if full_stokes:
            delta_vis = mp_policy.cast_to_vis(
                kron_product_2x2(g1, vis_model, conj_transpose(g2)))  # [B, Tm, Cm[, 2, 2]]
        else:
            delta_vis = mp_policy.cast_to_vis((g1 * jax.lax.conj(g2)) * vis_model)  # [B, Tm, Cm]

        return accumulate + delta_vis, ()

    if np.shape(vis_model)[0] != np.shape(gains)[0]:
        raise ValueError(
            f"Model visibilities and gains must have the same number of directions, got {np.shape(vis_model)[0]} and {np.shape(gains)[0]}")

    accumulate = jnp.zeros(np.shape(vis_model)[1:], dtype=vis_model.dtype)
    accumulate, _ = jax.lax.scan(body_fn, accumulate, (vis_model, gains), unroll=2)
    return accumulate


def compute_residual_BTC(vis_model, vis_data, gains, antenna1, antenna2):
    """
    Compute the residual between the model visibilities and the observed visibilities.

    Args:
        vis_model: [D, B, Tm,  Cm[,2,2]] the model visibilities per direction
        vis_data: [ B, Ts,Cs[,2,2]] the data visibilities, Ts = 0 mod Tm, Cs = 0 mod Cm i.e. Ts % Tm = 0, Cs % Cm = 0
        gains: [D,  A, Tm, Cm[,2,2]] the gains
        antenna1: [B] the antenna1
        antenna2: [B] the antenna2

    Returns:
        [B, Ts,  Cs[, 2, 2]] the residuals
    """

    accumulate = apply_gains_to_model_vis_BTC(vis_model, gains, antenna1, antenna2)

    Ts = np.shape(vis_data)[1]
    Tm = np.shape(accumulate)[1]
    Cs = np.shape(vis_data)[2]
    Cm = np.shape(accumulate)[2]
    if Tm > 1 and Ts != Tm:
        time_reps = Ts // Tm
        accumulate = jnp.repeat(accumulate, time_reps, axis=1)
    if Cm > 1 and Cs != Cm:
        freq_reps = Cs // Cm
        accumulate = jnp.repeat(accumulate, freq_reps, axis=2)
    return vis_data - accumulate
