from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_cal.common.corr_translation import linear_to_circular, \
    circular_to_linear, linear_to_stokes, circular_to_stokes, stokes_to_linear, stokes_to_circular

# From https://casa.nrao.edu/active/docs/doxygen/html/classcasa_1_1Stokes.html
CASA_CORR_TYPES = {
    0: "Undefined",  # undefined value
    1: "I",  # Stokes I
    2: "Q",  # Stokes Q
    3: "U",  # Stokes U
    4: "V",  # Stokes V
    5: "RR",  # Right-Right circular polarization
    6: "RL",  # Right-Left circular polarization
    7: "LR",  # Left-Right circular polarization
    8: "LL",  # Left-Left circular polarization
    9: "XX",  # Linear X-X polarization
    10: "XY",  # Linear X-Y polarization
    11: "YX",  # Linear Y-X polarization
    12: "YY",  # Linear Y-Y polarization
    13: "RX",  # Mixed correlation: Right-X
    14: "RY",  # Mixed correlation: Right-Y
    15: "LX",  # Mixed correlation: Left-X
    16: "LY",  # Mixed correlation: Left-Y
    17: "XR",  # Mixed correlation: X-Right
    18: "XL",  # Mixed correlation: X-Left
    19: "YR",  # Mixed correlation: Y-Right
    20: "YL",  # Mixed correlation: Y-Left
    21: "PP",  # Quasi-orthogonal correlation: P-P
    22: "PQ",  # Quasi-orthogonal correlation: P-Q
    23: "QP",  # Quasi-orthogonal correlation: Q-P
    24: "QQ",  # Quasi-orthogonal correlation: Q-Q
    25: "RCircular",  # Single dish: Right Circular
    26: "LCircular",  # Single dish: Left Circular
    27: "Linear",  # Single dish: Linear
    28: "Ptotal",  # Polarized intensity: (Q^2 + U^2 + V^2)^(1/2)
    29: "Plinear",  # Linearly polarized intensity: (Q^2 + U^2)^(1/2)
    30: "PFtotal",  # Polarization Fraction (Ptotal / I)
    31: "PFlinear",  # Linear Polarization Fraction (Plinear / I)
    32: "Pangle",  # Linear Polarization Angle: 0.5 * arctan(U/Q)
}

INV_CASA_CORR_TYPES = {v: k for k, v in CASA_CORR_TYPES.items()}


@partial(jax.jit, static_argnames=("from_corrs", "to_corrs"))
def broadcast_translate_corrs(coherencies: jax.Array,
                              from_corrs: Tuple[int | str, ...] | Tuple[Tuple[int | str, ...], Tuple[int | str, ...]],
                              to_corrs: Tuple[int | str, ...] | Tuple[Tuple[int | str, ...], Tuple[int | str, ...]]):
    """
    Broadcast the translation of coherencies to any array of CASA coherencies

    Args:
        coherencies: [..., num_corrs] array of coherencies.
        from_corrs: list of corrs.
        to_corrs: list of corrs.

    Returns:
        [..., num_corrs] array of coherencies in the type and order of corrs.
    """
    from_corrs_shape = np.shape(from_corrs)
    to_corrs_shape = np.shape(to_corrs)
    output_coherencies_shape = np.shape(coherencies)[:-len(from_corrs_shape)] + to_corrs_shape
    coherencies = jnp.reshape(coherencies, (-1,) + from_corrs_shape)
    coherencies = jax.vmap(partial(translate_corrs, from_corrs=from_corrs, to_corrs=to_corrs))(coherencies)
    coherencies = jax.lax.reshape(coherencies, output_coherencies_shape)
    return coherencies


def translate_corrs(coherencies: jax.Array,
                    from_corrs: List[int | str] | List[List[int | str]],
                    to_corrs: List[int | str] | List[List[int | str]]) -> jax.Array:
    """
    Convert linear coherencies to any array of CASA coherencies.

    Args:
        coherencies: any array of coherencies.
        from_corrs: any array of CASA coherencies.
        to_corrs: any array of CASA coherencies.

    Returns:
        coherencies in the type and order of corrs.
    """

    _from_corrs = from_corrs
    _to_corrs = to_corrs
    from_corrs, from_treedef = jax.tree.flatten(from_corrs)
    from_corrs = [(INV_CASA_CORR_TYPES[coor] if isinstance(coor, str) else coor) for coor in from_corrs]
    to_corrs, to_treedef = jax.tree.flatten(to_corrs)
    to_corrs = [(INV_CASA_CORR_TYPES[coor] if isinstance(coor, str) else coor) for coor in to_corrs]

    if np.size(coherencies) != len(from_corrs):
        raise ValueError(f"Input coherencies {np.shape(coherencies)} must match input coors {_from_corrs}.")

    if _from_corrs == _to_corrs:
        return coherencies

    if len(np.shape(coherencies)) != 1:
        coherencies = jnp.reshape(coherencies, (-1,))

    # Detect mismatch in coor type provided
    if detect_mixed_corrs(from_corrs):
        raise ValueError(f"Mixed correlation input types provided, {from_corrs}.")
    if detect_mixed_corrs(to_corrs):
        raise ValueError(f"Mixed correlation output types provided, {to_corrs}.")

    # Fill data dict with zeros
    zero = jnp.asarray(0., dtype=coherencies.dtype)
    data_dict = {
        CASA_CORR_TYPES[coor]: zero for coor in INV_CASA_CORR_TYPES.values()
    }
    # Update data dict with input coherencies
    data_dict.update(
        {
            CASA_CORR_TYPES[coor]: coherencies[i] for coor, i in zip(from_corrs, range(len(coherencies)))
        }
    )

    if is_linear_present(from_corrs):
        if is_circular_present(to_corrs):
            coh_circ = linear_to_circular(
                jnp.stack([data_dict['XX'], data_dict['XY'], data_dict['YX'], data_dict['YY']]),
                flat_output=True)
            data_dict.update(
                {
                    coor: coh_circ[i] for coor, i in
                    zip(['RR', 'RL', 'LR', 'LL'], range(len(coh_circ)))
                }
            )
        elif is_stokes_present(to_corrs):
            coh_stokes = linear_to_stokes(
                jnp.stack([data_dict['XX'], data_dict['XY'], data_dict['YX'], data_dict['YY']],
                          axis=-1), flat_output=True)
            data_dict.update(
                {
                    coor: coh_stokes[i] for coor, i in
                    zip(['I', 'Q', 'U', 'V'], range(len(coh_stokes)))
                }
            )
    elif is_circular_present(from_corrs):
        if is_linear_present(to_corrs):
            coh_lin = circular_to_linear(
                jnp.stack([data_dict['RR'], data_dict['RL'], data_dict['LR'], data_dict['LL']]),
                flat_output=True)
            data_dict.update(
                {
                    coor: coh_lin[i] for coor, i in zip(['XX', 'XY', 'YX', 'YY'], range(len(coh_lin)))
                }
            )
        elif is_stokes_present(to_corrs):
            coh_stokes = circular_to_stokes(
                jnp.stack([data_dict['RR'], data_dict['RL'], data_dict['LR'], data_dict['LL']],
                          axis=-1), flat_output=True)
            data_dict.update(
                {
                    coor: coh_stokes[i] for coor, i in
                    zip(['I', 'Q', 'U', 'V'], range(len(coh_stokes)))
                }
            )
    elif is_stokes_present(from_corrs):
        if is_linear_present(to_corrs):
            coh_lin = stokes_to_linear(
                jnp.stack([data_dict['I'], data_dict['Q'], data_dict['U'], data_dict['V']], axis=-1),
                flat_output=True)
            data_dict.update(
                {
                    coor: coh_lin[i] for coor, i in zip(['XX', 'XY', 'YX', 'YY'], range(len(coh_lin)))
                }
            )
        elif is_circular_present(to_corrs):
            coh_circ = stokes_to_circular(
                jnp.stack([data_dict['I'], data_dict['Q'], data_dict['U'], data_dict['V']], axis=-1),
                flat_output=True)
            data_dict.update(
                {
                    coor: coh_circ[i] for coor, i in
                    zip(['RR', 'RL', 'LR', 'LL'], range(len(coh_circ)))
                }
            )

    # Data dict is full now, construct the result
    output = [data_dict[CASA_CORR_TYPES[coor]] for coor in to_corrs]
    return jnp.asarray(jax.tree.unflatten(to_treedef, output), coherencies.dtype)


def is_linear_present(corrs: List[str]):
    """
    Detect if linear corrs are present in the provided corrs.

    Args:
        corrs: list of corrs.

    Returns:
        True if linear corrs are present, False otherwise.
    """
    corrs = [CASA_CORR_TYPES[corr] if isinstance(corr, int) else corr for corr in corrs]
    linear_corrs = {"XX", "XY", "YX", "YY"}
    return len(linear_corrs.intersection(set(corrs))) > 0


def is_circular_present(corrs: List[str]):
    """
    Detect if circular corrs are present in the provided corrs.

    Args:
        corrs: list of corrs.

    Returns:
        True if circular corrs are present, False otherwise.
    """
    corrs = [CASA_CORR_TYPES[corr] if isinstance(corr, int) else corr for corr in corrs]
    circular_corrs = {"RR", "RL", "LR", "LL"}
    return len(circular_corrs.intersection(set(corrs))) > 0


def is_stokes_present(corrs: List[str]):
    """
    Detect if stokes corrs are present in the provided corrs.

    Args:
        corrs: list of corrs.

    Returns:
        True if stokes corrs are present, False otherwise.
    """
    corrs = [CASA_CORR_TYPES[corr] if isinstance(corr, int) else corr for corr in corrs]
    stokes_corrs = {"I", "Q", "U", "V"}
    return len(stokes_corrs.intersection(set(corrs))) > 0


def detect_mixed_corrs(corrs: List[str]):
    """
    Detect if mixed corrs are present in the provided corrs.

    Args:
        corrs: list of corrs.

    Returns:
        True if mixed corrs are present, False otherwise.
    """
    corrs = [CASA_CORR_TYPES[corr] if isinstance(corr, int) else corr for corr in corrs]
    linear_present = is_linear_present(corrs)
    circular_present = is_circular_present(corrs)
    stokes_present = is_stokes_present(corrs)
    if linear_present:
        if circular_present or stokes_present:
            return True
    if circular_present:
        if linear_present or stokes_present:
            return True
    if stokes_present:
        if linear_present or circular_present:
            return True
    return False
