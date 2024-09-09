from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_cal.common.corr_translation import linear_to_circular, \
    circular_to_linear

CASA_CORR_TYPES = {
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
}

INV_CASA_CORR_TYPES = {v: k for k, v in CASA_CORR_TYPES.items()}


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
        raise ValueError(f"Input coherencies must match input coors {_from_corrs}.")

    if _from_corrs == _to_corrs:
        return coherencies

    if len(np.shape(coherencies)) != 1:
        coherencies = jnp.reshape(coherencies, (-1,))

    # Detect mismatch in coor type provided
    linear_provided_from = len(
        {"XX", "XY", "YX", "YY"}.intersection(set([CASA_CORR_TYPES[coor] for coor in from_corrs]))) > 0
    circular_provided_from = len(
        {"RR", "RL", "LR", "LL"}.intersection(set([CASA_CORR_TYPES[coor] for coor in from_corrs]))) > 0
    if linear_provided_from and circular_provided_from:
        raise ValueError(f"Both linear and circular correlation input types provided, {from_corrs}.")
    linear_provided_to = len(
        {"XX", "XY", "YX", "YY"}.intersection(set([CASA_CORR_TYPES[coor] for coor in to_corrs]))) > 0
    circular_provided_to = len(
        {"RR", "RL", "LR", "LL"}.intersection(set([CASA_CORR_TYPES[coor] for coor in to_corrs]))) > 0
    if linear_provided_to and circular_provided_to:
        raise ValueError(f"Both linear and circular correlation output types provided, {to_corrs}.")

    zero = jnp.asarray(0., dtype=coherencies.dtype)
    data_dict = {
        CASA_CORR_TYPES[coor]: zero for coor in INV_CASA_CORR_TYPES.values()
    }
    data_dict.update(
        {
            CASA_CORR_TYPES[coor]: coherencies[i] for coor, i in zip(from_corrs, range(len(coherencies)))
        }
    )

    if linear_provided_from and circular_provided_to:
        coh_circ = linear_to_circular(jnp.stack([data_dict['XX'], data_dict['XY'], data_dict['YX'], data_dict['YY']]),
                                      flat_output=True)
        data_dict.update(
            {
                CASA_CORR_TYPES[coor]: coh_circ[i] for coor, i in zip(['RR', 'RL', 'LR', 'LL'], range(len(coh_circ)))
            }
        )
    elif circular_provided_from and linear_provided_to:
        coh_lin = circular_to_linear(jnp.stack([data_dict['RR'], data_dict['RL'], data_dict['LR'], data_dict['LL']]),
                                     flat_output=True)
        data_dict.update(
            {
                CASA_CORR_TYPES[coor]: coh_lin[i] for coor, i in zip(['XX', 'XY', 'YX', 'YY'], range(len(coh_lin)))
            }
        )

    # Data dict is full now, construct the result
    output = [data_dict[CASA_CORR_TYPES[coor]] for coor in to_corrs]
    return jnp.asarray(jax.tree.unflatten(to_treedef, output), coherencies.dtype)
