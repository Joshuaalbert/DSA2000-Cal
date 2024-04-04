from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from dsa2000_cal.source_models.corr_translation import unflatten_coherencies, flatten_coherencies, linear_to_circular

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


def from_casa_corrs_to_linear(casa_coherencies: jax.Array, corrs: List[int], flat_output: bool = True) -> jax.Array:
    if np.size(casa_coherencies) != len(corrs):
        raise ValueError("The number of correlation types must match the number of correlation data.")
    if np.size(casa_coherencies) == 1 and np.shape(casa_coherencies) == ():
        casa_coherencies = lax.reshape(casa_coherencies, (1,))
    zero = jnp.asarray(0., dtype=casa_coherencies.dtype)
    XX = zero
    XY = zero
    YX = zero
    YY = zero
    RR = zero
    RL = zero
    LR = zero
    LL = zero
    for i, coor in enumerate(corrs):
        if CASA_CORR_TYPES[coor] == "XX":
            XX = casa_coherencies[i]
        elif CASA_CORR_TYPES[coor] == "XY":
            XY = casa_coherencies[i]
        elif CASA_CORR_TYPES[coor] == "YX":
            YX = casa_coherencies[i]
        elif CASA_CORR_TYPES[coor] == "YY":
            YY = casa_coherencies[i]
        elif CASA_CORR_TYPES[coor] == "RR":
            RR = casa_coherencies[i]
        elif CASA_CORR_TYPES[coor] == "RL":
            RL = casa_coherencies[i]
        elif CASA_CORR_TYPES[coor] == "LR":
            LR = casa_coherencies[i]
        elif CASA_CORR_TYPES[coor] == "LL":
            LL = casa_coherencies[i]
    # Detect coor type provided
    linear_provided = len({"XX", "XY", "YX", "YY"}.intersection(set([CASA_CORR_TYPES[coor] for coor in corrs]))) > 0
    circular_provided = len({"RR", "RL", "LR", "LL"}.intersection(set([CASA_CORR_TYPES[coor] for coor in corrs]))) > 0
    if linear_provided and circular_provided:
        raise ValueError("Both linear and circular correlation types provided.")

    if linear_provided:
        output = jnp.asarray([XX, XY, YX, YY])
    elif circular_provided:
        output = jnp.asarray([RR, RL, LR, LL])
    else:
        raise ValueError(f"Strange correlation types provided {corrs}.")

    if flat_output:
        return output

    return unflatten_coherencies(output)


def from_linear_to_casa_corrs(linear_coherencies: jax.Array, corrs: List[int]) -> jax.Array:
    if np.size(linear_coherencies) != 4:
        raise ValueError("Linear coherencies must have 4 elements.")
    if np.shape(linear_coherencies) == (2, 2):
        linear_coherencies = flatten_coherencies(linear_coherencies)
    XX, XY, YX, YY = linear_coherencies
    # Detect coor type provided
    linear_provided = len({"XX", "XY", "YX", "YY"}.intersection(set([CASA_CORR_TYPES[coor] for coor in corrs]))) > 0
    circular_provided = len({"RR", "RL", "LR", "LL"}.intersection(set([CASA_CORR_TYPES[coor] for coor in corrs]))) > 0
    if linear_provided and circular_provided:
        raise ValueError("Both linear and circular correlation types provided.")
    if linear_provided:
        data_dict = {
            INV_CASA_CORR_TYPES["XX"]: XX,
            INV_CASA_CORR_TYPES["XY"]: XY,
            INV_CASA_CORR_TYPES["YX"]: YX,
            INV_CASA_CORR_TYPES["YY"]: YY
        }
        return jnp.asarray([data_dict[coor] for coor in corrs])
    elif circular_provided:
        RR, RL, LR, LL = linear_to_circular(linear_coherencies, flat_output=True)
        data_dict = {
            INV_CASA_CORR_TYPES["RR"]: RR,
            INV_CASA_CORR_TYPES["RL"]: RL,
            INV_CASA_CORR_TYPES["LR"]: LR,
            INV_CASA_CORR_TYPES["LL"]: LL
        }
        return jnp.asarray([data_dict[coor] for coor in corrs])
    else:
        raise ValueError(f"Strange correlation types provided {corrs}.")
