import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as au
from jax._src.typing import SupportsDType

from dsa2000_cal.common.types import float_type, mp_policy


def quantity_to_jnp(q: au.Quantity, decompose_unit: au.Unit | str | None = None,
                    dtype: SupportsDType | str | None = None) -> jax.Array:
    """
    Convert an astropy quantity to a jax numpy array.

    Args:
        q: astropy quantity
        decompose_unit: astropy unit or str, optional. Default uses SI bases.
        dtype: jax dtype or str, optional. Default uses the quantity dtype.

    Returns:
        jax numpy array
    """
    if not isinstance(q, au.Quantity):
        raise ValueError(f"Expected astropy quantity, got {type(q)}")

    if dtype is None:
        if q.unit.is_equivalent('m'):
            dtype = mp_policy.position_dtype
        elif q.unit.is_equivalent('s'):
            dtype = mp_policy.time_dtype
        elif q.unit.is_equivalent('Hz'):
            dtype = mp_policy.frequency_dtype
        elif q.unit.is_equivalent('deg'):
            dtype = mp_policy.lmn_dtype
        elif q.unit.is_equivalent(au.dimensionless_unscaled):
            dtype = mp_policy.lmn_dtype
        else:
            dtype = float_type

    if decompose_unit is None:
        q = q.decompose(au.si.bases)
        return jnp.asarray(q.value, dtype=dtype)
    if not q.unit.is_equivalent(decompose_unit):
        raise ValueError(f"Expected equivalent unit {decompose_unit}, got {q.unit}")
    q = q.to(decompose_unit)
    return jnp.asarray(q.value, dtype=dtype)


def quantity_to_np(q: au.Quantity, decompose_unit: au.Unit | str | None = None,
                   dtype: SupportsDType | str | None = None) -> np.ndarray:
    """
    Convert an astropy quantity to a numpy array.

    Args:
        q: astropy quantity
        decompose_unit: astropy unit or str, optional. Default uses SI bases.
        dtype: jax dtype or str, optional. Default uses the quantity dtype.

    Returns:
        numpy array
    """
    if not isinstance(q, au.Quantity):
        raise ValueError(f"Expected astropy quantity, got {type(q)}")
    if decompose_unit is None:
        q = q.decompose(au.si.bases)
        # Use default unit
        if dtype is not None:
            return np.asarray(q.value, dtype=dtype)
        return np.asarray(q.value)
    if not q.unit.is_equivalent(decompose_unit):
        raise ValueError(f"Expected equivalent unit {decompose_unit}, got {q.unit}")
    q = q.to(decompose_unit)
    if dtype is not None:
        return np.asarray(q.value, dtype=dtype)
    return np.asarray(q.value)
