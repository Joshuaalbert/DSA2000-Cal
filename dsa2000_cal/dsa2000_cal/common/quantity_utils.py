import jax
import jax.numpy as jnp
from astropy import units as au
from jax._src.typing import SupportsDType


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
    if decompose_unit is None:
        q = q.decompose(au.si.bases)
        # Use default unit
        if dtype is not None:
            return jnp.asarray(q.value, dtype=dtype)
        return jnp.asarray(q.value)
    if not q.unit.is_equivalent(decompose_unit):
        raise ValueError(f"Expected equivalent unit {decompose_unit}, got {q.unit}")
    q = q.to(decompose_unit)
    if dtype is not None:
        return jnp.asarray(q.value, dtype=dtype)
    return jnp.asarray(q.value)
