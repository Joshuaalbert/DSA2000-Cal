import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as au, time as at
from jax._src.typing import SupportsDType

from dsa2000_cal.common.mixed_precision_utils import float_type, int_type, complex_type, mp_policy


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

    if dtype is None:
        if q.unit.is_equivalent('m'):
            dtype = mp_policy.length_dtype
        elif q.unit.is_equivalent('s'):
            dtype = mp_policy.time_dtype
        elif q.unit.is_equivalent('m/s'):
            dtype = mp_policy.length_dtype
        elif q.unit.is_equivalent('Hz'):
            dtype = mp_policy.freq_dtype
        elif q.unit.is_equivalent('deg'):
            dtype = mp_policy.angle_dtype
        else:
            if np.issubdtype(q.value.dtype, jnp.complexfloating):
                dtype = complex_type
            elif np.issubdtype(q.value.dtype, np.floating):
                dtype = float_type
            elif np.issubdtype(q.value.dtype, np.integer):
                dtype = int_type

    if decompose_unit is None:
        q = q.decompose(au.si.bases)
        return np.asarray(q.value, dtype=dtype)
    if not q.unit.is_equivalent(decompose_unit):
        raise ValueError(f"Expected equivalent unit {decompose_unit}, got {q.unit}")
    q = q.to(decompose_unit)
    return np.asarray(q.value, dtype=dtype)


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
    return jnp.asarray(quantity_to_np(q, decompose_unit, dtype))


def time_to_jnp(t: at.Time, ref_time: at.Time) -> jax.Array:
    """
    Convert an astropy time to a jax numpy array.

    Args:
        t: the astropy time
        ref_time: a reference time

    Returns:
        jax numpy array
    """
    return quantity_to_jnp((t.tt - ref_time.tt) * au.s)


