import jax
import numpy as np
from astropy import units as au, time as at, coordinates as ac

from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel


def check_cache(cache_model: SerialisableBaseModel, **kwargs):
    """
    Check that the cache model matches the expected values

    Args:
        cache_model: the cache model to check
        **kwargs: the expected values
    """
    print(f"Checking cached {cache_model.__class__.__name__} ...")
    for key, value in kwargs.items():
        # Assert that cached_values match
        cached_value = getattr(cache_model, key)
        print(f"Checking {key}...")
        print(f"Expected: {value}")
        print(f"Got: {cached_value}")
        if cached_value.__class__ != value.__class__:
            raise ValueError(f"Expected {key} to be of type {value.__class__} but got {cached_value.__class__}")
        if value.__class__ == au.Quantity:
            # print("quantity")
            if not cached_value.unit.is_equivalent(value.unit):
                raise ValueError(f"Expected {key} to have units {value.unit} but got {cached_value.unit}")
            sep = cached_value - value
            np.testing.assert_allclose(sep, 0.)
        elif value.__class__ == at.Time:
            # print('time')
            sep = (cached_value.tt - value.tt).sec
            np.testing.assert_allclose(sep, 0., atol=1e-3)
        elif value.__class__ == ac.EarthLocation:
            # print('location')
            # Get 3D seperation
            sep = cached_value.get_itrs().separation_3d(value.get_itrs())
            # print(sep, cached_value.get_itrs(), value.get_itrs())
            np.testing.assert_allclose(sep, 0., atol=1e-3 * au.m)
        elif value.__class__ == ac.ICRS:
            # print('icrs')
            sep = cached_value.separation(value)
            np.testing.assert_allclose(sep, 0., atol=1 * au.arcsec)
        elif value.__class__ == InterpolatedArray:
            # print('interpolated array')
            assert value.axis == cached_value.axis
            assert value.regular_grid == cached_value.regular_grid
            np.testing.assert_allclose(cached_value.x, value.x)
            np.testing.assert_allclose(cached_value.values, value.values)
        elif value.__class__ == np.ndarray:
            np.testing.assert_allclose(cached_value, value)
        elif isinstance(value, jax.Array):
            np.testing.assert_allclose(cached_value, value)
        elif isinstance(value, SerialisableBaseModel):
            check_cache(cached_value, **value.dict())
        else:
            raise ValueError(f"Unsupported type {value.__class__} for {key}")
