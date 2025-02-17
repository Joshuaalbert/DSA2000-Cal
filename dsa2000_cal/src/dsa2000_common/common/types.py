from typing import NamedTuple, List, Tuple

import numpy as np
from astropy import coordinates as ac, time as at, units as au
from jax._src.partition_spec import PartitionSpec

from dsa2000_common.common.array_types import FloatArray, IntArray
from dsa2000_common.common.serialise_utils import SerialisableBaseModel


class VisibilityCoords(NamedTuple):
    """
    Coordinates for a single visibility.
    """
    uvw: FloatArray | PartitionSpec  # [num_times, num_baselines, 3] the uvw coordinates
    times: FloatArray | PartitionSpec  # [num_times] the time relative to the reference time in TT scale (typically observation start)
    freqs: FloatArray | PartitionSpec  # [num_freqs] the frequency of the visibility
    antenna1: IntArray | PartitionSpec  # [num_baselines] the first antenna
    antenna2: IntArray | PartitionSpec  # [num_baselines] the second antenna


class DishEffectsParams(SerialisableBaseModel):
    # dish parameters
    dish_diameter: au.Quantity = 5. * au.m
    focal_length: au.Quantity = 2. * au.m

    # Dish effect parameters
    elevation_pointing_error_stddev: au.Quantity = 2. * au.arcmin
    cross_elevation_pointing_error_stddev: au.Quantity = 2. * au.arcmin
    axial_focus_error_stddev: au.Quantity = 3. * au.mm
    elevation_feed_offset_stddev: au.Quantity = 3. * au.mm
    cross_elevation_feed_offset_stddev: au.Quantity = 3. * au.mm
    horizon_peak_astigmatism_stddev: au.Quantity = 5. * au.mm
    surface_error_mean: au.Quantity = 3. * au.mm
    surface_error_stddev: au.Quantity = 1. * au.mm

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(DishEffectsParams, self).__init__(**data)
        # Use _check_measurement_set_meta_v0 as instance-wise validator
        _check_dish_effect_params(self)


def _check_dish_effect_params(dish_effect_params: DishEffectsParams):
    # Check units
    assert_congruent_unit(dish_effect_params.dish_diameter, au.m)
    assert_congruent_unit(dish_effect_params.focal_length, au.m)
    assert_congruent_unit(dish_effect_params.elevation_pointing_error_stddev, au.rad)
    assert_congruent_unit(dish_effect_params.cross_elevation_pointing_error_stddev, au.rad)
    assert_congruent_unit(dish_effect_params.axial_focus_error_stddev, au.m)
    assert_congruent_unit(dish_effect_params.elevation_feed_offset_stddev, au.m)
    assert_congruent_unit(dish_effect_params.cross_elevation_feed_offset_stddev, au.m)
    assert_congruent_unit(dish_effect_params.horizon_peak_astigmatism_stddev, au.m)
    assert_congruent_unit(dish_effect_params.surface_error_mean, au.m)
    assert_congruent_unit(dish_effect_params.surface_error_stddev, au.m)
    # Check shapes
    assert_scalar(
        dish_effect_params.dish_diameter,
        dish_effect_params.focal_length,
        dish_effect_params.elevation_pointing_error_stddev,
        dish_effect_params.cross_elevation_pointing_error_stddev,
        dish_effect_params.axial_focus_error_stddev,
        dish_effect_params.elevation_feed_offset_stddev,
        dish_effect_params.cross_elevation_feed_offset_stddev,
        dish_effect_params.horizon_peak_astigmatism_stddev,
        dish_effect_params.surface_error_mean,
        dish_effect_params.surface_error_stddev
    )


def assert_congruent_unit(x: au.Quantity, unit: au.Unit):
    if not isinstance(x, au.Quantity):
        raise ValueError(f"Expected {x} to be an astropy quantity")
    if not x.unit.is_equivalent(unit):
        raise ValueError(f"Expected {x} to be in {unit} units but got {x.unit}")


def assert_same_shapes(*x: au.Quantity, expected_shape: Tuple[int, ...] | None = None):
    if len(x) == 0:
        return
    for xi in x:
        if not isinstance(xi, au.Quantity):
            raise ValueError(f"Expected {xi} to be an astropy quantity")
    if expected_shape is None:
        expected_shape = x[0].shape
    for xi in x:
        if xi.shape != expected_shape:
            raise ValueError(f"Expected {xi} to have shape {expected_shape} but got {xi.shape}")


def assert_scalar(*x: au.Quantity):
    for xi in x:
        if not isinstance(xi, au.Quantity):
            raise ValueError(f"Expected {xi} to be an astropy quantity")
        if not xi.isscalar:
            raise ValueError(f"Expected {xi} to be a scalar but got {xi}")
