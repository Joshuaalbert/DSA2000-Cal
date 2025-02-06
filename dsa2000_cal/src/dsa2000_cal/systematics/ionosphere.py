import dataclasses
from typing import NamedTuple, Dict

import jax
import jax.numpy as jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel

from dsa2000_cal.common.array_types import FloatArray
from dsa2000_cal.common.interp_utils import InterpolatedArray


@dataclasses.dataclass(eq=False)
class Ionosphere:
    """
    Simulates the ionosphere using frozen flow model, and then interpolates the resulting TEC onto a regular grid.
    """

    model_antennas_gcrs: InterpolatedArray  # (time) -> [num_model_ant, 3]
    model_directions_gcrs: InterpolatedArray  # (time) -> [num_model_dir, 3]

def compute_ionosphere_intersection(
        x_gcrs, k_gcrs,
        x0_gcrs,
        bottom: FloatArray, width: FloatArray
):


    # |x(smin)| = |x + smin * k| = |x0| + bottom = v
    # ==> |x + smin * k|^2 = (|x0| + bottom)^2 = v^2
    # ==> x^2 + 2 smin x . k + smin^2 - v^2 = 0
    # ==> smin = - (x . k) +- sqrt((x . k)^2 +(x^2 - v^2))
    # choose the positive root
    xk = x_gcrs @ k_gcrs
    xx = x_gcrs @ x_gcrs
    x0_norm = jnp.linalg.norm(x0_gcrs)
    vmin = x0_norm + bottom
    vmax = vmin + width

    smin = - xk + jnp.sqrt(xk**2 - (xx - vmin**2))
    smax = - xk + jnp.sqrt(xk**2 - (xx - vmax**2))
    return smin, smax

def test_compute_ionosphere_intersection():
    x_gcrs = jnp.array([0.0, 0.0, 0.0])
    k_gcrs = jnp.array([0.0, 0.0, 1.0])
    x0_gcrs = jnp.array([0.0, 0.0, 0.0])
    bottom = 0.5
    width = 1.0
    smin, smax = compute_ionosphere_intersection(x_gcrs, k_gcrs, x0_gcrs, bottom, width)
    assert smin == 0.5
    assert smax == 1.5


def compute_covariance(
        kernel_fn,
        x1_gcrs, k1_gcrs, t1,
        x2_gcrs, k2_gcrs, t2,
        x0_gcrs,
        bottom: FloatArray, width: FloatArray, wind_velocity: FloatArray,
        resolution: int
) -> FloatArray:
    """
    Compute the TEC covariance.

    Args:
        x1_gcrs: [3] array of the first position in GCRS
        k1_gcrs: [3] array of the first direction in GCRS, normed
        t1: time of the first position
        x2_gcrs: [3] array of the second position in GCRS
        k2_gcrs: [3] array of the second direction in GCRS, normed
        t2: time of the second position
    """
    # Compute the intersection of the ionosphere with the line of sight
    s1min, s1max = compute_ionosphere_intersection(x1_gcrs, k1_gcrs, x0_gcrs, bottom, width)
    s2min, s2max = compute_ionosphere_intersection(x2_gcrs, k2_gcrs, x0_gcrs, bottom, width)
    # compute the double integral
    s1 = jnp.linspace(s1min, s1max, resolution+1)
    s2 = jnp.linspace(s2min, s2max, resolution+1)
    ds1 = (s1max - s1min) / resolution
    ds2 = (s2max - s2min) / resolution
    # Kernel function over pairs
    K_12 = jax.vmap(
        jax.vmap(
            lambda s1, s2: kernel_fn(
                x1_gcrs + s1 * k1_gcrs,
                x2_gcrs + s2 * k2_gcrs,
            ), in_axes=(None, 0)
        ), in_axes=(0, None)
    )(s1, s2) # [resolution+1, resolution+1]
    covariance = jnp.sum(K_12) * ds1 * ds2
    # mean1 = fed_mu * (s1max - s1min)


class IonosphereLayer(NamedTuple):
    bottom: FloatArray
    width: FloatArray
    wind_velocity: FloatArray
    kernel_params: Dict[str, FloatArray]