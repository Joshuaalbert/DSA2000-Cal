import jax
import numpy as np

from src.dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel


def test_base_geodesic_model_pytree():
    ra0 = 0.0
    dec0 = 0.0
    antennas_enu = np.zeros((10, 3))
    lmn_zenith = InterpolatedArray(np.ones(5), np.zeros((5, 3)))
    lmn_pointings = InterpolatedArray(np.ones(5), np.zeros((5, 10, 3)))
    tile_antennas = False
    base_geodesic_model = BaseGeodesicModel(ra0, dec0, antennas_enu, lmn_zenith, lmn_pointings,
                                            tile_antennas=tile_antennas)
    leaves, treedef = jax.tree.flatten(base_geodesic_model)
    pytree = jax.tree.unflatten(treedef, leaves)
    print(pytree)

    @jax.jit
    def f(model):
        return model

    base_geodesic_model = block_until_ready(f(base_geodesic_model))
