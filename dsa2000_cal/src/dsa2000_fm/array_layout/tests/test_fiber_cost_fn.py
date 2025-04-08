import time

import networkx as nx
import numpy as np
from astropy import time as at, coordinates as ac

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_fm.array_layout.fiber_cost_fn import compute_mst_cost, compute_flows, compute_mst_with_fiber_costs


def test_compute_mst():
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa1650_9P'))
    antennas = array.get_antennas()
    obstime = at.Time.now()
    array_location = array.get_array_location()
    root_antenna = ac.EarthLocation.from_geodetic(
        lon=np.min(antennas.geodetic.lon), lat=np.mean(antennas.geodetic.lat),
        height=array_location.geodetic.height
    )
    antennas = ac.EarthLocation(
        x=np.concatenate([root_antenna.x[None], antennas.x]),
        y=np.concatenate([root_antenna.y[None], antennas.y]),
        z=np.concatenate([root_antenna.z[None], antennas.z])
    )

    t0 = time.time()
    _ = compute_mst_cost(6, antennas, obstime, array_location)
    print(f"Time taken to compute MST: {time.time() - t0:.2f} seconds")

def test_compute_mst_with_fiber_costs():
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa1650_9P'))
    antennas = array.get_antennas()
    obstime = at.Time.now()
    array_location = array.get_array_location()
    root_antenna = ac.EarthLocation.from_geodetic(
        lon=np.min(antennas.geodetic.lon), lat=np.mean(antennas.geodetic.lat),
        height=array_location.geodetic.height
    )
    antennas = ac.EarthLocation(
        x=np.concatenate([root_antenna.x[None], antennas.x]),
        y=np.concatenate([root_antenna.y[None], antennas.y]),
        z=np.concatenate([root_antenna.z[None], antennas.z])
    )

    t0 = time.time()
    _ = compute_mst_with_fiber_costs(20, antennas, obstime, array_location,
                                     target_node=0,
                                     plot=True
                                     )
    print(f"Time taken to compute MST: {time.time() - t0:.2f} seconds")



def test_compute_flows():
    # Construct the tree:
    #        1
    #      / | \
    #     2  3  4
    #    / \
    #   5   6
    #
    # Expected flows (with 1 as the root):
    # - Edge (1,2) should carry a flow of 3 (nodes 2, 5, and 6).
    # - Edges (1,3) and (1,4) should carry a flow of 1 each.
    # - Edges (2,5) and (2,6) should carry a flow of 1 each.

    mst = nx.Graph()
    mst.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 5), (2, 6)])

    target = 1
    flows = compute_flows(mst, target)

    expected_flows = {
        (1, 2): 3,
        (1, 3): 1,
        (1, 4): 1,
        (2, 5): 1,
        (2, 6): 1,
    }

    # Assert that the computed flows match the expected values.
    assert flows == expected_flows
