import networkx as nx
import numpy as np
import pylab as plt
from astropy import coordinates as ac, time as at, units as au
from scipy.spatial import KDTree

from dsa2000_common.common.enu_frame import ENU
from dsa2000_fm.array_layout.sample_constraints import haversine


def compute_flows(mst, target):
    flows = {}  # To store flow for each edge (u, v) where u is parent and v is child

    def dfs(node, parent):
        # Start with count 1 for the current node
        count = 1
        for neighbor in mst.neighbors(node):
            if neighbor == parent:
                continue
            child_count = dfs(neighbor, node)
            # The flow on the edge from node to neighbor is the size of neighbor's subtree
            flows[(node, neighbor)] = child_count
            # If your graph is undirected and you need both directions, you might also store flows[(neighbor, node)] = child_count
            count += child_count
        return count

    dfs(target, None)
    return flows


def compute_mst_costed(G, target_node: int = 0):
    # G has edges with 'distance' in m
    for u, v in G.edges():
        G[u][v]['flow'] = 1
        G[u][v]['cost'] = 0

    fiber_map = dict(
        (k, v) for k, v in [
            ((0, 0), 0),
            ((1, 2), 12),
            ((3, 4), 24),
            ((4, 8), 48),
            ((8, 16), 96),
            ((17, 24), 144),
            ((25, 48), 288),
            ((49, 72), 432),
            ((73, 144), 864),
            ((145, 288), 1728)
        ]
    )

    def fiber_cost_fn(n_signal, dist):
        # for a, b in fiber_map.keys():
        #     if a <= n_signal <= b:
        #         n_fiber = fiber_map[(a, b)]
        #         break
        # else:
        n_fiber = n_signal * 6
        return dist * n_fiber * 0.35 / 96

    def distance_cost_fn(dist):
        return dist * 12.5

    mst = nx.minimum_spanning_tree(G, algorithm='kruskal', weight='distance')

    # Iteratively compute the MST with flow cost estimated per edge
    for iter_idx in range(50):
        # for each node find shorted path to target and add 1 to flow for each edge traversed
        flow = compute_flows(mst, target_node)
        for i, j in flow:
            G[i][j]['flow'] = flow[(i, j)]

        # We will slowly add in the flow cost using the previous MST for flow computation
        for u, v in G.edges():
            G[u][v]['cost'] = (
                    fiber_cost_fn(G[u][v]['flow'], G[u][v]['distance'])
                    + distance_cost_fn(G[u][v]['distance'])
            )
        mst = nx.minimum_spanning_tree(G, algorithm='kruskal', weight='cost')
        total_cost = sum([mst.edges[i, j]['cost'] for i, j in mst.edges()])
        print(f"Iteration {iter_idx}: Total Cost {total_cost:.2f}")

    # add distance edges
    for u, v in mst.edges():
        mst[u][v]['log10_distance'] = np.log10(G[u][v]['distance'])
    # add edge flow
    for u, v in mst.edges():
        mst[u][v]['log10_flow'] = np.log10(G[u][v]['flow'])
    return mst


def compute_mst_cost(k: int, antennas: ac.EarthLocation, obstime: at.Time, array_location: ac.EarthLocation):
    """
    Compute the minimal spanning tree of the array, calculate edge and node flow counts,
    and optionally plot the MST colored by flow counts.

    Each node (except the target) sends a signal to the target node, and the flow
    along each edge is the cumulative count of signals passing through that edge.
    Each node's flow count is given by the number of signals that pass through it
    (its subtree size when the MST is rooted at the target).

    Args:
        antennas: the antennas
        obstime: the observation time
        array_location: the location of the array

    Returns:
        total_distance: the total distance of the minimal spanning tree
        node_angles: the maximal angles between edges that connect to each node
        connections: the number of connections for each node
        edge_flow: a numpy array of shape [E] with the flow count on each edge in the MST
    """

    earth_radius = np.linalg.norm(array_location.get_itrs().cartesian.xyz.to(au.m).value)
    antennas_enu_xyz = antennas.get_itrs(obstime=obstime, location=array_location).transform_to(
        ENU(obstime=obstime, location=array_location)
    ).cartesian.xyz.T.to('m').value

    G = nx.Graph()

    # Compute the cost for k nearest neighbors, k large enough to encompass MST
    tree_kd = KDTree(antennas_enu_xyz)
    nn_dists, nn_idxs = tree_kd.query(antennas_enu_xyz, k=k + 1)
    nn_idxs = nn_idxs[:, 1:]
    # nn_dists = nn_dists[:, 1:]

    antenna_lon = antennas.geodetic.lon.to('rad').value
    antenna_lat = antennas.geodetic.lat.to('rad').value
    nn_dists_haversine = earth_radius * haversine(
        antenna_lon[:, None],
        antenna_lat[:, None],
        antenna_lon[nn_idxs],
        antenna_lat[nn_idxs]
    )  # [N, k]

    for i in range(antennas_enu_xyz.shape[0]):
        for dist, j in zip(nn_dists_haversine[i], nn_idxs[i]):
            G.add_edge(i, int(j), distance=float(dist))

    mst = nx.minimum_spanning_tree(G, algorithm='kruskal', weight='distance')

    total_cost = sum([mst.edges[i, j]['distance'] for i, j in mst.edges()])
    return total_cost


def compute_mst_with_fiber_costs(k: int, antennas: ac.EarthLocation, obstime: at.Time, array_location: ac.EarthLocation,
                                 target_node: int = 0,
                                 plot: bool = False,
                                 save_file: str | None = None):
    """
    Compute the minimal spanning tree of the array, calculate edge and node flow counts,
    and optionally plot the MST colored by flow counts.

    Each node (except the target) sends a signal to the target node, and the flow
    along each edge is the cumulative count of signals passing through that edge.
    Each node's flow count is given by the number of signals that pass through it
    (its subtree size when the MST is rooted at the target).

    Args:
        antennas: the antennas
        obstime: the observation time
        array_location: the location of the array
        target_node: the target node to which signals are sent (default: 0)
        plot: whether to plot the minimal spanning tree, angles, and flow counts
        save_file: the file prefix to save the plots to

    Returns:
        total_distance: the total distance of the minimal spanning tree
        node_angles: the maximal angles between edges that connect to each node
        connections: the number of connections for each node
        edge_flow: a numpy array of shape [E] with the flow count on each edge in the MST
    """

    earth_radius = np.linalg.norm(array_location.get_itrs().cartesian.xyz.to(au.m).value)
    antennas_enu_xyz = antennas.get_itrs(obstime=obstime, location=array_location).transform_to(
        ENU(obstime=obstime, location=array_location)
    ).cartesian.xyz.T.to('m').value

    G = nx.Graph()

    # Compute the cost for k nearest neighbors, k large enough to encompass MST
    tree_kd = KDTree(antennas_enu_xyz)
    nn_dists, nn_idxs = tree_kd.query(antennas_enu_xyz, k=k + 1)
    # nn_dists = nn_dists[:, 1:]

    antenna_lon = antennas.geodetic.lon.to('rad').value
    antenna_lat = antennas.geodetic.lat.to('rad').value
    nn_dists_haversine = earth_radius * haversine(
        antenna_lon[:, None],
        antenna_lat[:, None],
        antenna_lon[nn_idxs],
        antenna_lat[nn_idxs]
    )  # [N, k]

    for i in range(antennas_enu_xyz.shape[0]):
        for dist, j in zip(nn_dists_haversine[i], nn_idxs[i]):
            G.add_edge(i, int(j), distance=float(dist))

    mst = compute_mst_costed(G, target_node=target_node)

    def compute_max_angle(n):
        angles = []
        for i in mst.neighbors(n):
            for j in mst.neighbors(n):
                if i != j:
                    a = antennas_enu_xyz[i] - antennas_enu_xyz[n]
                    b = antennas_enu_xyz[j] - antennas_enu_xyz[n]
                    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                    angles.append(np.pi - np.arccos(cos_angle))
        if len(angles) > 0:
            return np.max(angles) * 180 / np.pi
        else:
            return 0

    for n in mst.nodes():
        mst.nodes[n]['max_angle'] = compute_max_angle(n)
        mst.nodes[n]['connections'] = mst.degree[n]

    total_cost = sum([mst.edges[i, j]['cost'] for i, j in mst.edges()])
    total_distance = sum([10 ** mst.edges[i, j]['log10_distance'] for i, j in mst.edges()])
    node_angles = np.array([mst.nodes[i]['max_angle'] for i in mst.nodes()])
    connections = np.array([mst.nodes[i]['connections'] for i in mst.nodes()])
    edge_cost = np.array([mst[i][j]['cost'] for i, j in mst.edges()])

    # --- Plotting ---
    if plot:
        plot_nodes(
            mst=mst,
            antennas_enu_xyz=antennas_enu_xyz,
            attr_name='connections',
            title=f'MST: Total Cost {total_cost:.2f} $ Total Distance: {total_distance:.2f} m',
            colorbar_name='# Connections',
            save_file=save_file + '_mst_connections.png' if save_file is not None else None
        )
        plot_nodes(
            mst=mst,
            antennas_enu_xyz=antennas_enu_xyz,
            attr_name='max_angle',
            title=f'MST: Total Cost {total_cost:.2f} $ Total Distance: {total_distance:.2f} m',
            colorbar_name='Maximal Angle (deg)',
            save_file=save_file + '_mst_angles.png' if save_file is not None else None
        )
        plot_edges(
            mst=mst,
            antennas_enu_xyz=antennas_enu_xyz,
            attr_name='log10_flow',
            title=f'MST: Total Cost {total_cost:.2f} $ Total Distance: {total_distance:.2f} m',
            colorbar_name='log10(Flow)',
            save_file=save_file + '_mst_edges_flow.png' if save_file is not None else None
        )
        plot_edges(
            mst=mst,
            antennas_enu_xyz=antennas_enu_xyz,
            attr_name='log10_distance',
            title=f'MST: Total Cost {total_cost:.2f} $ Total Distance: {total_distance:.2f} m',
            colorbar_name='log10(Distance (m))',
            save_file=save_file + '_mst_edges_distance.png' if save_file is not None else None
        )
        plot_edges(
            mst=mst,
            antennas_enu_xyz=antennas_enu_xyz,
            attr_name='cost',
            title=f'MST: Total Cost {total_cost:.2f} $ Total Distance: {total_distance:.2f} m',
            colorbar_name='Cost ($)',
            save_file=save_file + '_mst_edges_cost.png' if save_file is not None else None
        )

    return total_cost, node_angles, connections, edge_cost


def plot_nodes(mst, antennas_enu_xyz, attr_name, title, colorbar_name, save_file):
    colors = [mst.nodes[i][attr_name] for i in mst.nodes()]
    # Plot 2: MST colored by maximal angle (existing plot)
    fig = plt.figure(figsize=(10, 10))
    sc = plt.scatter(antennas_enu_xyz[:, 0], antennas_enu_xyz[:, 1], c=colors,
                     cmap='jet')
    plt.colorbar(sc, label=colorbar_name)
    for i, j in mst.edges():
        plt.plot([antennas_enu_xyz[i, 0], antennas_enu_xyz[j, 0]],
                 [antennas_enu_xyz[i, 1], antennas_enu_xyz[j, 1]], 'k-')
    plt.title(title)
    plt.xlabel('East (km)')
    plt.ylabel('North (km)')
    if save_file is not None:
        plt.savefig(save_file)
        plt.close(fig)
    else:
        plt.show()


def plot_edges(mst, antennas_enu_xyz, attr_name, title, colorbar_name, save_file):
    colors = [mst[i][j][attr_name] for i, j in mst.edges()]
    # Plot G with cost on edges
    fig, ax = plt.subplots(figsize=(10, 10))
    # Color nodes
    ax.scatter(antennas_enu_xyz[:, 0], antennas_enu_xyz[:, 1], c='black', s=1)
    # Plot edges with color based on attr.
    normaliser = plt.Normalize(vmin=min(colors), vmax=max(colors))
    for i, j in mst.edges():
        cost = mst[i][j][attr_name]
        color = plt.cm.jet(normaliser(cost))
        ax.plot([antennas_enu_xyz[i, 0], antennas_enu_xyz[j, 0]],
                [antennas_enu_xyz[i, 1], antennas_enu_xyz[j, 1]], color=color,
                lw=2)

    mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    mappable.set_array([normaliser.vmin, normaliser.vmax])
    plt.colorbar(mappable, label=colorbar_name, ax=ax)

    plt.title(title)
    plt.xlabel('East (km)')
    plt.ylabel('North (km)')
    if save_file is not None:
        plt.savefig(save_file)
        plt.close(fig)
    else:
        plt.show()
