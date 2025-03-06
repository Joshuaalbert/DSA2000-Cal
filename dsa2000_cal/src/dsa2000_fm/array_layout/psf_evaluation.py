import itertools
import os.path
from typing import Generator, Tuple, List

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
from scipy.special import j0
from tomographic_kernel.frames import ENU
from tqdm import tqdm

from dsa2000_assets.array_constraints.array_constraint_content import ArrayConstraintsV3
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.astropy_utils import mean_itrs
from dsa2000_common.common.mixed_precision_utils import mp_policy
from dsa2000_common.common.quantity_utils import quantity_to_np
from dsa2000_common.common.serialise_utils import SerialisableBaseModel
from dsa2000_fm.abc import AbstractArrayConstraint
from dsa2000_fm.array_layout.geo_constraints import RegionSampler, haversine

tfpd = tfp.distributions


# Compose PSF
def rotation_matrix_change_dec(delta_dec: FloatArray):
    """
    Get rotation matrix for changing DEC by delta_dec.

    Args:
        delta_dec: the change in DEC in radians

    Returns:
        R: the rotation matrix
    """
    # Rotate up or down changing DEC, but keeping RA constant.
    # Used for projecting ENU system
    c, s = jnp.cos(delta_dec), jnp.sin(delta_dec)
    R = jnp.asarray(
        [
            [1., 0., 0.],
            [0., c, -s],
            [0., s, c]
        ]
    )
    return R


def rotate_coords(antennas: FloatArray, dec_from: FloatArray, dec_to: FloatArray) -> FloatArray:
    """
    Rotate the antennas from one DEC to another DEC.

    Args:
        antennas: [..., 3]
        dec_from: the DEC to rotate from
        dec_to: the DEC to rotate to

    Returns:
        [..., 3] the rotated antennas
    """
    # East to east
    delta_dec = dec_to - dec_from
    east, north, up = antennas[..., 0], antennas[..., 1], antennas[..., 2]
    east_prime = east
    north_prime = jnp.cos(delta_dec) * north - jnp.sin(delta_dec) * up
    up_prime = jnp.sin(delta_dec) * north + jnp.cos(delta_dec) * up
    return jnp.stack([east_prime, north_prime, up_prime], axis=-1)


def deproject_antennas(antennas_projected: FloatArray, latitude: FloatArray, transit_dec: FloatArray) -> FloatArray:
    """
    Deproject the antennas from the projected coordinates.

    Args:
        antennas_projected: [..., 3]
        latitude: the latitude of the array
        transit_dec: the transit DEC of the array

    Returns:
        [..., 3] the deprojected antennas
    """
    antennas = rotate_coords(antennas_projected, transit_dec, latitude)
    # antennas = antennas.at[..., 2].set(0.)
    return antennas


def project_antennas(antennas: FloatArray, latitude: FloatArray, transit_dec: FloatArray) -> FloatArray:
    """
    Project the antennas to the projected coordinates.

    Args:
        antennas: [..., 3] the antennas in ENU at the latitude
        latitude: the latitude of the array
        transit_dec: the transit DEC of the array

    Returns:
        [..., 3] the projected antennas
    """
    antennas_projected = rotate_coords(antennas, latitude, transit_dec)
    # antennas_projected = antennas_projected.at[..., 2].set(0.)
    return antennas_projected


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


def compute_mst(k: int, antennas: ac.EarthLocation, obstime: at.Time, array_location: ac.EarthLocation,
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
    nn_idxs = nn_idxs[:, 1:]
    nn_dists = nn_dists[:, 1:]

    for i in range(antennas_enu_xyz.shape[0]):
        for dist, j in zip(nn_dists[i], nn_idxs[i]):
            # compute haversine distance
            dist = earth_radius * haversine(
                antennas[i].geodetic.lon.to('rad').value,
                antennas[i].geodetic.lat.to('rad').value,
                antennas[j].geodetic.lon.to('rad').value,
                antennas[j].geodetic.lat.to('rad').value
            )
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
    total_distance = sum([10**mst.edges[i, j]['log10_distance'] for i, j in mst.edges()])
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


def test_compute_mst():
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000_optimal_v1'))
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

    _ = compute_mst(20, antennas, obstime, array_location,
                    target_node=0,
                    plot=True,
                    save_file=None)


def compute_psf(antennas: FloatArray, lmn: FloatArray, freq: FloatArray, latitude: FloatArray,
                transit_dec: FloatArray, with_autocorr: bool = True) -> FloatArray:
    """
    Compute the point spread function of the array. Uses short cut,

    B(l,m) = (sum_i e^(-i2pi (u_i l + v_i m)))^2/N^2

    To remove auto-correlations, there are N values of 1 to subtract from N^2 values, then divide by (N-1)N
    PSF(l,m) = (N^2 B(l,m) - N)/(N-1)/N = (N B(l,m) - 1)/(N-1) where B(l,m) in [0, 1].
    Thus the amount of negative is (-1/(N-1))

    Args:
        antennas: [N, 3] the antennas in ENU
        lmn: [..., 3] the lmn coordinates to evaluate the PSF
        freq: [] the frequency in Hz

    Returns:
        psf: [...] the point spread function
    """

    def compute_shard_psf(antennas, lmn_shard, freq, latitude, transit_dec):
        antennas = project_antennas(antennas, latitude, transit_dec)
        wavelength = mp_policy.cast_to_length(299792458. / freq)
        r = antennas / wavelength
        delay = -2 * jnp.pi * jnp.sum(r * lmn_shard[..., None, :], axis=-1)  # [..., N]
        N = antennas.shape[-2]
        voltage_beam = jax.lax.complex(jnp.cos(delay), jnp.sin(delay))  # [..., N]
        voltage_beam = jnp.mean(voltage_beam, axis=-1)  # [...]
        power_beam = jnp.abs(voltage_beam) ** 2
        if with_autocorr:
            return power_beam
        return jnp.reciprocal(N - 1) * (N * power_beam - 1)

    _, psf = jax.lax.scan(
        lambda carry, lmn: (None, compute_shard_psf(antennas, lmn, freq, latitude, transit_dec)),
        None,
        lmn
    )

    return psf


def compute_ideal_psf_distribution(key, lmn: FloatArray, freq: FloatArray, latitude: FloatArray,
                                   transit_dec: FloatArray, base_projected_array: FloatArray, num_samples: int,
                                   num_antennas: int | None = None):
    def body_fn(carry, key):
        x, x2 = carry
        if num_antennas is not None:
            key, sub_key = jax.random.split(key)
            replace_idxs = jax.random.choice(sub_key, num_antennas, (num_antennas,), replace=False)
            ants = base_projected_array[replace_idxs]
        else:
            ants = base_projected_array

        psf = _sample_ideal_psf(
            key,
            lmn,
            freq,
            latitude,
            transit_dec,
            ants,
            with_autocorr=True
        )
        log_psf = 10 * jnp.log10(psf)
        x = x + log_psf
        x2 = x2 + log_psf ** 2
        return (x, x2), None

    init_x = jnp.zeros(lmn.shape[:-1])
    (x, x2), _ = jax.lax.scan(
        body_fn,
        (init_x, init_x),
        jax.random.split(key, num_samples)
    )
    mean = x / num_samples
    std = jnp.sqrt(jnp.abs(x2 / num_samples - mean ** 2))
    return mean, std


def _sample_ideal_psf(key, lmn: FloatArray, freq: FloatArray, latitude: FloatArray,
                      transit_dec: FloatArray, base_projected_array: FloatArray, with_autocorr: bool) -> FloatArray:
    """
    Compute the ideal point spread function of the array

    Args:
        lmn: [Nr, Ntheta, 3]
        freq: []
        latitude: []
        transit_dec: []

    Returns:
        psf: [Nr, Ntheta]
    """
    antenna_projected_dist = tfpd.Normal(loc=0, scale=200.)
    delta = antenna_projected_dist.sample(base_projected_array.shape, key).at[:, 2].set(0.)
    antennas_enu = base_projected_array + delta
    psf = compute_psf(antennas_enu, lmn, freq, latitude, transit_dec, with_autocorr=with_autocorr)
    return psf


def compute_ideal_psf(s, R, sigma, Rmin=0):
    """
    Compute the ideal point spread function for a given s, R, and
    sigma. The ideal point spread function is the Fourier transform of the
    Gaussian beam pattern.

    Args:
        s: the spatial frequency
        R: the radius of the Gaussian beam pattern in units of wavelength
        sigma: the standard deviation of the Gaussian beam pattern in units of wavelength

    Returns:
        F(s): the ideal point spread function at spatial frequency s
    """
    # Define the integrand for the radial Fourier transform.
    integrand = lambda r: r * np.exp(-r ** 2 / (2 * sigma ** 2)) * j0(2 * np.pi * s * r)
    # Compute the radial integral from 0 to R.
    result, _ = quad(integrand, Rmin, R)
    return 2 * np.pi * result


def compute_fwhm(R, sigma, Rmin=0):
    """
    Determine the ideal beam size s that corresponds to the half-maximum of the
    ideal point spread function.

    Args:
        R: the radius of the Gaussian beam pattern in units of wavelength
        sigma: the standard deviation of the Gaussian beam pattern in units

    Returns:
        s: the spatial frequency corresponding to the half-maximum of the ideal point spread function
    """

    F0 = compute_ideal_psf(0., R, sigma, Rmin=Rmin)

    def half_max_eq(s, R, sigma, Rmin):
        # F(0) is the peak:
        # F0 = 2 * np.pi * sigma ** 2 * (1 - np.exp(-R ** 2 / (2 * sigma ** 2)))
        # We want F(s) - F(0)/2 = 0.
        return compute_ideal_psf(s, R, sigma, Rmin=Rmin) - F0 / 2

    # Use a root-finding algorithm to solve half_max_eq(s) = 0.
    # We choose a bracket [0, s_max] where the root is expected.
    s_max = np.pi / 2
    sol = root_scalar(half_max_eq, args=(R, sigma, Rmin), bracket=[0, s_max], method='brentq')
    return sol.root * 2


def find_sigma(R, s_fwhm, Rmin=0):
    def eq(sigma, R, s_fwhm, Rmin):
        return compute_fwhm(R, sigma, Rmin=Rmin) - s_fwhm

    sol = root_scalar(eq, args=(R, s_fwhm, Rmin), bracket=[R / 100, R * 100], method='brentq')
    return sol.root


def sample_projected_antennas(key, R, sigma, num_antennas):
    # Create a truncated 2D normal distribution with mean 0 and standard deviation sigma, truncated at -R and R.
    dist = tfpd.TruncatedNormal(loc=0., scale=sigma / np.sqrt(2.), low=0, high=R)
    # Sample from the distribution.
    key1, key2 = jax.random.split(key)
    radius = dist.sample((num_antennas,), key1)
    theta = jax.random.uniform(key2, (num_antennas,), minval=0., maxval=2 * np.pi)
    projected_positions = jnp.stack([
        radius * jnp.cos(theta),
        radius * jnp.sin(theta),
        jnp.zeros_like(radius)
    ], axis=-1)
    return projected_positions


def pareto_front(points):
    """
    Computes the Pareto front (non-dominated set) of log(L) of PSF and distance of minimal spanning tree.
    The front is computed for larger values of log(L) of PSF and smaller values of distance of minimal spanning tree.

    Parameters:
        points: [N, 2] the points to compute the Pareto front.
            Column 0: distance of minimal spanning tree
            Column 1: log(L) of PSF

    Returns:
        pareto_points (ndarray): The set of points on the Pareto front.
    """

    hull = ConvexHull(points, incremental=False)
    # vertices = hull.vertices  # [nvertex] representing vertices
    simplices = hull.simplices  # [nfacet, 2] representing facets in 2D
    eqs = hull.equations  # [nfacet, 3] representing n planes in 3D  (normal, offset)
    normals = eqs[:, :2]  # [nfacet, 2]
    # offsets = eqs[:, 2]  # [nfacet]
    # Keep simplex if normal points towards increasing log(L) of PSF and decreasing distance of minimal spanning tree
    keep_mask = (normals[:, 1] > 0) & (normals[:, 0] < 0)
    return np.unique(simplices[keep_mask].flatten())


def get_pareto_eqs(hull: ConvexHull):
    simplices = hull.simplices  # [nfacet, 2] representing facets in 2D
    eqs = hull.equations  # [nfacet, 3] representing n planes in 3D  (normal, offset)
    normals = eqs[:, :2]  # [nfacet, 2]
    offsets = eqs[:, 2]  # [nfacet]
    # Keep simplex if normal points towards increasing log(L) of PSF and decreasing distance of minimal spanning tree
    keep_mask = (normals[:, 0] < 0) & (normals[:, 1] > 0)
    if not np.any(keep_mask):
        keep_mask = normals[:, 1] > 0
    # normals = normals[keep_mask]
    # offsets = offsets[keep_mask]
    simplices_lengths = np.linalg.norm(hull.points[simplices[:, 0]] - hull.points[simplices[:, 1]], axis=1)
    simplices = simplices[keep_mask]
    simplices_lengths = simplices_lengths[keep_mask]
    vertex_idxs = np.unique(simplices.flatten())
    return normals, offsets, simplices, simplices_lengths, vertex_idxs


# Example usage


def is_violation(
        check_idx: int, antennas: ac.EarthLocation,
        array_location: ac.EarthLocation, obstime: at.Time,
        additional_buffer: float, minimal_antenna_sep: float,
        aoi_data: List[Tuple[RegionSampler, float]],
        constraint_data: List[Tuple[RegionSampler, float]],
        verbose: bool = False
):
    """
    Check if the proposed antenna location violates any constraints.

    Args:
        check_idx: the index of the antenna to check
        antennas: the antennas
        array_location: the location of the array
        obstime: the observation time
        additional_buffer: an additional buffer from boundaries, in meters, on top of data provided.
        minimal_antenna_sep: the minimal separation between antennas, in meters.
        aoi_data: list of tuples of samplers and buffers for the area of interest
        constraint_data: list of tuples of samplers and buffers for the constraints

    Returns:
        bool: True if the proposed antenna location violates any constraints, False otherwise
    """
    sample_proposal = [antennas[check_idx].geodetic.lon.to('deg').value,
                       antennas[check_idx].geodetic.lat.to('deg').value]
    earth_radius = np.linalg.norm(array_location.get_itrs().cartesian.xyz.to(au.m).value)

    aoi_samplers, aoi_buffers = zip(*aoi_data)
    constraint_samplers, constraint_buffers = zip(*constraint_data)

    antennas_enu = antennas.get_itrs(
        obstime=obstime, location=array_location
    ).transform_to(
        ENU(obstime=obstime, location=array_location)
    ).cartesian.xyz.to('m').value.T  # [N, 3]

    buffer_satisfy = []
    for aoi_sampler, buffer in zip(aoi_samplers, aoi_buffers):
        if aoi_sampler.closest_approach(*sample_proposal)[1] == 0:
            # it's inside so we care about it.
            # Check that it far enough from all AOI perimeters
            _, angular_dist = aoi_sampler.closest_approach_to_boundary(*sample_proposal)
            dist = np.pi / 180. * angular_dist * earth_radius
            if dist >= buffer + additional_buffer:
                buffer_satisfy.append(True)
    if len(buffer_satisfy) == 0:
        # not in any AOI
        if verbose:
            print(f"Antenna {check_idx} not in any AOI")
        return True
    # Check all buffer constraints satisfied (including overlaps). Should merge first.
    if not all(buffer_satisfy):
        if verbose:
            print(f"Antenna {check_idx} violates AOI buffer constraints")
        return True

    # Check that it is far enough from all constraint regions including buffer
    for constraint_sampler, buffer in zip(constraint_samplers, constraint_buffers):
        _, angular_dist = constraint_sampler.closest_approach(*sample_proposal)
        dist = np.pi / 180. * angular_dist * earth_radius
        if dist <= buffer + additional_buffer:
            if verbose:
                print(f"Antenna {check_idx} violates constraint buffer constraints")
            return True

    # Check that it is far enough from other antennas, excluding the one being replaced
    sample_enu = ac.EarthLocation.from_geodetic(
        lon=sample_proposal[0] * au.deg,
        lat=sample_proposal[1] * au.deg,
        height=array_location.geodetic.height
    ).get_itrs(
        obstime=obstime, location=array_location
    ).transform_to(
        ENU(obstime=obstime, location=array_location)
    ).cartesian.xyz.to('m').value  # [3]
    dists = np.linalg.norm(antennas_enu - sample_enu, axis=-1)  # [N]
    dists[check_idx] = np.inf
    if np.min(dists) < minimal_antenna_sep:
        if verbose:
            print(f"Antenna {check_idx} violates minimal antenna separation")
        return True

    return False


def sample_aoi(
        replace_idx: int, antennas: ac.EarthLocation, array_location: ac.EarthLocation, obstime: at.Time,
        additional_buffer: float, minimal_antenna_sep: float,
        aoi_data: List[Tuple[RegionSampler, float]],
        constraint_data: List[Tuple[RegionSampler, float]]
) -> ac.EarthLocation:
    """
    Sample a new antenna location within the area of interest.

    Args:
        replace_idx: the index of the antenna to replace
        antennas: the antennas
        array_location: the location of the array
        obstime: the observation time
        additional_buffer: an additional buffer from boundaries, in meters, on top of data provided.
        minimal_antenna_sep: the minimal separation between antennas, in meters.
        aoi_data: list of tuples of samplers and buffers for the area of interest
        constraint_data: list of tuples of samplers and buffers for the constraints

    Returns:
        antennas: the antennas with the replaced antenna, a copy
    """
    aoi_samplers, aoi_buffers = zip(*aoi_data)
    areas = np.asarray([s.total_area for s in aoi_samplers])
    aoi_probs = areas / areas.sum()

    # modify a copy
    antennas = antennas.copy()

    while True:
        # Choose a AOI proportional to the area of the AOI
        sampler_idx = np.random.choice(len(aoi_samplers), p=aoi_probs)
        sampler = aoi_samplers[sampler_idx]
        # Get a sample within the AOI
        sample_proposal = sampler.get_samples_within(1)[0]  # lon, lat

        # Count how many AOIs contain the sample
        count_contain = sum(
            [(1 if sampler.closest_approach(*sample_proposal)[1] == 0 else 0) for sampler in aoi_samplers],
            start=0
        )

        if np.random.uniform() > 1. / count_contain:
            # Takes into AOI overlap
            continue

        antennas[replace_idx] = ac.EarthLocation.from_geodetic(
            lon=sample_proposal[0] * au.deg,
            lat=sample_proposal[1] * au.deg,
            height=array_location.geodetic.height
        )

        if is_violation(
                replace_idx, antennas, array_location, obstime, additional_buffer, minimal_antenna_sep,
                aoi_data, constraint_data
        ):
            continue

        return antennas


class SamplePoint(SerialisableBaseModel):
    antennas: ac.EarthLocation
    latitude: au.Quantity


class SampleEvaluation(SerialisableBaseModel):
    quality: float
    cost: float
    antennas: ac.EarthLocation


class Results(SerialisableBaseModel):
    array_location: ac.EarthLocation
    obstime: at.Time
    additional_buffer: au.Quantity
    minimal_antenna_sep: au.Quantity
    evaluations: List[SampleEvaluation]


def point_generator(
        plot_dir: str,
        results_file: str,
        array_constraint: AbstractArrayConstraint,
        antennas: ac.EarthLocation | None,
        array_location: ac.EarthLocation | None,
        obstime: at.Time | None,
        additional_buffer: au.Quantity,
        minimal_antenna_sep: au.Quantity
) -> Generator[SamplePoint, SampleEvaluation, None]:
    os.makedirs(plot_dir, exist_ok=True)
    aoi_data = array_constraint.get_area_of_interest_regions()
    # merge AOI's
    merged_aoi_sampler = RegionSampler.merge([s for s, _ in aoi_data])
    merged_buffer = max([b for _, b in aoi_data])
    aoi_data = [(merged_aoi_sampler, merged_buffer)]
    constraint_data = array_constraint.get_constraint_regions()

    additional_buffer_m = float(quantity_to_np(additional_buffer, 'm'))
    minimal_antenna_sep_m = float(quantity_to_np(minimal_antenna_sep, 'm'))

    if os.path.exists(results_file):
        results = Results.parse_file(results_file)
    else:
        results = Results(
            array_location=array_location,
            obstime=obstime,
            additional_buffer=additional_buffer,
            minimal_antenna_sep=minimal_antenna_sep,
            evaluations=[]
        )
        # we'll populate initial evaluations
        if antennas is None:
            raise ValueError("Initial antennas must be provided if results file does not exist.")
        if array_location is None:
            raise ValueError("Array location must be provided if results file does not exist.")
        if obstime is None:
            raise ValueError("Observation time must be provided if results file does not exist.")
        latitude = quantity_to_np(results.array_location.geodetic.lat, 'rad')
        # Make sure initial antennas satisfy constraints
        for check_idx in range(len(antennas)):
            if is_violation(
                    check_idx, antennas, array_location, obstime, additional_buffer_m,
                    minimal_antenna_sep_m, aoi_data, constraint_data
            ):
                print(f"Initial antenna {check_idx} violates constraints. Replacing")
                antennas = sample_aoi(
                    check_idx, antennas, array_location, obstime, additional_buffer_m,
                    minimal_antenna_sep_m, aoi_data, constraint_data
                )
        # Add initial points to create a hull
        evaluation = yield SamplePoint(antennas=antennas, latitude=results.array_location.geodetic.lat)
        results.evaluations.append(evaluation)
        for _ in range(2):
            # Choose a random antenna to replace
            replace_idx = np.random.choice(len(antennas))
            antennas = sample_aoi(
                replace_idx=replace_idx,
                antennas=antennas,
                array_location=results.array_location,
                obstime=results.obstime,
                additional_buffer=additional_buffer_m,
                minimal_antenna_sep=minimal_antenna_sep_m,
                aoi_data=aoi_data,
                constraint_data=constraint_data
            )
            evaluation = yield SamplePoint(antennas=antennas, latitude=results.array_location.geodetic.lat)
            results.evaluations.append(evaluation)
        # hull = ConvexHull(points=np.asarray([[e.cost, e.quality] for e in results.evaluations]), incremental=True)
        # _, _, _, _, vertex_idxs = get_pareto_eqs(hull)
        # while len(vertex_idxs) == 0:
        #  #   Choose a random antenna to replace
        # replace_idx = np.random.choice(len(antennas))
        # antennas = sample_aoi(
        #     replace_idx=replace_idx,
        #     antennas=antennas,
        #     array_location=results.array_location,
        #     obstime=results.obstime,
        #     additional_buffer=additional_buffer_m,
        #     minimal_antenna_sep=minimal_antenna_sep_m,
        #     aoi_data=aoi_data,
        #     constraint_data=constraint_data
        # )
        # evaluation = yield SamplePoint(antennas=antennas, latitude=results.array_location.geodetic.lat)
        # results.evaluations.append(evaluation)
        # hull.add_points(np.asarray([evaluation.cost, evaluation.quality])[None, :], restart=False)
        # _, _, _, _, vertex_idxs = get_pareto_eqs(hull)
        # del hull

    hull = ConvexHull(points=np.asarray([[e.cost, e.quality] for e in results.evaluations]), incremental=True)
    pbar = tqdm(itertools.count())
    while True:
        normals, offsets, simplices, simplices_lengths, vertex_idxs = get_pareto_eqs(hull)
        # Choose a frontier point at random, as a seed point for iteration
        simplicies_probs = simplices_lengths / np.sum(simplices_lengths)
        simplex_idx = np.random.choice(len(simplices_lengths), p=simplicies_probs)
        vertex_idx = np.random.choice(simplices[simplex_idx, :])
        vertex_antennas = results.evaluations[vertex_idx].antennas
        # Choose a random antenna to replace
        replace_idx = np.random.choice(len(vertex_antennas))
        proposal_antennas = sample_aoi(
            replace_idx=replace_idx,
            antennas=vertex_antennas,
            array_location=results.array_location,
            obstime=results.obstime,
            additional_buffer=additional_buffer_m,
            minimal_antenna_sep=minimal_antenna_sep_m,
            aoi_data=aoi_data,
            constraint_data=constraint_data
        )
        evaluation = yield SamplePoint(antennas=proposal_antennas, latitude=results.array_location.geodetic.lat)
        point = np.asarray([evaluation.cost, evaluation.quality])
        # Check if the new point is on the Pareto front using eqs
        for normal, offset in zip(normals, offsets):
            if np.dot(normal, point) + offset > 0:
                # The new point is on the hull
                results.evaluations.append(evaluation)
                hull.add_points(point[None, :], restart=False)
                # protect from KeyboardInterrupt
                try:
                    pass
                except KeyboardInterrupt:
                    raise
                finally:
                    with open(results_file, 'w') as f:
                        f.write(results.json(indent=2))
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    sc = ax.scatter(hull.points[:, 0], hull.points[:, 1], c=range(len(hull.points)), cmap='jet')
                    plt.colorbar(sc, ax=ax, label='Iteration')
                    ax.scatter(hull.points[vertex_idxs, 0], hull.points[vertex_idxs, 1], c='black', marker='*',
                               label='Previous Pareto front')
                    ax.scatter(point[0], point[1], c='blue', marker='x',
                               label='New point')
                    ax.set_xlabel('Cost')
                    ax.set_ylabel('Quality')
                    ax.legend()
                    fig.savefig(os.path.join(plot_dir, f"pareto_front.png"))
                    plt.close(fig)
                pbar.update(1)
                break


def evaluate_psf(antennas_enu, lmn, latitude, freqs, decs, target_log_psf_mean, target_log_psf_stddev):
    """
    Evaluate the PSF of the array and compare to the target PSF.

    Args:
        antennas_enu: [N, 3] the antennas in projected
        lmn: [..., 3]
        latitude: the latitude of the array
        freqs: [M] the frequencies in Hz
        decs: [M] the declinations in radians
        target_log_psf_mean: [...] the target log PSF mean
        target_log_psf_stddev: [...] the target log PSF standard deviation

    Returns:
        quality: the negative chi squared value
    """
    psf = jax.vmap(
        lambda freq, dec: compute_psf(antennas_enu, lmn, freq, latitude, dec, with_autocorr=True)
    )(freqs, decs)

    log_psf = 10 * jnp.log10(psf)

    residuals = (log_psf - target_log_psf_mean) / target_log_psf_stddev

    chi2 = jnp.mean(residuals ** 2)
    return -chi2


def plot_solution(plot_folder: str, solution_file: str, aoi_data: List[Tuple[RegionSampler, float]],
                  constraint_data: List[Tuple[RegionSampler, float]]):
    # Plot solutions
    # solution_file = "solution.txt"
    if not os.path.exists(solution_file):
        raise FileNotFoundError(f"Solution file {solution_file} not found")

    with open(solution_file, 'r') as f:
        coords = []
        for line in f:
            if line.startswith("#"):
                continue
            x, y, z = line.strip().split(',')
            coords.append((float(x), float(y), float(z)))
    coords = np.asarray(coords)
    antennas = ac.EarthLocation.from_geocentric(
        coords[:, 0] * au.m,
        coords[:, 1] * au.m,
        coords[:, 2] * au.m
    )

    obstime = at.Time('2021-01-01T00:00:00', format='isot', scale='utc')
    array_location = mean_itrs(antennas.get_itrs()).earth_location

    # antennas_enu = antennas.get_itrs(
    #     obstime=obstime, location=array_location
    # ).transform_to(
    #     ENU(obstime=obstime, location=array_location)
    # ).cartesian.xyz.to('m').value.T

    # Plot along with regions
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # array_constraint = ArrayConstraintV2()
    # aoi_data = array_constraint.get_area_of_interest_regions()
    # constraint_data = array_constraint.get_constraint_regions()
    for sampler, buffer in aoi_data:
        # sampler.info()
        sampler.plot_region(ax=ax, color='blue')
    for sampler, buffer in constraint_data:
        sampler.plot_region(ax=ax, color='none')

    # ax.scatter(antennas_enu[:, 0], antennas_enu[:, 1], s=1, c='green', alpha=0.5, marker='.')
    ax.scatter(antennas.geodetic.lon.deg, antennas.geodetic.lat.deg, s=1, c='green', alpha=0.5, marker='.')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_title('Antenna layout')
    ax.set_xlim(-114.6, -114.3)
    ax.set_ylim(39.45, 39.70)
    fig.savefig(os.path.join(plot_folder, f'antenna_solution.png'))
    plt.show()

    # Plot violations
    for idx, point in enumerate(antennas):
        for sampler, buffer in constraint_data:
            (px, py), dist = sampler.closest_approach(point.geodetic.lon.deg, point.geodetic.lat.deg)
            earth_radius = np.linalg.norm(point.get_itrs().cartesian.xyz.to(au.m).value)
            dist = np.pi / 180 * dist * earth_radius

            if dist < buffer:
                print('Agree')
                sampler.info()
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                sampler.plot_region(ax=ax, color='none')
                ax.scatter(px, py, c='g')
                ax.scatter(point.geodetic.lon.deg, point.geodetic.lat.deg, c='b')
                bbox = min(point.geodetic.lon.deg, px), max(point.geodetic.lon.deg, px), min(point.geodetic.lat.deg,
                                                                                             py), max(
                    point.geodetic.lat.deg, py)
                ax.set_xlim(bbox[0] - 0.005, bbox[1] + 0.005)
                ax.set_ylim(bbox[2] - 0.005, bbox[3] + 0.005)
                ax.set_title(f"{dist} {buffer}")
                plt.show()


def test_point_generator():
    antennas = ac.EarthLocation.from_geocentric(
        [0, 0, 0, 1, 2, 3] * au.km,
        [0, 1, 0, 2, 3, 4] * au.km,
        [0, 0, 1, 3, 4, 5] * au.km
    )
    array_location = ac.EarthLocation.of_site('vla')
    obstime = at.Time("2021-01-01T00:00:00", scale='utc')
    gen = point_generator(
        results_file='test_results.json',
        plot_dir='test_plots',
        array_constraint=ArrayConstraintsV3(),
        antennas=antennas,
        array_location=array_location,
        obstime=obstime,
        additional_buffer=0 * au.m,
        minimal_antenna_sep=8 * au.m
    )
    gen_response = None
    count = 0
    while True:
        try:
            sample_point = gen.send(gen_response)
        except StopIteration:
            break
        quality = np.random.normal()
        cost, _, _ = compute_mst(
            k=5,
            antennas=sample_point.antennas,
            obstime=obstime,
            array_location=array_location,
            plot=True,
            save_file='mst'
        )
        gen_response = SampleEvaluation(
            quality=quality,
            cost=cost,
            antennas=sample_point.antennas
        )
        count += 1

        if count > 50:
            break
