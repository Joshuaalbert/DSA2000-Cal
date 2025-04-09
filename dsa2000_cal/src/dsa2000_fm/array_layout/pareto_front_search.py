import itertools
import os
from typing import Generator, List

import numpy as np
import pylab as plt
from astropy import coordinates as ac, time as at, units as au
from scipy.spatial import ConvexHull
from tqdm import tqdm

from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.quantity_utils import quantity_to_np
from dsa2000_common.common.serialise_utils import SerialisableBaseModel
from dsa2000_fm.abc import AbstractArrayConstraint
from dsa2000_fm.array_layout.sample_constraints import RegionSampler, is_violation, sample_aoi


def _get_pareto_eqs(hull: ConvexHull):
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


class SamplePoint(SerialisableBaseModel):
    antennas: ac.EarthLocation
    latitude: au.Quantity


class SampleEvaluation(SerialisableBaseModel):
    quality: float
    cost: float
    antennas: ac.EarthLocation
    done: bool = False


class Results(SerialisableBaseModel):
    array_location: ac.EarthLocation
    obstime: at.Time
    additional_buffer: au.Quantity
    minimal_antenna_sep: au.Quantity
    evaluations: List[SampleEvaluation]

def build_search_point_generator(
        plot_dir: str,
        results_file: str,
        array_constraint: AbstractArrayConstraint,
        antennas: ac.EarthLocation | None,
        array_location: ac.EarthLocation | None,
        obstime: at.Time | None,
        additional_buffer: au.Quantity,
        minimal_antenna_sep: au.Quantity
) -> Generator[SamplePoint, SampleEvaluation, None]:
    """
    Generate points for the optimization process. The points are generated
    by sampling the area of interest and checking if they satisfy the
    constraints. If they do, they are added to the list of evaluations.
    The points are generated in a way that they are not too close to each
    other, and they are not too close to the boundaries of the area of interest.


    Args:
        plot_dir: the directory to save the plots
        results_file: the file to save the results
        array_constraint: the array constraint to use
        antennas: the antennas to use
        array_location: the location of the array
        obstime: the observation time
        additional_buffer: an additional buffer from boundaries, in meters, on top of data provided.
        minimal_antenna_sep: the minimal separation between antennas, in meters.

    Returns:
        a generator that yields SamplePoint objects, and accepts SampleEvaluation objects
    """
    os.makedirs(plot_dir, exist_ok=True)
    aoi_data = array_constraint.get_area_of_interest_regions()
    # merge AOI's
    merged_aoi_sampler = RegionSampler.merge([s for s, _ in aoi_data])
    merged_buffer = max([b for _, b in aoi_data])
    aoi_data = [(merged_aoi_sampler, merged_buffer)]
    constraint_data = array_constraint.get_constraint_regions()

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for sampler, buffer in aoi_data:
        sampler.plot_region(ax=ax, color='blue')
    for sampler, buffer in constraint_data:
        sampler.plot_region(ax=ax, color='red')
    ax.scatter(antennas.lon.deg, antennas.lat.deg, c='green', marker='x', label='Antennas')
    ax.scatter(array_location.lon.deg, array_location.lat.deg, c='black', marker='*', label='Array location')
    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_xlim(-114.6, -114.3)
    ax.set_ylim(39.45, 39.70)
    ax.legend()
    fig.savefig(os.path.join(plot_dir, "aoi_prior.png"))
    plt.close(fig)

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
                    minimal_antenna_sep_m, aoi_data, constraint_data, verbose=False
            ):
                dsa_logger.info(f"Initial antenna {check_idx} violates constraints. Replacing")
                antennas = sample_aoi(
                    check_idx, antennas, array_location, obstime, additional_buffer_m,
                    minimal_antenna_sep_m, aoi_data, constraint_data
                )

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for sampler, buffer in aoi_data:
            sampler.plot_region(ax=ax, color='blue')
        for sampler, buffer in constraint_data:
            sampler.plot_region(ax=ax, color='red')
        ax.scatter(antennas.lon.deg, antennas.lat.deg, c='green', marker='x', label='Antennas')
        ax.scatter(array_location.lon.deg, array_location.lat.deg, c='black', marker='*', label='Array location')
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_xlim(-114.6, -114.3)
        ax.set_ylim(39.45, 39.70)
        ax.legend()
        fig.savefig(os.path.join(plot_dir, "aoi_init.png"))
        plt.close(fig)
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
        normals, offsets, simplices, simplices_lengths, vertex_idxs = _get_pareto_eqs(hull)
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
        if evaluation.done:
            break


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
