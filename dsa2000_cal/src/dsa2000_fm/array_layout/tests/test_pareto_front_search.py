import numpy as np
import pylab as plt
from astropy import coordinates as ac, units as au, time as at

from dsa2000_assets.array_constraints.spring_valley_31b.array_constraint import ArrayConstraintsV3
from dsa2000_fm.array_layout.fiber_cost_fn import compute_mst_cost
from dsa2000_fm.array_layout.pareto_front_search import build_search_point_generator, SampleEvaluation, pareto_front


def test_point_generator():
    antennas = ac.EarthLocation.from_geocentric(
        [0, 0, 0, 1, 2, 3] * au.km,
        [0, 1, 0, 2, 3, 4] * au.km,
        [0, 0, 1, 3, 4, 5] * au.km
    )
    array_location = ac.EarthLocation.of_site('vla')
    obstime = at.Time("2021-01-01T00:00:00", scale='utc')
    search_point_gen = build_search_point_generator(
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
            sample_point = search_point_gen.send(gen_response)
        except StopIteration:
            break
        quality = np.random.normal()
        cost, _, _ = compute_mst_cost(
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


def test_pareto_front():
    np.random.seed(42)
    points = np.random.normal(size=(1000, 2)) - 5  # Generate random (x, y) points in [-5, 5]

    pareto_points_idxs = pareto_front(points)
    # plot
    plt.scatter(points[:, 0], points[:, 1], label='Points')
    plt.scatter(points[pareto_points_idxs, 0], points[pareto_points_idxs, 1], color='red', label='Pareto points')
    plt.ylabel('Quality')
    plt.xlabel('Cost')
    plt.legend()
    plt.show()
