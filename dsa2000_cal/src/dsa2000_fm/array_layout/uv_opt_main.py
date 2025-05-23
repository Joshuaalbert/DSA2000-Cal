import os

os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
# os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import pylab as plt
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from dsa2000_fm.array_layout.optimal_transport import compute_ideal_uv_distribution, evaluate_uv_distribution
from dsa2000_fm.array_layout.fiber_cost_fn import compute_mst_cost

from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.astropy_utils import mean_itrs
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.ray_utils import TimerLog
from dsa2000_fm.array_layout.pareto_front_search import SampleEvaluation, \
    build_quality_only_search_point_generator
from dsa2000_common.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_fm.abc import AbstractArrayConstraint
from dsa2000_assets.registries import array_registry
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.array_constraints.v6.array_constraint import ArrayConstraintsV6

tfpd = tfp.distributions


def main(
        uv_bins,
        uv_grid,
        target_dist,
        transit_dec,
        obstimes,
        ref_time,
        run_name: str,
        init_config: str | None,
        target_array_name: str,
        array_constraint: AbstractArrayConstraint,
        num_evaluations: int,
        num_antennas: int | None = None
):
    os.makedirs(run_name, exist_ok=True)
    plot_folder = os.path.join(run_name, 'plots')
    os.makedirs(plot_folder, exist_ok=True)

    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(target_array_name))
    array_location = array.get_array_location()
    phase_tracking = ENU(0, 0, 1, obstime=ref_time, location=array_location).transform_to(ac.ICRS())
    ra0 = quantity_to_jnp(phase_tracking.ra, 'rad')
    # dec0 = quantity_to_jnp(phase_tracking.dec, 'rad')

    with TimerLog("Loading initial configuration"):
        if init_config is not None:
            coords = []
            with open(init_config, 'r') as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    x, y, z = line.strip().split(',')
                    coords.append((float(x), float(y), float(z)))
            coords = np.asarray(coords)
            antennas0 = ac.EarthLocation.from_geocentric(
                coords[:, 0] * au.m,
                coords[:, 1] * au.m,
                coords[:, 2] * au.m
            )
            array_location = mean_itrs(antennas0.get_itrs()).earth_location
        else:

            antennas0 = array.get_antennas()
            antennas0_enu = antennas0.get_itrs(obstime=ref_time, location=array_location).transform_to(ENU(
                obstime=ref_time, location=array_location
            )).cartesian.xyz.T
            # elongate the array in north-south direction
            latitude = quantity_to_jnp(array_location.geodetic.lat, 'rad')
            antennas0_enu[:, 1] /= np.cos(latitude)
            antennas0 = ENU(antennas0_enu[:, 0], antennas0_enu[:, 1], antennas0_enu[:, 2],
                            obstime=ref_time, location=array_location).transform_to(ac.ITRS(obstime=ref_time,
                                                                                            location=array_location)
                                                                                    ).earth_location

            # antennas_proj = project_antennas(quantity_to_np(antennas0_enu), latitude, 0)
            # print('antennas0_enu[:, 2]',antennas0_enu[:, 2])
            # print('antennas_proj[:, 2]',antennas_proj[:, 2])
            # plt.scatter(antennas_proj[:, 0], antennas_proj[:, 1], s=1, c=antennas_proj[:, 2], marker='.')
            # plt.scatter(antennas0_enu[:, 0], antennas0_enu[:, 1], s=1, c=antennas0_enu[:, 2].value, marker='.')
            # plt.show()

    if num_antennas is not None:
        keep_idxs = np.random.choice(antennas0.shape[0], num_antennas, replace=False)
        antennas0 = antennas0[keep_idxs]

    antennas = antennas0.copy()
    dsa_logger.info(f"Number of antennas: {len(antennas)}")

    times_jax = time_to_jnp(obstimes, ref_time)

    dsa_logger.info(f"Run name: {run_name}")
    dsa_logger.info(f"Initial configuration: {init_config}")
    dsa_logger.info(f"Array constraint: {array_constraint}")
    dsa_logger.info(f"Number of antennas: {len(antennas)}")

    # Performing the optimization
    gen = build_quality_only_search_point_generator(
        results_file=os.path.join(run_name, 'results.json'),
        plot_dir=plot_folder,
        array_constraint=array_constraint,
        antennas=antennas,
        array_location=array_location,
        obstime=ref_time,
        additional_buffer=0 * au.m,
        minimal_antenna_sep=8 * au.m
    )

    gen_response = None
    count_evals = 0
    while count_evals < num_evaluations:
        count_evals += 1
        try:
            sample_point = gen.send(gen_response)
        except StopIteration:
            break

        proposal_antennas_gcrs = quantity_to_jnp(
            sample_point.antennas.get_gcrs(obstime=ref_time).cartesian.xyz.T,
            'm'
        )
        quality = evaluate_uv_distribution(
            proposal_antennas_gcrs,
            times_jax,
            ra0,
            transit_dec,
            uv_bins,
            uv_grid,
            target_dist
        )
        cost = compute_mst_cost(
            k=6,
            antennas=sample_point.antennas,
            obstime=ref_time,
            array_location=array_location
        )
        if np.isnan(cost):
            dsa_logger.warning(f"Cost is NaN for {sample_point}")
        if np.isnan(quality):
            dsa_logger.warning(f"Quality is NaN for {sample_point}")
        gen_response = SampleEvaluation(
            quality=quality,
            cost=cost,
            antennas=sample_point.antennas
        )
    # Store to final_config
    final_config = os.path.join(run_name, 'final_config.txt')
    with open(final_config, 'w') as f:
        for antenna in antennas:
            f.write(f"{antenna.x.to('m').value},{antenna.y.to('m').value},{antenna.z.to('m').value}\n")
    return final_config


if __name__ == '__main__':
    import warnings

    # Suppress specific UserWarnings
    # warnings.filterwarnings(
    #     "ignore",
    #     category=UserWarning,
    #     module="pyogrio"
    # )

    warnings.filterwarnings(
        "ignore",
        category=UserWarning
    )

    ref_time = at.Time("2025-06-10T00:00:00", format='isot', scale='utc')
    obstimes = ref_time[None]

    du = 100 * au.m
    R = 16000 * au.m
    target_fwhm = 3.14 * au.arcsec
    freq = 1350 * au.MHz
    uv_bins, uv_grid, target_dist = compute_ideal_uv_distribution(du, R, target_fwhm, freq)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(target_dist, extent=[uv_bins[0], uv_bins[-1], uv_bins[0], uv_bins[-1]], origin='lower',
                   interpolation='nearest', cmap='jet')
    ax.set_xlabel('u (m)')
    ax.set_ylabel('v (m)')
    ax.set_title('Target distribution')
    plt.colorbar(im, ax=ax, label='Density')
    plt.savefig(f'target_uv_distribution_{target_fwhm.to("arcsec").value}arcsec_psf.png')
    plt.close(fig)

    init_config = './pareto_opt_solution_a/dsa1650_9P_a_optimal_v1.2.txt'
    np.random.seed(0)
    while True:
        # From smallest to largest, so smaller one fits in next as good starting point
        for prefix in ['a']:
            array_constraint = ArrayConstraintsV6(prefix)
            run_name = f"pareto_opt_v6_{prefix}_v2"
            final_config = main(
                uv_bins=uv_bins, uv_grid=uv_grid, target_dist=target_dist,
                obstimes=obstimes,
                transit_dec=0.,
                ref_time=ref_time,
                target_array_name='dsa1650_9P',
                init_config=init_config,
                run_name=run_name,
                num_antennas=None,
                num_evaluations=100000,
                array_constraint=array_constraint
            )
            init_config = final_config
