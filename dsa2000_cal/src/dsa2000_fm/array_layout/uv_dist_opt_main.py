import os
import time

os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
# os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import pylab as plt
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from dsa2000_fm.array_layout.optimal_transport import compute_ideal_uv_distribution, accumulate_uv_distribution
from dsa2000_fm.array_layout.sample_constraints import RegionSampler, sample_aoi

from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_fm.abc import AbstractArrayConstraint
from dsa2000_assets.registries import array_registry
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.array_constraints.v6.array_constraint import ArrayConstraintsV6

tfpd = tfp.distributions


def main(
        uv_bins,
        target_dist,
        conv_size,
        transit_dec,
        obstimes,
        ref_time,
        run_name: str,
        target_array_name: str,
        array_constraint: AbstractArrayConstraint,
        num_antennas: int,
        num_trials_per_antenna: int
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

    times_jax = time_to_jnp(obstimes, ref_time)

    dsa_logger.info(f"Run name: {run_name}")
    dsa_logger.info(f"Array constraint: {array_constraint}")

    # Set up constraint data
    aoi_data = array_constraint.get_area_of_interest_regions()
    # merge AOI's
    merged_aoi_sampler = RegionSampler.merge([s for s, _ in aoi_data])
    merged_buffer = max([b for _, b in aoi_data])
    aoi_data = [(merged_aoi_sampler, merged_buffer)]
    constraint_data = array_constraint.get_constraint_regions()

    # Two totally random antennas to start with
    antennas = ac.EarthLocation(
        x=[0, 0] * au.m,
        y=[0, 0] * au.m,
        z=[0, 0] * au.m
    )
    antennas = sample_aoi(
        replace_idx=0,
        antennas=antennas,
        array_location=array_location,
        obstime=ref_time,
        additional_buffer=0.,
        minimal_antenna_sep=8.,
        aoi_data=aoi_data,
        constraint_data=constraint_data
    )
    antennas = sample_aoi(
        replace_idx=1,
        antennas=antennas,
        array_location=array_location,
        obstime=ref_time,
        additional_buffer=0.,
        minimal_antenna_sep=8.,
        aoi_data=aoi_data,
        constraint_data=constraint_data
    )

    while len(antennas) < num_antennas:
        t0 = time.time()
        best_config = ac.EarthLocation(
            x=np.concatenate([antennas.x, antennas.x[:1]]),
            y=np.concatenate([antennas.y, antennas.y[:1]]),
            z=np.concatenate([antennas.z, antennas.z[:1]])
        )
        best_loss = np.inf
        replace_idx = len(best_config) - 1
        objective = []
        acceptances = 0
        for _ in range(num_trials_per_antenna):
            # and a random antenna to end, and sample AOI to replace it
            proposal_antennas = sample_aoi(
                replace_idx, best_config, array_location, ref_time, additional_buffer=0.,
                minimal_antenna_sep=8., aoi_data=aoi_data, constraint_data=constraint_data
            )
            # Evaluate the new config
            proposal_antennas_gcrs = quantity_to_jnp(
                proposal_antennas.get_gcrs(obstime=ref_time).cartesian.xyz.T,
                'm'
            )
            dist = accumulate_uv_distribution(proposal_antennas_gcrs, times_jax, ra0, transit_dec, uv_bins,
                                              conv_size=conv_size)
            target_dist *= np.sum(dist) / np.sum(target_dist)
            diff = dist - target_dist
            loss = np.max(np.abs(diff))
            objective.append(loss)
            # Evaluate the quality
            if loss < best_loss:
                acceptances += 1
                best_loss = loss
                best_config = proposal_antennas
                save_name = os.path.join(run_name, f'best_config_{len(antennas)}.txt')
                with open(save_name, 'w') as f:
                    for antenna in best_config:
                        f.write(f"{antenna.x.to('m').value},{antenna.y.to('m').value},{antenna.z.to('m').value}\n")
                # Plot the distribution
                fig, axs = plt.subplots(2, 2, figsize=(16, 16))
                axs = axs.flatten()
                # Target, dist, diff
                axs[0].imshow(
                    target_dist.T, extent=[uv_bins[0], uv_bins[-1], uv_bins[0], uv_bins[-1]],
                    origin='lower', interpolation='nearest', cmap='jet'
                )
                axs[0].set_title('Target distribution')
                axs[0].set_xlabel('u (m)')
                axs[0].set_ylabel('v (m)')
                axs[1].imshow(
                    dist.T, extent=[uv_bins[0], uv_bins[-1], uv_bins[0], uv_bins[-1]],
                    origin='lower', interpolation='nearest', cmap='jet'
                )
                axs[1].set_title('Proposed distribution')
                axs[1].set_xlabel('u (m)')
                axs[1].set_ylabel('v (m)')
                axs[2].imshow(
                    diff.T, extent=[uv_bins[0], uv_bins[-1], uv_bins[0], uv_bins[-1]],
                    origin='lower', interpolation='nearest', cmap='jet'
                )
                axs[2].set_title('Difference distribution')
                axs[2].set_xlabel('u (m)')
                axs[2].set_ylabel('v (m)')
                plt.colorbar(ax=axs[0], label=f'Density (ant/{du * du})')
                plt.colorbar(ax=axs[1], label=f'Density (ant/{du * du})')
                plt.colorbar(ax=axs[2], label=f'Density (ant/{du * du})')

                axs[3].plot(objective)
                axs[3].set_title('Objective function')
                axs[3].set_xlabel('Trial')
                axs[3].set_ylabel('Objective (ant/{du * du})')
                axs[3].set_yscale('log')

                save_fig = os.path.join(plot_folder, f'uv_dist_{len(antennas)}.png')
                plt.savefig(save_fig)
                plt.close(fig)
        t1 = time.time()
        acceptance_rate = acceptances / num_trials_per_antenna
        antennas = best_config
        dsa_logger.info(
            f"{len(antennas)} ({100 * len(antennas) / num_antennas:.3f}%): Best loss: {best_loss:.3e} | Acceptance rate: {acceptance_rate:.5f} | Time: {t1 - t0:.2f} seconds"
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

    du = 20 * au.m
    dconv = 200 * au.m
    conv_size = (int(dconv / du) // 2) * 2 + 1
    R = 16000 * au.m
    target_fwhm = 3.14 * au.arcsec
    freq = 1350 * au.MHz
    uv_bins, uv_grid, target_dist = compute_ideal_uv_distribution(du, R, target_fwhm, freq)

    np.random.seed(0)
    while True:
        # From smallest to largest, so smaller one fits in next as good starting point
        for prefix in ['a']:
            array_constraint = ArrayConstraintsV6(prefix)
            run_name = f"pareto_opt_v6_{prefix}_v2"
            final_config = main(
                uv_bins=uv_bins,
                target_dist=target_dist,
                conv_size=conv_size,
                obstimes=obstimes,
                transit_dec=0.,
                ref_time=ref_time,
                target_array_name='dsa1650_9P',
                run_name=run_name,
                array_constraint=array_constraint,
                num_antennas=1650,
                num_trials_per_antenna=10
            )
