import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'

import datetime
import time
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import pylab as plt
import numpy as np
import jax
import tensorflow_probability.substrates.jax as tfp

from dsa2000_fm.array_layout.optimal_transport import compute_ideal_uv_distribution, accumulate_uv_distribution
from dsa2000_fm.array_layout.sample_constraints import RegionSampler, sample_aoi, is_violation
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_fm.abc import AbstractArrayConstraint
from dsa2000_assets.registries import array_registry
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.array_constraints.v6.array_constraint import ArrayConstraintsV6

tfpd = tfp.distributions


def run(
        uv_bins,
        target_dist,
        conv_size,
        transit_dec,
        obstimes,
        ref_time,
        run_name: str,
        target_array_name: str,
        min_antenna_sep_m: float,
        array_constraint: AbstractArrayConstraint,
        num_antennas: int,
        num_trials_per_antenna: int,
        num_time_per_antenna_s: float,
        loss_obj: str,
        deadline: datetime.datetime | None = None,
        resume_ant: int | None = None
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
    if resume_ant is None:
        antennas = ac.EarthLocation(
            x=[0] * au.m,
            y=[0] * au.m,
            z=[0] * au.m
        )
        antennas = sample_aoi(
            replace_idx=0,
            antennas=antennas,
            array_location=array_location,
            obstime=ref_time,
            additional_buffer=0.,
            minimal_antenna_sep=min_antenna_sep_m,
            aoi_data=aoi_data,
            constraint_data=constraint_data
        )
    else:
        resume_file = os.path.join(run_name, f'best_config_{resume_ant:04d}.txt')
        if not os.path.exists(resume_file):
            raise FileNotFoundError(f"Resume file {resume_file} does not exist.")
        x, y, z = [], [], []
        with open(resume_file, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                _x, _y, _z = line.strip().split(',')
                x.append(float(_x))
                y.append(float(_y))
                z.append(float(_z))
            antennas = ac.EarthLocation.from_geocentric(
                x * au.m,
                y * au.m,
                z * au.m
            )

    # Check for violating antennas and remove them
    for check_idx in reversed(range(len(antennas))):
        if is_violation(
                check_idx=check_idx,
                antennas=antennas,
                array_location=array_location,
                obstime=ref_time,
                additional_buffer=0.,
                minimal_antenna_sep=min_antenna_sep_m,
                aoi_data=aoi_data,
                constraint_data=constraint_data,
                verbose=False
        ):
            # remove that antenna
            antennas = ac.EarthLocation(
                x=np.concatenate((antennas.x[:check_idx], antennas.x[check_idx + 1:])),
                y=np.concatenate((antennas.y[:check_idx], antennas.y[check_idx + 1:])),
                z=np.concatenate((antennas.z[:check_idx], antennas.z[check_idx + 1:]))
            )

    while len(antennas) < num_antennas:
        # clear JAX cache
        jax.clear_caches()
        t0 = time.time()
        best_config = ac.EarthLocation(
            x=np.concatenate([antennas.x, antennas.x[:1]]),
            y=np.concatenate([antennas.y, antennas.y[:1]]),
            z=np.concatenate([antennas.z, antennas.z[:1]])
        )
        best_loss = np.inf

        best_dist = best_diff = target_dist
        replace_idx = -1
        objective = []
        acceptances = 0
        trials = 0

        if deadline is not None:
            time_left = deadline - datetime.datetime.now(datetime.timezone.utc)
            num_time_per_antenna_s = time_left.total_seconds() / (num_antennas - len(antennas))

        assert num_time_per_antenna_s is not None

        while (time.time() - t0 < num_time_per_antenna_s) and (trials < num_trials_per_antenna):
            trials += 1
            # sample AOI to replace it
            proposal_antennas = sample_aoi(
                replace_idx, best_config, array_location, ref_time, additional_buffer=0.,
                minimal_antenna_sep=min_antenna_sep_m, aoi_data=aoi_data, constraint_data=constraint_data
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
            if loss_obj == 'lst_sq':
                loss = np.sqrt(np.mean(np.abs(diff) ** 2))
            elif loss_obj == 'max':
                loss = np.max(np.abs(diff))
            else:
                raise ValueError(f"loss_obj must be 'lst_sq' or 'max' of {loss_obj}")
            objective.append(loss)
            # Evaluate the quality
            if loss < best_loss:
                acceptances += 1
                best_loss = loss
                best_config = proposal_antennas
                best_dist = dist
                best_diff = diff
        t1 = time.time()
        acceptance_rate = 100 * acceptances / num_trials_per_antenna
        antennas = best_config
        plot_result(target_dist, best_dist, antennas, best_diff, objective, uv_bins, plot_folder)
        save_name = os.path.join(run_name, f'best_config_{len(antennas):04d}.txt')
        with open(save_name, 'w') as f:
            for antenna in best_config:
                f.write(f"{antenna.x.to('m').value},{antenna.y.to('m').value},{antenna.z.to('m').value}\n")
        dsa_logger.info(
            f"{len(antennas)} ({100 * len(antennas) / num_antennas:.3f}%): Best loss: {best_loss:.3e} | Acceptance rate: {acceptance_rate:.6f}% | # Trials: {trials} | Time: {t1 - t0:.2f} seconds"
        )
    # Store to final_config
    final_config = os.path.join(run_name, 'final_config.txt')
    with open(final_config, 'w') as f:
        for antenna in antennas:
            f.write(f"{antenna.x.to('m').value},{antenna.y.to('m').value},{antenna.z.to('m').value}\n")


def plot_result(target_dist, dist, antennas, diff, objective, uv_bins, plot_folder):
    du = uv_bins[1] - uv_bins[0]
    # Plot the distribution
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    axs = axs.flatten()
    # Target, dist, diff
    im1 = axs[0].imshow(
        target_dist.T, extent=[uv_bins[0], uv_bins[-1], uv_bins[0], uv_bins[-1]],
        origin='lower', interpolation='nearest', cmap='jet'
    )
    axs[0].set_title('Target distribution')
    axs[0].set_xlabel('u (m)')
    axs[0].set_ylabel('v (m)')
    im2 = axs[1].imshow(
        dist.T, extent=[uv_bins[0], uv_bins[-1], uv_bins[0], uv_bins[-1]],
        origin='lower', interpolation='nearest', cmap='jet'
    )
    axs[1].set_title('Proposed distribution')
    axs[1].set_xlabel('u (m)')
    axs[1].set_ylabel('v (m)')
    im3 = axs[2].imshow(
        diff.T, extent=[uv_bins[0], uv_bins[-1], uv_bins[0], uv_bins[-1]],
        origin='lower', interpolation='nearest', cmap='jet'
    )
    axs[2].set_title('Difference distribution')
    axs[2].set_xlabel('u (m)')
    axs[2].set_ylabel('v (m)')
    plt.colorbar(im1, ax=axs[0], label=f'Density (ant/{du * du})')
    plt.colorbar(im2, ax=axs[1], label=f'Density (ant/{du * du})')
    plt.colorbar(im3, ax=axs[2], label=f'Density (ant/{du * du})')
    axs[3].plot(np.minimum.accumulate(objective))
    axs[3].set_title('Objective (cum.min. loss)')
    axs[3].set_xlabel('Trial')
    axs[3].set_ylabel('Loss (ant/{du * du})')
    axs[3].set_yscale('log')
    save_fig = os.path.join(plot_folder, f'uv_dist_{len(antennas):04d}.png')
    plt.savefig(save_fig)
    plt.close(fig)


def main(target_fwhm_arcsec, loss_obj, prefix, num_antennas, num_trials_per_antenna, num_time_per_antenna_s,
         deadline_dt,
         min_antenna_sep_m,
         du: au.Quantity,
         dconv: au.Quantity,
         resume_ant):
    warnings.filterwarnings(
        "ignore",
        category=UserWarning
    )

    ref_time = at.Time("2025-06-10T00:00:00", format='isot', scale='utc')
    obstimes = ref_time[None]

    conv_size = (int(dconv / du) // 2) * 2 + 1
    R = 16000 * au.m
    freq = 1350 * au.MHz

    # num_time_per_antenna_s = 26 # 12h
    # np.random.seed(0)
    # From smallest to largest, so smaller one fits in next as good starting pointf
    target_fwhm = target_fwhm_arcsec * au.arcsec
    uv_bins, uv_grid, target_dist = compute_ideal_uv_distribution(du, R, target_fwhm, freq)
    array_constraint = ArrayConstraintsV6(prefix)
    run_name = f"pareto_opt_v6_{prefix}_{target_fwhm.to('arcsec').value}arcsec_{loss_obj}_v2"
    run(
        uv_bins=uv_bins,
        target_dist=target_dist,
        conv_size=conv_size,
        obstimes=obstimes,
        transit_dec=0.,
        ref_time=ref_time,
        target_array_name='dsa1650_9P',
        run_name=run_name,
        array_constraint=array_constraint,
        num_antennas=num_antennas,
        num_trials_per_antenna=num_trials_per_antenna,
        num_time_per_antenna_s=num_time_per_antenna_s,
        loss_obj=loss_obj,
        min_antenna_sep_m=min_antenna_sep_m,
        deadline=deadline_dt,
        resume_ant=resume_ant
    )


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

    deadline = datetime.datetime.fromisoformat("2025-04-22T09:00:00-07:00")

    main(
        target_fwhm_arcsec=2.61,
        loss_obj='lst_sq',
        prefix='a',
        num_antennas=1650,
        num_trials_per_antenna=10000, # max 10000
        num_time_per_antenna_s=None,
        deadline_dt=deadline,  # use deadline to set time per round
        min_antenna_sep_m=0.,
        du=20 * au.m,
        dconv=120 * au.m,
        resume_ant=1024
    )
