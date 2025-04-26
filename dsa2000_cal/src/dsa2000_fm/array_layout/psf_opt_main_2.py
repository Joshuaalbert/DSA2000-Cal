import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'

import gc
import datetime
import time
import warnings

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import pylab as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from dsa2000_common.common.astropy_utils import get_time_of_local_meridean

from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.ray_utils import TimerLog
from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_fm.abc import AbstractArrayConstraint
from dsa2000_common.common.quantity_utils import quantity_to_np
from dsa2000_fm.array_layout.psf_quality_fn import dense_annulus, create_target, evaluate_psf
from dsa2000_assets.registries import array_registry
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.array_constraints.v6.array_constraint import ArrayConstraintsV6
from dsa2000_fm.array_layout.sample_constraints import RegionSampler, sample_aoi, is_violation
from dsa2000_common.common.array_types import FloatArray

tfpd = tfp.distributions


def create_lmn_target():
    lm = dense_annulus(
        inner_radius=0.,
        outer_radius=quantity_to_np(200 * au.arcsec),
        dl=quantity_to_np(1. * au.arcsec),
        frac=1.,
        dtype=jnp.float64
    )
    # M = np.shape(dense_inner)[0] # for balancing far sidelobe
    # lm = np.concatenate(
    #     [
    #         dense_inner,
    #         # sparse_annulus(key=jax.random.PRNGKey(0), inner_radius=quantity_to_np(1 * au.arcmin),
    #         #                outer_radius=quantity_to_np(0.5 * au.deg), num_samples=M, dtype=jnp.float64),
    #         # sparse_annulus(key=jax.random.PRNGKey(1), inner_radius=quantity_to_np(0.5 * au.deg),
    #         #                outer_radius=quantity_to_np(1.5 * au.deg), num_samples=M, dtype=jnp.float64)
    #     ],
    #     axis=0
    # )

    n = np.sqrt(1. - np.square(lm).sum(axis=-1, keepdims=True))
    return jnp.concatenate([lm, n], axis=-1)


def create_psf_target(target_array_name: str, freqs: au.Quantity, ra0: au.Quantity, dec0s: au.Quantity,
                      ref_time: at.Time, num_antennas: int | None):
    np.random.seed(0)
    key = jax.random.PRNGKey(0)

    dsa_logger.info(f"Target array name: {target_array_name}")

    with TimerLog("Creating LMN sample points"):
        lmn_target = create_lmn_target()

        dsa_logger.info(f"LMN shape: {lmn_target.shape}")

        with TimerLog("Calculating target PSF"):
            target_psf_dB_mean, target_psf_dB_stddev = create_target(
                key=key,
                target_array_name=target_array_name,
                lmn=lmn_target,
                freqs=freqs, ra0=ra0, dec0s=dec0s,
                ref_time=ref_time,
                num_samples=1000,
                accumulate_dtype=jnp.float32,
                num_antennas=num_antennas
            )
            # Check finite
            if not np.all(np.isfinite(target_psf_dB_mean)):
                raise ValueError("Target PSF mean is not finite")
            if not np.all(np.isfinite(target_psf_dB_stddev)):
                raise ValueError("Target PSF stddev is not finite")

    return lmn_target, target_psf_dB_mean, target_psf_dB_stddev


def evaluate_loss(proposal_antennas: ac.EarthLocation, lmn_target: FloatArray, freqs: FloatArray, ref_time: at.Time,
                  ra0: FloatArray, dec0s: FloatArray, target_psf_dB_mean: FloatArray, target_psf_dB_stddev: FloatArray):
    # Evaluate the new config
    proposal_antennas_gcrs = quantity_to_jnp(
        proposal_antennas.get_gcrs(obstime=ref_time).cartesian.xyz.T,
        'm'
    )
    loss, psf_dB, z_scores = evaluate_psf(
        antennas_gcrs=proposal_antennas_gcrs,
        lmn=lmn_target, freqs=freqs, ra0=ra0,
        dec0s=dec0s,
        target_psf_dB_mean=target_psf_dB_mean,
        target_psf_dB_stddev=target_psf_dB_stddev,
        accumulate_dtype=jnp.float32
    )
    return loss, psf_dB, z_scores


def run(
        lmn_target: FloatArray,
        target_psf_dB_mean: FloatArray, target_psf_dB_stddev: FloatArray,
        freqs: FloatArray, ra0: FloatArray, dec0s: FloatArray,
        array_location,
        ref_time,
        run_name: str,
        min_antenna_sep_m: float,
        array_constraint: AbstractArrayConstraint,
        num_antennas: int,
        num_trials_per_antenna: int,
        num_time_per_antenna_s: float,
        deadline: datetime.datetime | None = None,
        resume_ant: int | None = None,
        random_refinement: bool = False
):
    os.makedirs(run_name, exist_ok=True)
    plot_folder = os.path.join(run_name, 'plots')
    os.makedirs(plot_folder, exist_ok=True)

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

    while random_refinement or len(antennas) < num_antennas:
        # clear JAX cache
        jax.clear_caches()
        gc.collect()
        t0 = time.time()
        objective = []
        acceptances = 0
        trials = 0

        if len(antennas) < num_antennas:
            best_config = ac.EarthLocation(
                x=np.concatenate([antennas.x, antennas.x[:1]]),
                y=np.concatenate([antennas.y, antennas.y[:1]]),
                z=np.concatenate([antennas.z, antennas.z[:1]])
            )
            best_loss = np.inf
            best_dist = best_diff = target_psf_dB_mean, target_psf_dB_stddev
            replace_idx = -1
            if deadline is not None:
                time_left = deadline - datetime.datetime.now(datetime.timezone.utc)
                num_time_per_antenna_s = time_left.total_seconds() / (num_antennas - len(antennas))

        else:
            best_config = antennas
            best_loss, best_dist, best_diff = evaluate_loss(
                antennas, lmn_target, freqs, ref_time, ra0, dec0s, target_psf_dB_mean, target_psf_dB_stddev
            )
            replace_idx = np.random.randint(0, len(antennas))

        assert num_time_per_antenna_s is not None, "num_time_per_antenna_s must be set"

        while (time.time() - t0 < num_time_per_antenna_s) and (trials < num_trials_per_antenna):
            trials += 1
            # sample AOI to replace it
            proposal_antennas = sample_aoi(
                replace_idx, best_config, array_location, ref_time, additional_buffer=0.,
                minimal_antenna_sep=min_antenna_sep_m, aoi_data=aoi_data, constraint_data=constraint_data
            )
            # loss, psf_dB, z_scores
            loss, dist, diff = evaluate_loss(
                proposal_antennas, lmn_target, freqs, ref_time, ra0, dec0s, target_psf_dB_mean, target_psf_dB_stddev
            )

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
        plot_result(target_psf_dB_mean, best_dist, best_diff, antennas, objective, lmn_target, aoi_data,
                    constraint_data, plot_folder)
        save_name = os.path.join(run_name, f'best_config_{len(antennas):04d}.txt')
        with open(save_name, 'w') as f:
            for antenna in best_config:
                f.write(f"{antenna.x.to('m').value},{antenna.y.to('m').value},{antenna.z.to('m').value}\n")
        dsa_logger.info(
            f"{len(antennas)} ({100 * len(antennas) / num_antennas:.3f}%): Best loss: {best_loss:.3e} | "
            f"Acceptance rate: {acceptance_rate:.6f}% | # Trials: {trials} | Time: {t1 - t0:.2f} seconds"
        )
        if deadline is not None:
            if datetime.datetime.now(datetime.timezone.utc) > deadline:
                dsa_logger.info(f"Deadline reached. Stopping.")
                break
    # Store to final_config
    final_config = os.path.join(run_name, 'final_config.txt')
    with open(final_config, 'w') as f:
        for antenna in antennas:
            f.write(f"{antenna.x.to('m').value},{antenna.y.to('m').value},{antenna.z.to('m').value}\n")


def plot_result(target_dist, dist, diff, antennas, objective, lmn_target, aoi_data, constraint_data, plot_folder):
    # Get the 0 dec slice
    target_dist = target_dist[0]
    dist = dist[0]
    diff = diff[0]
    rad2arcsec = 3600 * 180 / np.pi
    l = lmn_target[..., 0] * rad2arcsec
    m = lmn_target[..., 1] * rad2arcsec
    # Plot the distribution
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    axs = axs.flatten()
    # Target, dist, diff
    im1 = axs[0].scatter(
        l, m,
        c=target_dist,
        s=1,
        cmap='jet',
        vmin=-60, vmax=10 * np.log10(1)
    )
    axs[0].set_title('Target distribution')
    axs[0].set_xlabel('l (proj. arcsec)')
    axs[0].set_ylabel('m (proj. arcsec)')
    im2 = axs[1].scatter(
        l, m,
        c=dist,
        s=1,
        cmap='jet',
        vmin=-60, vmax=10 * np.log10(1)
    )
    axs[1].set_title('Proposed distribution')
    axs[1].set_xlabel('l (proj. arcsec)')
    axs[1].set_ylabel('m (proj. arcsec)')
    p5, p95 = np.percentile(diff, [5, 95])
    vmin = min(p5, 0)
    vmax = max(p95, 0)
    vmin, vmax = min(vmin, -vmax), max(vmax, -vmin)
    im3 = axs[2].scatter(
        l, m,
        c=diff,
        cmap='jet',
        s=1,
        vmin=vmin, vmax=vmax
    )
    axs[2].set_title('Difference distribution')
    axs[2].set_xlabel('l (proj. arcsec)')
    axs[2].set_ylabel('m (proj. arcsec)')
    fig.colorbar(im1, ax=axs[0], label=f'Power (dB)')
    fig.colorbar(im2, ax=axs[1], label=f'Power (dB)')
    fig.colorbar(im3, ax=axs[2], label=f'Z-score')

    for sampler, buffer in aoi_data:
        # sampler.info()
        sampler.plot_region(ax=axs[3], color='blue')
    for sampler, buffer in constraint_data:
        sampler.plot_region(ax=axs[3], color='none')

    axs[3].scatter(antennas.geodetic.lon.deg, antennas.geodetic.lat.deg, s=1, c='green', alpha=0.5, marker='.')
    axs[3].set_xlabel('Longitude [deg]')
    axs[3].set_ylabel('Latitude [deg]')
    axs[3].set_title('Antenna layout')
    axs[3].set_xlim(-114.6, -114.3)
    axs[3].set_ylim(39.45, 39.70)

    # make a small inset in the upper‐right of axs[3], sized at 30%×30% of the parent
    axins = inset_axes(axs[3],
                       width="30%",  # width of inset: 30% of parent axes
                       height="30%",  # height of inset
                       loc='upper right',
                       borderpad=1)

    # plot cumulative‐min objective
    cummin_obj = np.minimum.accumulate(objective)
    axins.plot(cummin_obj, color='k', lw=1)
    axins.set_title('Cum.‐min loss', fontsize=8, pad=2)
    axins.set_xlabel('Trial', fontsize=6)
    axins.set_ylabel('Loss', fontsize=6)
    axins.tick_params(axis='both', which='major', labelsize=6)

    # give the inset a white background at alpha=0.7 so it doesn't obscure too much
    axins.patch.set_facecolor('white')
    axins.patch.set_alpha(0.7)

    save_fig = os.path.join(plot_folder, f'best_solution_{len(antennas):04d}.png')
    plt.savefig(save_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(
        target_array_name,
        prefix,
        num_antennas,
        num_trials_per_antenna,
        num_time_per_antenna_s,
        deadline_dt,
        min_antenna_sep_m,
        resume_ant: int | None,
        random_refinement: bool
):
    warnings.filterwarnings(
        "ignore",
        category=UserWarning
    )

    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(target_array_name))
    array_location = array.get_array_location()
    ref_time = at.Time("2025-06-10T00:00:00", format='isot', scale='utc')
    phase_tracking = ENU(0, 0, 1, obstime=ref_time, location=array_location).transform_to(ac.ICRS())
    ra0 = phase_tracking.ra
    ref_time = get_time_of_local_meridean(ac.ICRS(ra=ra0, dec=0 * au.deg), location=array_location, ref_time=ref_time)
    dsa_logger.info(f"Reference time: {ref_time.isot}")

    # freqs = np.linspace(700e6, 2000e6, 2) * au.Hz
    freqs = [1350] * au.MHz
    dec0s = [0] * au.deg

    lmn_target, target_psf_dB_mean, target_psf_dB_stddev = create_psf_target(
        target_array_name=target_array_name,
        freqs=freqs,
        ra0=ra0,
        dec0s=dec0s,
        ref_time=ref_time,
        num_antennas=None
    )

    array_constraint = ArrayConstraintsV6(prefix)
    run_name = f"pareto_opt_v6_{prefix}_{target_array_name}_v2.4"
    run(
        lmn_target, target_psf_dB_mean, target_psf_dB_stddev,
        freqs=quantity_to_jnp(freqs, 'Hz'),
        ra0=quantity_to_jnp(ra0, 'rad'),
        dec0s=quantity_to_jnp(dec0s, 'rad'),
        array_location=array_location,
        ref_time=ref_time,
        run_name=run_name,
        array_constraint=array_constraint,
        num_antennas=num_antennas,
        num_trials_per_antenna=num_trials_per_antenna,
        num_time_per_antenna_s=num_time_per_antenna_s,
        min_antenna_sep_m=min_antenna_sep_m,
        deadline=deadline_dt,
        resume_ant=resume_ant,
        random_refinement=random_refinement
    )


if __name__ == '__main__':
    deadline = datetime.datetime.fromisoformat("2025-04-27T12:00:00-07:00")

    main(
        target_array_name='dsa1650_9279',
        prefix='a',
        num_antennas=1650,
        num_trials_per_antenna=100000,  # max 10000
        num_time_per_antenna_s=10,
        deadline_dt=deadline,  # use deadline to set time per round
        min_antenna_sep_m=8.,
        resume_ant=None,
        random_refinement=True
    )
