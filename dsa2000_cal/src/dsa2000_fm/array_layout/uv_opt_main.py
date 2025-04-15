import os

from dsa2000_fm.array_layout.optimal_transport import compute_ideal_uv_distribution, evaluate_uv_distribution

os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
# os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import pylab as plt
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.astropy_utils import mean_itrs
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.ray_utils import TimerLog
from dsa2000_fm.array_layout.fiber_cost_fn import compute_mst_cost
from dsa2000_fm.array_layout.pareto_front_search import build_search_point_generator, SampleEvaluation
from dsa2000_common.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_fm.abc import AbstractArrayConstraint
from dsa2000_common.common.quantity_utils import quantity_to_np
from dsa2000_fm.array_layout.psf_quality_fn import dense_annulus, sparse_annulus, create_target
from dsa2000_assets.registries import array_registry
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.array_constraints.v6.array_constraint import ArrayConstraintsV6

tfpd = tfp.distributions


def create_lmn_target():
    dense_inner = dense_annulus(inner_radius=0., outer_radius=quantity_to_np(1 * au.arcmin),
                                dl=quantity_to_np(0.8 * au.arcsec), frac=1., dtype=jnp.float64)
    M = np.shape(dense_inner)[0]
    lm = np.concatenate(
        [
            dense_inner,
            sparse_annulus(key=jax.random.PRNGKey(0), inner_radius=quantity_to_np(1 * au.arcmin),
                           outer_radius=quantity_to_np(0.5 * au.deg), num_samples=M // 2, dtype=jnp.float64),
            sparse_annulus(key=jax.random.PRNGKey(1), inner_radius=quantity_to_np(0.5 * au.deg),
                           outer_radius=quantity_to_np(1.5 * au.deg), num_samples=M // 2, dtype=jnp.float64)
        ],
        axis=0
    )
    n = 1. - np.square(lm).sum(axis=-1, keepdims=True)
    return jnp.concatenate([lm, n], axis=-1)


def create_lmn_inner():
    lm = np.concatenate(
        [
            dense_annulus(inner_radius=0., outer_radius=quantity_to_np(1 * au.arcmin),
                          dl=quantity_to_np(0.25 * au.arcsec), frac=1., dtype=jnp.float64)
        ],
        axis=0
    )
    n = 1. - np.square(lm).sum(axis=-1, keepdims=True)
    return jnp.concatenate([lm, n], axis=-1)


def create_psf_target(run_name, target_array_name, freqs, decs, num_antennas: int | None):
    os.makedirs(run_name, exist_ok=True)
    plot_folder = os.path.join(run_name, 'plots')
    os.makedirs(plot_folder, exist_ok=True)

    np.random.seed(0)
    key = jax.random.PRNGKey(0)

    dsa_logger.info(f"Target array name: {target_array_name}")

    with TimerLog("Creating LMN sample points"):
        lmn_inner = create_lmn_inner()
        lmn_target = create_lmn_target()

        dsa_logger.info(f"LMN shape: {lmn_target.shape}")
        dsa_logger.info(f"LMN inner shape: {lmn_inner.shape}")

        with TimerLog("Calculating inner target PSF"):
            target_psf_dB_mean_inner, target_psf_dB_stddev_inner = create_target(
                key=key,
                target_array_name=target_array_name,
                lmn=lmn_inner,
                freqs=freqs,
                transit_decs=decs[:1],
                num_samples=20,
                num_antennas=num_antennas,
                accumulate_dtype=jnp.float32
            )
            # Plots the target
            fig, axs = plt.subplots(2, 1, figsize=(16, 16), squeeze=False)
            sc = axs[0, 0].scatter(lmn_inner[..., 0].flatten(), lmn_inner[..., 1].flatten(),
                                   c=target_psf_dB_mean_inner.flatten(), s=1, cmap='jet', marker='.',
                                   vmin=-60, vmax=10 * np.log10(0.5))
            plt.colorbar(sc, ax=axs[0, 0], label='Power (dB)')
            axs[0, 0].set_xlabel('l (proj.rad)')
            axs[0, 0].set_ylabel('m (proj.rad)')
            axs[0, 0].set_title('Target mean inner PSF')
            sc = axs[1, 0].scatter(lmn_inner[..., 0].flatten(), lmn_inner[..., 1].flatten(),
                                   c=target_psf_dB_stddev_inner.flatten(), s=1,
                                   cmap='jet', marker='.')
            plt.colorbar(sc, ax=axs[1, 0], label='Power (dB)')
            axs[1, 0].set_xlabel('l (proj.rad)')
            axs[1, 0].set_ylabel('m (proj.rad)')
            axs[1, 0].set_title('Target stddev inner PSF')
            fig.savefig(os.path.join(plot_folder, f'target_psf_inner.png'))
            plt.close(fig)

        with TimerLog("Calculating target PSF"):
            target_psf_dB_mean, target_psf_dB_stddev = create_target(
                key=key,
                target_array_name=target_array_name,
                lmn=lmn_target,
                freqs=freqs,
                transit_decs=decs,
                num_samples=1000,
                num_antennas=num_antennas,
                accumulate_dtype=jnp.float32
            )
            # Check finite
            if not np.all(np.isfinite(target_psf_dB_mean)):
                raise ValueError("Target PSF mean is not finite")
            if not np.all(np.isfinite(target_psf_dB_stddev)):
                raise ValueError("Target PSF stddev is not finite")
            # Plots the target
            fig, axs = plt.subplots(2, 1, figsize=(16, 16), squeeze=False)
            sc = axs[0, 0].scatter(lmn_target[..., 0].flatten(), lmn_target[..., 1].flatten(),
                                   c=target_psf_dB_mean[0].flatten(), s=1, cmap='jet', marker='.',
                                   vmin=-60, vmax=10 * np.log10(0.5))
            plt.colorbar(sc, ax=axs[0, 0], label='Power (dB)')
            axs[0, 0].set_xlabel('l (proj.rad)')
            axs[0, 0].set_ylabel('m (proj.rad)')
            axs[0, 0].set_title('Target mean PSF')
            sc = axs[1, 0].scatter(lmn_target[..., 0].flatten(), lmn_target[..., 1].flatten(),
                                   c=target_psf_dB_stddev[0].flatten(), s=1,
                                   cmap='jet', marker='.')
            plt.colorbar(sc, ax=axs[1, 0], label='Power (dB)')
            axs[1, 0].set_xlabel('l (proj.rad)')
            axs[1, 0].set_ylabel('m (proj.rad)')
            axs[1, 0].set_title('Target stddev PSF')
            fig.savefig(os.path.join(plot_folder, f'target_psf.png'))
            plt.close(fig)
    return lmn_target, target_psf_dB_mean, target_psf_dB_stddev


def main(
        key,
        uv_bins,
        uv_grid,
        target_dist,
        obsfreqs,
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

    np.random.seed(0)
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(target_array_name))
    array_location = array.get_array_location()
    phase_tracking = ENU(0, 0, 1, obstime=ref_time, location=array_location).transform_to(ac.ICRS())
    ra0 = quantity_to_jnp(phase_tracking.ra, 'rad')
    dec0 = quantity_to_jnp(phase_tracking.dec, 'rad')

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

    freqs_jax = quantity_to_jnp(obsfreqs, 'Hz')
    times_jax = time_to_jnp(obstimes, ref_time)

    dsa_logger.info(f"Run name: {run_name}")
    dsa_logger.info(f"Initial configuration: {init_config}")
    dsa_logger.info(f"Array constraint: {array_constraint}")
    dsa_logger.info(f"Number of antennas: {len(antennas)}")

    dsa_logger.info(f"Number of frequencies: {len(obsfreqs)}")

    # Performing the optimization
    gen = build_search_point_generator(
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
        key, sample_key = jax.random.split(key)
        quality = evaluate_uv_distribution(
            sample_key,
            proposal_antennas_gcrs,
            times_jax,
            freqs_jax,
            ra0,
            dec0,
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

    obsfreqs = np.linspace(700e6, 2000e6, 250) * au.Hz
    ref_time = at.Time("2025-06-10T00:00:00", format='isot', scale='utc')
    obstimes = ref_time + np.linspace(0., 7., 8) * au.min

    du = 100 * au.m
    R = 16000 * au.m
    target_fwhm = 3. * au.arcsec
    max_freq = 2 * au.GHz
    uv_bins, uv_grid, target_dist = compute_ideal_uv_distribution(du, R, target_fwhm, max_freq)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(target_dist, extent=[uv_bins[0], uv_bins[-1], uv_bins[0], uv_bins[-1]], origin='lower',
              interpolation='nearest', cmap='jet')
    ax.set_xlabel('u (lambda)')
    ax.set_ylabel('v (lambda)')
    ax.set_title('Target distribution')
    plt.colorbar(ax=ax, label='Density')
    plt.savefig(f'target_uv_distribution_{target_fwhm.to("arcsec").value}arcsec_psf.png')
    plt.close(fig)

    init_config = 'pareto_opt_v6_a/final_config.txt'
    key = jax.random.PRNGKey(0)
    while True:
        # From smallest to largest, so smaller one fits in next as good starting point
        for prefix in ['a', 'e']:
            array_constraint = ArrayConstraintsV6(prefix)
            run_name = f"pareto_opt_v6_{prefix}_v2"
            key, sample_key = jax.random.split(key)
            final_config = main(
                key=sample_key,
                uv_bins=uv_bins, uv_grid=uv_grid, target_dist=target_dist,
                obsfreqs=obsfreqs,
                obstimes=obstimes,
                ref_time=ref_time,
                target_array_name='dsa1650_9P',
                init_config=init_config,
                run_name=run_name,
                num_antennas=None,
                num_evaluations=500,
                array_constraint=array_constraint
            )
            init_config = final_config
