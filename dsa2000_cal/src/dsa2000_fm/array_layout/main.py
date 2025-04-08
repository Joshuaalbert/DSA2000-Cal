import os

from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.ray_utils import TimerLog
from dsa2000_fm.array_layout.fiber_cost_fn import compute_mst
from dsa2000_fm.array_layout.pareto_front_search import build_search_point_generator, SampleEvaluation

os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
# os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"


import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import pylab as plt
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.astropy_utils import mean_itrs

from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_fm.abc import AbstractArrayConstraint
from dsa2000_common.common.quantity_utils import quantity_to_np
from dsa2000_fm.array_layout.psf_quality_fn import dense_annulus, sparse_annulus, create_target, evaluate_psf
from dsa2000_assets.registries import array_registry
from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.array_constraints.array_constraint_content import ArrayConstraintsV6

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

tfpd = tfp.distributions


def create_lmn_target():
    lm = np.concatenate(
        [
            dense_annulus(inner_radius=0., outer_radius=quantity_to_np(1 * au.arcmin),
                          dl=quantity_to_np((3.3 / 7) * au.arcsec), frac=1., dtype=jnp.float64),
            sparse_annulus(key=jax.random.PRNGKey(0), inner_radius=quantity_to_np(1 * au.arcmin),
                           outer_radius=quantity_to_np(0.5 * au.deg), num_samples=1000, dtype=jnp.float64),
            sparse_annulus(key=jax.random.PRNGKey(1), inner_radius=quantity_to_np(0.5 * au.deg),
                           outer_radius=quantity_to_np(1.5 * au.deg), num_samples=1000, dtype=jnp.float64)
        ],
        axis=0
    )
    n = 1. - np.square(lm).sum(axis=-1, keepdims=True)
    return jnp.concatenate([lm, n], axis=-1)


def create_lmn_inner():
    lm = np.concatenate(
        [
            dense_annulus(inner_radius=0., outer_radius=quantity_to_np(1 * au.arcmin),
                          dl=quantity_to_np((3.3 / 7) * au.arcsec), frac=1., dtype=jnp.float64)
        ],
        axis=0
    )
    n = 1. - np.square(lm).sum(axis=-1, keepdims=True)
    return jnp.concatenate([lm, n], axis=-1)


def main(
        run_name: str,
        init_config: str | None,
        target_array_name: str,
        array_constraint: AbstractArrayConstraint,
        num_antennas: int | None = None
):
    os.makedirs(run_name, exist_ok=True)
    plot_folder = os.path.join(run_name, 'plots')
    os.makedirs(plot_folder, exist_ok=True)

    key = jax.random.PRNGKey(0)
    np.random.seed(0)

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
            fill_registries()
            array = array_registry.get_instance(array_registry.get_match(target_array_name))
            antennas0 = array.get_antennas()
            array_location = array.get_array_location()

    if num_antennas is not None:
        keep_idxs = np.random.choice(antennas0.shape[0], num_antennas, replace=False)
        antennas0 = antennas0[keep_idxs]

    obstime = at.Time('2021-01-01T00:00:00', format='isot', scale='utc')
    antennas = antennas0.copy()

    freqs = np.linspace(700e6, 2000e6, 10000) * au.Hz
    decs = [0, -30, 30, 60, 90] * au.deg
    freqs_jax = quantity_to_jnp(freqs, 'Hz')
    decs_jax = quantity_to_jnp(decs, 'rad')

    dsa_logger.info(f"Run name: {run_name}")
    dsa_logger.info(f"Target array name: {target_array_name}")
    dsa_logger.info(f"Initial configuration: {init_config}")
    dsa_logger.info(f"Array constraint: {array_constraint}")
    dsa_logger.info(f"Number of antennas: {len(antennas)}")

    dsa_logger.info(f"Number of frequencies: {len(freqs)}")
    dsa_logger.info(f"Declinations: {decs}")

    with TimerLog("Creating LMN sample points"):
        lmn_inner = create_lmn_inner()
        lmn = create_lmn_target()

        dsa_logger.info(f"LMN shape: {lmn.shape}")
        dsa_logger.info(f"LMN inner shape: {lmn_inner.shape}")

        with TimerLog("Calculating target PSF"):
            target_log_psf_mean, target_log_psf_stddev = create_target(
                key=key,
                target_array_name=target_array_name,
                lmn=lmn_inner,
                freqs=freqs,
                transit_decs=decs,
                num_samples=1000,
                num_antennas=num_antennas,
                accumulate_dtype=jnp.float32
            )
        with TimerLog("Calculating inner target PSF"):
            target_log_psf_mean_inner, target_log_psf_stddev_inner = create_target(
                key=key,
                target_array_name=target_array_name,
                lmn=lmn_inner,
                freqs=freqs, transit_decs=decs[:1],
                num_samples=20,
                num_antennas=num_antennas,
                accumulate_dtype=jnp.float32
            )

        # Plots the target
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        sc = axs[0, 0].scatter(lmn_inner[..., 0].flatten(), lmn_inner[..., 1].flatten(),
                               c=target_log_psf_mean_inner.flatten(), s=1, cmap='jet', marker='.',
                               vmin=-60, vmax=10 * np.log10(0.5))
        plt.colorbar(sc, ax=axs[0, 0], label='Power (dB)')
        axs[0, 0].set_xlabel('l (proj.rad)')
        axs[0, 0].set_ylabel('m (proj.rad)')
        axs[0, 0].set_title('Target mean inner PSF')
        sc = axs[1, 0].scatter(lmn_inner[..., 0].flatten(), lmn_inner[..., 1].flatten(),
                               c=target_log_psf_stddev_inner.flatten(), s=1,
                               cmap='jet', marker='.')
        plt.colorbar(sc, ax=axs[1, 0], label='Power (dB)')
        axs[1, 0].set_xlabel('l (proj.rad)')
        axs[1, 0].set_ylabel('m (proj.rad)')
        axs[1, 0].set_title('Target stddev inner PSF')
        fig.savefig(os.path.join(plot_folder, f'target_psf.png'))
        plt.close(fig)

    # Performing the optimization

    gen = build_search_point_generator(
        results_file=os.path.join(run_name, 'results.json'),
        plot_dir=plot_folder,
        array_constraint=array_constraint,
        antennas=antennas,
        array_location=array_location,
        obstime=obstime,
        additional_buffer=0 * au.m,
        minimal_antenna_sep=8 * au.m
    )

    gen_response = None
    while True:
        try:
            sample_point = gen.send(gen_response)
        except StopIteration:
            break
        proposal_antennas_enu = quantity_to_jnp(
            sample_point.antennas.get_itrs(obstime=obstime, location=array_location).transform_to(
                ENU(obstime=obstime, location=array_location)
            ).cartesian.xyz.T,
            'm'
        )
        latitude = quantity_to_jnp(sample_point.latitude, 'rad')
        quality = evaluate_psf(
            antennas_enu=proposal_antennas_enu,
            lmn=lmn,
            latitude=latitude,
            freqs=freqs_jax,
            decs=decs_jax,
            target_log_psf_mean=target_log_psf_mean,
            target_log_psf_stddev=target_log_psf_stddev
        )
        cost, _, _ = compute_mst(
            k=6,
            antennas=sample_point.antennas,
            obstime=obstime,
            array_location=array_location,
            plot=False,
            save_file='mst'
        )
        gen_response = SampleEvaluation(
            quality=quality,
            cost=cost,
            antennas=sample_point.antennas
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

    init_config = None
    for prefix in ['full', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        array_constraint = ArrayConstraintsV6(prefix)
        run_name = f"pareto_opt_v6_{prefix}"
        new_config = main(
            target_array_name='dsa1650_9P',
            init_config=init_config,
            run_name=run_name,
            num_antennas=None,
            array_constraint=array_constraint
        )
        init_config = new_config
