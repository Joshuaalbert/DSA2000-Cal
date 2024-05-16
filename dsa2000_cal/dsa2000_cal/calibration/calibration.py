import time as time_mod
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Tuple, List

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jaxopt
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import lax
from jax import numpy as jnp
from jax._src.typing import SupportsDType

from dsa2000_cal.common.coord_utils import lmn_to_icrs
from dsa2000_cal.common.jax_utils import pytree_unravel, promote_pytree
from dsa2000_cal.common.plot_utils import plot_antenna_gains
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords, VisibilityData, MeasurementSet
from dsa2000_cal.predict.vec_utils import kron_product
from dsa2000_cal.simulation.simulate_visibilties import SimulateVisibilities
from dsa2000_cal.source_models.corr_translation import flatten_coherencies
from dsa2000_cal.source_models.fits_stokes_I_source_model import FitsStokesISourceModel
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel

tfpd = tfp.distributions


class CalibrationParams(NamedTuple):
    gains_real: jnp.ndarray  # [source, time, ant, chan, 2, 2]
    gains_imag: jnp.ndarray  # [source, time, ant, chan, 2, 2]


class CalibrationData(NamedTuple):
    visibility_coords: VisibilityCoords
    image: jnp.ndarray  # [source, chan, 2, 2]
    lmn: jnp.ndarray  # [source, 3]
    freqs: jnp.ndarray  # [chan]
    obs_vis: jnp.ndarray  # [row, chan, 2, 2]
    obs_vis_weight: jnp.ndarray  # [row, chan, 2, 2]


class CalibrationSolutions(SerialisableBaseModel):
    """
    Calibration solutions, stored in a serialisable format.
    """
    directions: ac.ICRS  # [source]
    times: at.Time  # [time]
    antennas: ac.EarthLocation  # [ant]
    antenna_labels: List[str]  # [ant]
    freqs: au.Quantity  # [chan]
    gains: np.ndarray  # [source, time, ant, chan, 2, 2]


@dataclass(eq=False)
class Calibration:
    # models to calibrate based on. Each model gets a gain direction in the flux weighted direction.
    wsclean_source_models: List[WSCleanSourceModel]
    fits_source_models: List[FitsStokesISourceModel]

    preapply_gain_model: GainModel | None

    # Calibration parameters
    num_iterations: int
    inplace_subtract: bool
    plot_folder: str
    solution_cadence: au.Quantity | None = None
    average_interval: au.Quantity | None = None
    residual_ms_folder: str | None = None
    seed: int = 42
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64
    verbose: bool = False
    num_shards: int = 1

    def __post_init__(self):
        self.num_calibrators = len(self.wsclean_source_models) + len(self.fits_source_models)
        self.key = jax.random.PRNGKey(self.seed)
        if self.solution_cadence is not None and not self.solution_cadence.unit.is_equivalent(au.s):
            raise ValueError("solution_cadence must be in seconds.")
        if self.average_interval is not None and not self.average_interval.unit.is_equivalent(au.s):
            raise ValueError("average_interval must be in seconds.")

    def calibrate(self, ms: MeasurementSet) -> MeasurementSet:
        """
        Calibrate the measurement set, and return the subtracted measurement set.

        Args:
            ms: the measurement set to calibrate

        Returns:
            the subtracted measurement set
        """

        from jax.experimental import mesh_utils
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec
        from jax.sharding import NamedSharding

        P = PartitionSpec

        if len(jax.devices()) < self.num_shards:
            raise ValueError(
                f"Number of devices {len(jax.devices())} is less than the number of shards {self.num_shards}"
            )

        devices = mesh_utils.create_device_mesh((self.num_shards,),
                                                devices=jax.devices()[:self.num_shards])
        mesh = Mesh(devices, axis_names=('chan',))

        def tree_device_put(tree, sharding):
            return jax.tree_map(lambda x: jax.device_put(x, sharding), tree)

        # We perform averaging, by holding the gains constant for a period of time (average_interval)
        # 0, 1 --> integration_time = 1s, average_interval = 2.2s --> num_blocks = 2
        # 0, 1 --> integration_time = 1s, average_interval = 0.5s --> num_blocks = 1
        if self.average_interval is None:
            num_blocks = 1
        else:
            num_blocks = max(1, int(self.average_interval / ms.meta.integration_time))
            rounded_average_interval = num_blocks * ms.meta.integration_time
            print(f"Rounded average interval: {rounded_average_interval}")

        print(f"Running calibration with averaging interval {self.average_interval} / {num_blocks} blocks.")

        # Ensure the freqs are the same in the models
        for wsclean_source_model in self.wsclean_source_models:
            if not np.allclose(ms.meta.freqs.to('Hz'), wsclean_source_model.freqs.to('Hz')):
                raise ValueError("Frequencies in the measurement set and source models must match.")
        for fits_source_model in self.fits_source_models:
            if not np.allclose(ms.meta.freqs.to('Hz'), fits_source_model.freqs.to('Hz')):
                raise ValueError("Frequencies in the measurement set and source models must match.")
        if not self.inplace_subtract:
            if self.residual_ms_folder is None:
                raise ValueError("If not inplace subtracting, residual_ms_folder must be provided.")
            ms = ms.clone(ms_folder=self.residual_ms_folder)
            print("Created a new measurement set for residuals.")

        init_params = self.get_init_params(
            num_source=self.num_calibrators,
            num_time=num_blocks if self.average_interval is None else 1,
            # If averaging, we only have one gain per average interval
            num_ant=len(ms.meta.antennas),
            num_chan=len(ms.meta.freqs)
        )  # [num_source, num_blocks/1 , num_ant, num_freqs, 2, 2]

        calibrator_lmn = au.Quantity(
            np.stack(
                [
                    model.flux_weighted_lmn() for model in self.wsclean_source_models
                ] + [
                    model.flux_weighted_lmn() for model in self.fits_source_models
                ],
                axis=0)
        )  # [num_calibrators, 3]

        cal_sources = lmn_to_icrs(
            lmn=calibrator_lmn,
            phase_tracking=ms.meta.phase_tracking,
            time=ms.ref_time
        )  # [num_calibrators]

        for cal_idx, cal_source in enumerate(cal_sources):
            print(f"Calibrator {cal_idx}: {cal_source}")

        # Metrics
        residual_sum = 0.
        t0 = time_mod.time()

        solutions = []
        # TODO: Apply UV cutoff to ignore galactic plane
        gen = ms.create_block_generator(vis=True, weights=True, flags=True, relative_time_idx=True,
                                        num_blocks=num_blocks)
        gen_response = None
        iteration_idx = 0
        while True:
            try:
                times, visibility_coords, data = gen.send(gen_response)
            except StopIteration:
                break

            if self.preapply_gain_model is not None:
                # Since we pass a single `time` we need time_idx to be relative.
                preapply_gains = []
                for time in times:
                    _preapply_gains = self.preapply_gain_model.compute_gain(
                        freqs=ms.meta.freqs,
                        time=time,
                        phase_tracking=ms.meta.phase_tracking,
                        array_location=ms.meta.array_location,
                        sources=cal_sources
                    )  # [num_calibrators, num_ant, num_chan, 2, 2]
                    preapply_gains.append(_preapply_gains)
                preapply_gains = jnp.stack(preapply_gains,
                                           axis=1)  # [num_calibrators, num_time, num_ant, num_chan, 2, 2]
            else:
                preapply_gains = jnp.tile(
                    jnp.eye(2, dtype=self.float_dtype)[None, None, None, None, :, :],
                    reps=(self.num_calibrators, num_blocks, len(ms.meta.antennas), len(ms.meta.freqs), 1, 1)
                )

            self.key, key = jax.random.split(self.key)

            data_dict = dict(
                freqs=quantity_to_jnp(ms.meta.freqs),
                preapply_gains=preapply_gains,
                init_params=init_params,
                vis_data=jax.tree_map(jnp.asarray, data),
                vis_coords=jax.tree_map(jnp.asarray, visibility_coords)
            )

            data_dict = dict(
                freqs=tree_device_put(data_dict['freqs'], NamedSharding(mesh, P('chan'))),
                preapply_gains=tree_device_put(data_dict['preapply_gains'],
                                               NamedSharding(mesh, P(None, None, None, 'chan'))),
                init_params=tree_device_put(data_dict['init_params'], NamedSharding(mesh, P(None, None, None, 'chan'))),
                vis_data=tree_device_put(data_dict['vis_data'], NamedSharding(mesh, P(None, 'chan'))),
                vis_coords=tree_device_put(data_dict['vis_coords'], NamedSharding(mesh, P()))
            )

            params, (neg_log_likelihood, final_state), residual = self._solve_jax(
                **data_dict
            )
            gen_response = VisibilityData(
                vis=np.asarray(residual)
            )
            solutions.append(np.asarray(params.gains_real + 1j * params.gains_imag))
            residual_sum += np.sum(residual)

            # Plot results
            fig, axs = plt.subplots(1, 1, figsize=(6, 6), squeeze=False)
            axs[0][0].plot(neg_log_likelihood)
            axs[0][0].set_title(fr"Cadence window: {iteration_idx} $-\log \mathcal{{L}}$")
            axs[0][0].set_xlabel("Solver Iteration")
            axs[0][0].set_ylabel("Negative Log Likelihood")
            fig.tight_layout()
            fig.savefig(f"{self.plot_folder}/solver_progress_neg_log_likelihood_cadence_window_{iteration_idx}.png")
            plt.close(fig)
            # Pass forward
            init_params = params
            iteration_idx += 1

        gains = np.concatenate(solutions, axis=1)  # [num_calibrators, num_time, num_ant, num_chan, 2, 2]
        calibration_solutions = CalibrationSolutions(
            gains=gains,
            directions=cal_sources,
            times=ms.meta.times,
            antennas=ms.meta.antennas,
            antenna_labels=ms.meta.antenna_names,
            freqs=ms.meta.freqs
        )
        t1 = time_mod.time()
        solution_file = "calibration_solutions.json"
        with open(solution_file, "w") as fp:
            fp.write(calibration_solutions.json(indent=2))

        print(f"Completed calibration in {t1 - t0} seconds, with residual sum {residual_sum}. "
              f"Calibration solutions saved to {solution_file}.")
        print(f"Residuals stored in {ms}.")
        for antenna_idx in range(len(ms.meta.antennas), len(ms.meta.antennas) // 20):
            fig = plot_antenna_gains(calibration_solutions, antenna_idx=antenna_idx, direction_idx=0)
            fig.savefig(f"{self.plot_folder}/antenna_{antenna_idx}_calibration_solutions.png")
            plt.close(fig)
        print(f"Plots saved to {self.plot_folder}.")

        return ms

    def _build_log_likelihood_and_subtract(self, freqs: jax.Array, preapply_gains: jax.Array, vis_data: VisibilityData,
                                           vis_coords: VisibilityCoords):
        """
        Build the log likelihood function.

        Args:
            freqs: [num_chans] the frequencies in Hz
            preapply_gains: [num_cal, num_time, num_ant, num_chan, 2, 2]
            vis_data: [num_row] batched visibility data
            vis_coords: [num_row] visibility coordinates

        Returns:
            log_likelihood: Callable[[gains], jax.Array]
        """

        simulator = SimulateVisibilities(
            wsclean_source_models=self.wsclean_source_models,
            fits_source_models=self.fits_source_models,
            convention=self.convention,
            dtype=self.dtype,
            verbose=self.verbose,
            plot_folder=self.plot_folder
        )

        vis = simulator.predict_model_visibilities(freqs=freqs, apply_gains=preapply_gains,
                                                   vis_coords=vis_coords)  # [num_cal, num_row, num_chan, 2, 2]

        # vis = jax.lax.with_sharding_constraint(vis, NamedSharding(mesh, P(None, None, 'chan')))'
        # TODO: https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#fsdp-tp-with-shard-map-at-the-top-level

        # vis now contains the model visibilities for each calibrator

        def _apply_gains(gains: jax.Array) -> jax.Array:
            if self.average_interval is None:
                # There is one time index anyways, and it's the same gain for all times.
                time_selection = 0
            else:
                # We index into the time dimension based on the time index in the visibility coordinates.
                time_selection = vis_coords.time_idx

            # V_ij = G_i * V_ij * G_j^H
            g1 = gains[:, vis_coords.antenna_1, time_selection, ...]  # [cal_dirs, num_rows, num_chans, 2, 2]
            g2 = gains[:, vis_coords.antenna_2, time_selection, ...]  # [cal_dirs, num_rows, num_chans, 2, 2]

            @partial(jax.vmap, in_axes=(2, 2, 2), out_axes=2)  # over num_chans
            @partial(jax.vmap, in_axes=(0, 0, 0))  # over cal_dirs
            @partial(jax.vmap, in_axes=(0, 0, 0))  # over num_rows
            def transform(g1, g2, vis):
                return flatten_coherencies(kron_product(g1, vis, g2.T.conj()))  # [4]

            model_vis = transform(g1, g2, vis)  # [cal_dirs, num_rows, num_chan, 4]
            model_vis = jnp.sum(model_vis, axis=0)  # [num_rows, num_chan, 4]
            return model_vis

        def _subtract_model(gains: jax.Array):
            model_vis = _apply_gains(gains)
            residual = vis_data.vis - model_vis
            return residual

        def _log_likelihood(gains: jax.Array) -> jax.Array:
            """
            Compute the log probability of the data given the gains.

            Args:
                gains: [cal_dirs, num_ant, num_time / 1, num_chans, 2, 2]

            Returns:
                log_prob: scalar
            """
            model_vis = _apply_gains(gains)  # [num_rows, num_chan, 4]

            vis_variance = 1. / vis_data.weights  # Should probably use measurement set SIGMA here
            vis_stddev = jnp.sqrt(vis_variance)
            obs_dist_real = tfpd.Normal(*promote_pytree('vis_real', (vis_data.vis.real, vis_stddev)))
            obs_dist_imag = tfpd.Normal(*promote_pytree('vis_imag', (vis_data.vis.imag, vis_stddev)))
            log_prob = obs_dist_real.log_prob(jnp.real(model_vis)) + obs_dist_imag.log_prob(
                jnp.imag(model_vis))  # [num_rows, num_chan, 4]

            # Mask out flagged data or zero-weighted data.
            log_prob = jnp.where(jnp.bitwise_or(vis_data.weights == 0, vis_data.flags), -jnp.inf, log_prob)

            return jnp.sum(log_prob)

        return _log_likelihood, _subtract_model

    @property
    def float_dtype(self):
        # Given self.dtype is complex, find float dtype
        return jnp.real(jnp.zeros((), dtype=self.dtype)).dtype

    def get_init_params(self, num_source: int, num_time: int, num_ant: int, num_chan: int) -> CalibrationParams:
        """
        Get initial parameters.

        Args:
            num_source: number of sources
            num_time: number of times
            num_ant: number of antennas
            num_chan: number of channels

        Returns:
            initial parameters: (gains_real, gains_imag) of shape (num_source, num_time, num_ant, num_chan, 2, 2)
        """
        return CalibrationParams(
            gains_real=jnp.tile(jnp.eye(2, dtype=self.float_dtype)[None, None, None, None, ...],
                                (num_source, num_time, num_ant, num_chan, 1, 1)),
            gains_imag=jnp.tile(jnp.zeros((2, 2), dtype=self.float_dtype)[None, None, None, None, ...],
                                (num_source, num_time, num_ant, num_chan, 1, 1))
        )

    @partial(jax.jit, static_argnums=(0,))
    def _solve_jax(self, freqs: jax.Array, preapply_gains: jax.Array, init_params: CalibrationParams,
                   vis_data: VisibilityData,
                   vis_coords: VisibilityCoords) -> Tuple[
        CalibrationParams, Tuple[jax.Array, jaxopt.OptStep], jax.Array]:

        log_prob_fn, subtract_fn = self._build_log_likelihood_and_subtract(freqs=freqs,
                                                                           preapply_gains=preapply_gains,
                                                                           vis_data=vis_data,
                                                                           vis_coords=vis_coords)

        ravel_fn, unravel_fn = pytree_unravel(init_params)
        init_params_flat = ravel_fn(init_params)

        def objective_fn(params_flat: jax.Array):
            params = unravel_fn(params_flat)
            gains = params.gains_real + 1j * params.gains_imag
            return -log_prob_fn(gains=gains)

        solver = jaxopt.LBFGS(
            fun=objective_fn,
            maxiter=self.num_iterations,
            jit=True,
            unroll=False,
            use_gamma=True
        )

        # Unroll ourself
        def body_fn(carry, x):
            params_flat, state = carry
            params_flat, state = solver.update(params=params_flat, state=state)
            return (params_flat, state), state.value

        carry = (init_params_flat, solver.init_state(init_params=init_params_flat))

        (params_flat, final_state), results = lax.scan(body_fn, carry, xs=jnp.arange(self.num_iterations))

        params = unravel_fn(params_flat)

        residual = subtract_fn(gains=params.gains_real + 1j * params.gains_imag)
        return params, (results, final_state), residual
