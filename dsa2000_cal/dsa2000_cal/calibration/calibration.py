import time as time_mod
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Tuple, Literal

import astropy.units as au
import jax
import jaxopt
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import lax
from jax import numpy as jnp

from dsa2000_cal.calibration.bfgs import BFGS
from dsa2000_cal.calibration.levenburg_marquardt import LevenbergMarquardt
from dsa2000_cal.calibration.gain_prior_models import AbstractGainPriorModel, \
    ReplicatedGainProbabilisticModel
from dsa2000_cal.common.coord_utils import lmn_to_icrs
from dsa2000_cal.common.plot_utils import plot_antenna_gains
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.forward_model.sky_model import SkyModel
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords, VisibilityData, MeasurementSet
from dsa2000_cal.simulation.rime_model import RIMEModel
from dsa2000_cal.types import CalibrationSolutions

tfpd = tfp.distributions


class CalibrationData(NamedTuple):
    visibility_coords: VisibilityCoords
    image: jnp.ndarray  # [source, chan, 2, 2]
    lmn: jnp.ndarray  # [source, 3]
    freqs: jnp.ndarray  # [chan]
    obs_vis: jnp.ndarray  # [row, chan, 2, 2]
    obs_vis_weight: jnp.ndarray  # [row, chan, 2, 2]


@dataclass(eq=False)
class Calibration:
    """
    Calibration main class. It loads in a measurement set, and calibrates it block by block, with a single gain
    solution per block. Calibration is only performed once per cadence block using the last solution for block in
    between updates. The solution_interval determines how much data is used to compute the solution, and over this time
    interval a single gain is solved for. The validity_interval determines how long the solution is valid for, meaning
    that solutions are forward applied until the solutions are older than the validity_interval.

    Args:
        sky_model: the sky model to calibrate based on. Each model gets a gain direction in the flux weighted direction.
        preapply_gain_model: the gain model to preapply before calibration.
        num_iterations: the number of iterations to run the solver for.
        inplace_subtract: if True, the calibration is performed in place, otherwise a new measurement set is created.
        plot_folder: the folder to save plots to.
        validity_interval: the interval over which the solution is valid.
        solution_interval: the interval over which the solution is computed.
        residual_ms_folder: the folder to save residuals to.
        seed: the random seed.
        convention: the calibration convention.
        dtype: the dtype to use.
        verbose: if True, print verbose output.
        num_shards: the number of shards to use.
    """
    # models to calibrate based on. Each model gets a gain direction in the flux weighted direction.
    sky_model: SkyModel
    rime_model: RIMEModel
    gain_prior_model: AbstractGainPriorModel

    preapply_gain_model: GainModel | None

    # Calibration parameters
    num_iterations: int
    inplace_subtract: bool
    plot_folder: str
    validity_interval: au.Quantity | None = None
    solution_interval: au.Quantity | None = None
    residual_ms_folder: str | None = None
    seed: int = 42
    verbose: bool = False
    num_shards: int = 1

    solver: Literal['BFGS', 'LM', 'LBFGS'] = 'BFGS'

    def __post_init__(self):
        if self.validity_interval is not None and not self.validity_interval.unit.is_equivalent(au.s):
            raise ValueError("solution_cadence must be in seconds.")
        if self.solution_interval is not None and not self.solution_interval.unit.is_equivalent(au.s):
            raise ValueError("average_interval must be in seconds.")

    @property
    def key(self):
        return jax.random.PRNGKey(self.seed)

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
                                                devices=jax.local_devices()[:self.num_shards])
        mesh = Mesh(devices, axis_names=('chan',))

        def tree_device_put(tree, sharding):
            return jax.tree_map(lambda x: jax.device_put(x, sharding), tree)

        # We perform averaging, by holding the gains constant for a period of time (average_interval)
        # 0, 1 --> integration_time = 1s, average_interval = 2.2s --> num_blocks = 2
        # 0, 1 --> integration_time = 1s, average_interval = 0.5s --> num_blocks = 1
        if self.solution_interval is None:
            num_blocks = 1
            cadence_interval = 1
        else:
            num_blocks = int(self.solution_interval / ms.meta.integration_time)
            if num_blocks <= 0:
                raise ValueError(
                    f"Solution interval {self.solution_interval} "
                    f"must be at least integration time {ms.meta.integration_time}"
                )
            rounded_solution_interval = num_blocks * ms.meta.integration_time
            print(f"Rounded solution interval: {rounded_solution_interval}")
            if self.validity_interval is None:
                cadence_interval = 1
            else:
                cadence_interval = int(self.validity_interval / rounded_solution_interval)
                if cadence_interval <= 0:
                    raise ValueError(
                        f"The validity interval {self.validity_interval} "
                        f"must be at leas the rounded solution interval {rounded_solution_interval}."
                    )

        print(f"Running with solution interval {self.solution_interval} / {num_blocks} blocks.")
        print(f"Running with validity interval {self.validity_interval} / {cadence_interval} blocks")

        # Ensure the freqs are the same in the models
        for wsclean_source_model in self.sky_model.component_models:
            if not np.allclose(ms.meta.freqs.to('Hz'), wsclean_source_model.freqs.to('Hz')):
                raise ValueError("Frequencies in the measurement set and source models must match.")
        for fits_source_model in self.sky_model.fits_models:
            if not np.allclose(ms.meta.freqs.to('Hz'), fits_source_model.freqs.to('Hz')):
                raise ValueError("Frequencies in the measurement set and source models must match.")
        if not self.inplace_subtract:
            if self.residual_ms_folder is None:
                raise ValueError("If not inplace subtracting, residual_ms_folder must be provided.")
            ms = ms.clone(ms_folder=self.residual_ms_folder)
            print("Created a new measurement set for residuals.")

        calibrator_lmn = au.Quantity(
            np.stack(
                [
                    model.flux_weighted_lmn() for model in self.sky_model.component_models
                ] + [
                    model.flux_weighted_lmn() for model in self.sky_model.fits_models
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
        init_params = None
        cadence_idx = 0
        apply_idx = 0
        key = self.key
        solve_durations = []
        while True:
            t0_inner = time_mod.time()
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
                    reps=(self.sky_model.num_sources, num_blocks, len(ms.meta.antennas), len(ms.meta.freqs), 1, 1)
                )

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
                init_params=tree_device_put(
                    data_dict['init_params'],
                    NamedSharding(mesh, P())  # Single gain for all
                ) if init_params is not None else None,
                vis_data=tree_device_put(data_dict['vis_data'], NamedSharding(mesh, P(None, 'chan'))),
                vis_coords=tree_device_put(data_dict['vis_coords'], NamedSharding(mesh, P()))
            )

            if apply_idx % cadence_interval == 0:
                num_iterations = self.num_iterations
            else:
                num_iterations = 0

            key, solve_key = jax.random.split(key)
            solution, params, (neg_log_likelihood, final_state), residual = self._solve_jax(
                key=solve_key,
                **data_dict,
                num_iterations=num_iterations
            )
            gen_response = VisibilityData(
                vis=np.asarray(residual)
            )
            # Replicate the solutions along time dimension
            solution = np.tile(solution[:, None, :, :, :, :],
                               (1, num_blocks, 1, 1, 1, 1))  # [num_cal, num_blocks, num_ant, num_chan, 2, 2]
            solutions.append(solution)
            residual_sum += np.sum(residual)

            # Pass forward
            init_params = params

            if apply_idx % cadence_interval == 0:
                # Plot results
                fig, axs = plt.subplots(1, 1, figsize=(6, 6), squeeze=False)
                axs[0][0].plot(neg_log_likelihood)
                axs[0][0].set_title(fr"Cadence window: {cadence_idx} $-\log \mathcal{{L}}$")
                axs[0][0].set_xlabel("Solver Iteration")
                axs[0][0].set_ylabel("Negative Log Likelihood")
                fig.tight_layout()
                fig.savefig(f"{self.plot_folder}/solver_progress_neg_log_likelihood_cadence_window_{cadence_idx}.png")
                plt.close(fig)
                cadence_idx += 1
            else:
                pass
            apply_idx += 1
            t1_inner = time_mod.time()
            solve_durations.append(t1_inner - t0_inner)

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
        for antenna_idx in range(0, len(ms.meta.antennas), len(ms.meta.antennas) // 20):
            fig = plot_antenna_gains(calibration_solutions, antenna_idx=antenna_idx, direction_idx=0)
            fig.savefig(f"{self.plot_folder}/antenna_{antenna_idx}_calibration_solutions.png")
            plt.close(fig)
        fig, axs = plt.subplots(1, 1, figsize=(6, 6), squeeze=False)
        axs[0][0].plot(solve_durations)
        axs[0][0].set_title("Solver Durations")
        axs[0][0].set_xlabel("Chunk Index")
        axs[0][0].set_ylabel("Duration (s)")
        fig.tight_layout()
        fig.savefig(f"{self.plot_folder}/solver_durations.png")
        plt.close(fig)
        print(f"Plots saved to {self.plot_folder}.")

        return ms

    @property
    def float_dtype(self):
        # Given self.dtype is complex, find float dtype
        return jnp.real(jnp.zeros((), dtype=self.rime_model.dtype)).dtype

    @partial(jax.jit, static_argnames=['self', 'num_iterations'])
    def _solve_jax(self, key, freqs: jax.Array, preapply_gains: jax.Array, init_params: jax.Array | None,
                   vis_data: VisibilityData,
                   vis_coords: VisibilityCoords,
                   num_iterations: int) -> Tuple[jax.Array,
    jax.Array, Tuple[jax.Array, jaxopt.OptStep], jax.Array]:

        gain_probabilistic_model = ReplicatedGainProbabilisticModel(
            rime_model=self.rime_model,
            gain_prior_model=self.gain_prior_model,
            preapply_gains=preapply_gains,
            freqs=freqs,
            vis_data=vis_data,
            vis_coords=vis_coords
        )

        if init_params is None:
            init_params = gain_probabilistic_model.get_init_params()
            # init_params = jnp.zeros((model.U_ndims,))  # [ndims_U]

        def objective_fn(params: jax.Array):
            return -gain_probabilistic_model.log_prob_joint(params)

        def residual_fn(params: jax.Array):
            vis_model, _ = gain_probabilistic_model.forward(params)
            vis_residuals = vis_data.vis - vis_model
            vis_residuals *= jnp.sqrt(vis_data.weights)
            vis_residuals = lax.reshape(vis_residuals, (np.size(vis_residuals),))
            return jnp.concatenate([jnp.real(vis_residuals), jnp.imag(vis_residuals)])

        if self.solver == 'LM':
            solver = LevenbergMarquardt(
                residual_fun=residual_fn,
                maxiter=self.num_iterations
            )
        elif self.solver == 'BFGS':
            solver = BFGS(
                fun=objective_fn,
                maxiter=self.num_iterations
            )
        else:
            raise ValueError(f"Solver {self.solver} not supported.")

        # Unroll ourself
        def body_fn(carry, x):
            params_flat, state = carry
            params_flat, state = solver.update(params=params_flat, state=state)
            return (params_flat, state), state.value

        carry = (init_params, solver.init_state(init_params=init_params))

        (params, final_state), results = lax.scan(body_fn, carry, xs=jnp.arange(num_iterations))

        vis_model, gains = gain_probabilistic_model.forward(params)
        vis_residuals = vis_data.vis - vis_model

        return gains, params, (results, final_state), vis_residuals
