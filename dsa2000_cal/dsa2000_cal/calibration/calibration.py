import os
import time as time_mod
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Literal, List, Any

import astropy.units as au
import jax
import jaxopt
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import lax
from jax import numpy as jnp

from dsa2000_cal.calibration.bfgs import BFGS
from dsa2000_cal.calibration.lbfgs import LBFGS
from dsa2000_cal.calibration.levenburg_marquardt import LevenbergMarquardt
from dsa2000_cal.calibration.probabilistic_models.probabilistic_model import AbstractProbabilisticModel
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.measurement_sets.measurement_set import VisibilityData, MeasurementSet
from dsa2000_cal.types import CalibrationSolutions
from dsa2000_cal.uvw.far_field import VisibilityCoords

tfpd = tfp.distributions


@dataclass(eq=False)
class Calibration:
    """
    Calibration main class. It loads in a measurement set, and calibrates it block by block, with a single gain
    solution per block. Calibration is only performed once per cadence block using the last solution for block in
    between updates. The solution_interval determines how much data is used to compute the solution, and over this time
    interval a single gain is solved for. The validity_interval determines how long the solution is valid for, meaning
    that solutions are forward applied until the solutions are older than the validity_interval.

    Args:
        probabilistic_models: the probabilistic models to calibrate based on.
        num_iterations: the number of iterations to run the solver for.
        inplace_subtract: if True, the calibration is performed in place, otherwise a new measurement set is created.
        plot_folder: the folder to save plots to.
        solution_folder: the folder to save solutions to.
        validity_interval: the interval over which the solution is valid, must be a multiple of solution_interval.
        solution_interval: the interval over which the solution is computed, does not enforce any gain constraints,
            e.g. constant over time, rather controls how much memory to use.
        residual_ms_folder: the folder to save residuals to.
        seed: the random seed.
        verbose: if True, print verbose output.
        num_shards: the number of shards to use.
    """
    # models to calibrate based on. Each model gets a gain direction in the flux weighted direction.
    probabilistic_models: List[AbstractProbabilisticModel]

    # Calibration parameters
    num_iterations: int
    inplace_subtract: bool
    plot_folder: str
    solution_folder: str
    validity_interval: au.Quantity | None = None
    solution_interval: au.Quantity | None = None
    residual_ms_folder: str | None = None
    seed: int = 42
    verbose: bool = False
    num_shards: int = 1
    devices: List[jax.Device] | None = None

    solver: Literal['BFGS', 'LM', 'LBFGS'] = 'BFGS'

    def __post_init__(self):
        # Create folders
        os.makedirs(self.plot_folder, exist_ok=True)
        os.makedirs(self.solution_folder, exist_ok=True)

        if self.validity_interval is not None and not self.validity_interval.unit.is_equivalent(au.s):
            raise ValueError("solution_cadence must be in seconds.")
        if self.solution_interval is not None and not self.solution_interval.unit.is_equivalent(au.s):
            raise ValueError("average_interval must be in seconds.")
        if self.devices is None:
            self.devices = jax.local_devices()

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

        # Ensure number of freqs is a multiple of the number of shards
        if len(ms.meta.freqs) % self.num_shards != 0:
            raise ValueError(
                f"Number of freqs {len(ms.meta.freqs)} is not a multiple of the number of shards {self.num_shards}"
            )

        if len(jax.devices()) < self.num_shards:
            raise ValueError(
                f"Number of devices {len(jax.devices())} is less than the number of shards {self.num_shards}"
            )

        devices = mesh_utils.create_device_mesh(
            mesh_shape=(self.num_shards,),
            devices=self.devices[:self.num_shards]
        )
        mesh = Mesh(devices, axis_names=('chan',))

        def tree_device_put(tree, sharding):
            return jax.tree_map(lambda x: jax.device_put(x, sharding), tree)

        # Determine how many blocks to process at once, and how many blocks to apply the solution over.
        solution_interval = self.solution_interval
        if solution_interval is None:
            solution_interval = ms.meta.integration_time
        if solution_interval < ms.meta.integration_time:
            raise ValueError(
                f"Solution interval {solution_interval} must be at least integration time {ms.meta.integration_time}"
            )

        num_blocks = (solution_interval / ms.meta.integration_time).value
        if not num_blocks.is_integer():
            raise ValueError(
                f"Solution interval {solution_interval} must be a multiple of integration time {ms.meta.integration_time}"
            )
        num_blocks = int(num_blocks)
        actual_solution_interval = num_blocks * ms.meta.integration_time
        if actual_solution_interval != solution_interval:
            print(f"Rounded solution interval to {actual_solution_interval}")
            solution_interval = actual_solution_interval

        validity_interval = self.validity_interval
        if validity_interval is None:
            validity_interval = solution_interval

        # Esure that the validity interval is a multiple of the solution interval
        if not (validity_interval / solution_interval).value.is_integer():
            raise ValueError(
                f"Validity interval {validity_interval} must be a multiple of solution interval {solution_interval}"
            )

        cadence_interval = int(validity_interval / solution_interval)

        print(
            f"Running with:"
            f"\n\tsolution interval {solution_interval} ({num_blocks} integrations)"
            f"\n\tvalidity interval {validity_interval} ({cadence_interval * num_blocks} integrations)"
        )

        if not self.inplace_subtract:
            if self.residual_ms_folder is None:
                raise ValueError("If not inplace subtracting, residual_ms_folder must be provided.")
            ms = ms.clone(ms_folder=self.residual_ms_folder)
            print("Created a new measurement set for residuals.")

        # Metrics
        residual_sum = 0.
        solve_durations = []
        residual_mae = []
        t0 = time_mod.time()

        # Inputs
        freqs_jax = quantity_to_jnp(ms.meta.freqs)

        solutions = []
        # TODO: Apply UV cutoff to ignore galactic plane
        gen = ms.create_block_generator(
            vis=True, weights=True, flags=True, relative_time_idx=True,
            num_blocks=num_blocks
        )
        gen_response = None
        last_params: Any | None = None
        cadence_idx = 0
        key = self.key

        while True:

            # Only solve every cadence_interval
            if cadence_idx % cadence_interval == 0:
                print(f"Performing solve for cadence window {cadence_idx}.")
                num_iterations = self.num_iterations
            else:
                print(f"Skipping solve for cadence window {cadence_idx}.")
                num_iterations = 0
            key, solve_key = jax.random.split(key)

            t0_inner = time_mod.time()
            # Get `num_blocks` time integrations of data
            try:
                times, visibility_coords, data = gen.send(gen_response)
            except StopIteration:
                break

            # Shard data over mesh by channel
            times_jax = jnp.asarray((times.tt - ms.ref_time.tt).sec)  # [num_time]
            data_dict = dict(
                freqs=tree_device_put(freqs_jax, NamedSharding(mesh, P('chan'))),
                times=tree_device_put(times_jax, NamedSharding(mesh, P())),
                init_params=last_params,  # already shard as prior output
                vis_data=tree_device_put(
                    jax.tree_map(jnp.asarray, data),
                    NamedSharding(mesh, P(None, 'chan'))
                ),
                vis_coords=tree_device_put(
                    jax.tree_map(jnp.asarray, visibility_coords),
                    NamedSharding(mesh, P())
                )
            )

            gain_solutions, params, neg_log_likelihood, final_state, residual = self._solve_jax(
                key=solve_key,
                **data_dict,
                num_iterations=num_iterations
            )
            neg_log_likelihood.block_until_ready()

            cadence_idx += 1
            # Update metrics
            t1_inner = time_mod.time()
            solve_durations.append(t1_inner - t0_inner)
            residual_sum += np.sum(residual)
            residual_mae.append(np.mean(np.abs(residual)))

            # Store subtracted data
            # TODO: store uncertainty estimate in weights.
            gen_response = VisibilityData(
                vis=np.asarray(residual)
            )

            # Pass forward
            last_params = params

            # Print shard layout
            print("Last params shard layout:")
            for param in last_params:
                jax.debug.visualize_array_sharding(param)

            if cadence_idx % cadence_interval == 0:
                print(
                    f"Solved {num_iterations} for cadence window {cadence_idx} in {solve_durations[-1]} seconds "
                    f"({solve_durations[-1] / num_iterations} s/iter)."
                )

                # Plot results
                fig, axs = plt.subplots(1, 1, figsize=(6, 6), squeeze=False)
                axs[0][0].plot(neg_log_likelihood)
                axs[0][0].set_title(fr"Cadence window: {cadence_idx} $-\log \mathcal{{L}}$")
                axs[0][0].set_xlabel("Solver Iteration")
                axs[0][0].set_ylabel("Negative Log Likelihood")
                fig.tight_layout()
                fig.savefig(f"{self.plot_folder}/solver_progress_neg_log_likelihood_cadence_window_{cadence_idx}.png")
                plt.close(fig)

                # Save gains to files
                for model_idx, gain_solution in enumerate(gain_solutions):
                    # Save to file
                    solution = CalibrationSolutions(
                        gains=np.asarray(gain_solution),
                        times=times,
                        antennas=ms.meta.antennas,
                        antenna_labels=ms.meta.antenna_names,
                        freqs=ms.meta.freqs,
                        pointings=ms.meta.pointings
                    )
                    file_name = os.path.join(
                        self.solution_folder, f"calibration_solution_m{model_idx:03d}_c{cadence_idx:03d}.json"
                    )
                    with open(file_name, "w") as fp:
                        fp.write(solution.json(indent=2))
        # Measure total solve time.
        t1 = time_mod.time()

        print(f"Completed calibration in {t1 - t0} seconds, with residual sum {residual_sum}.")
        print(f"Residuals stored in {ms}. Solutions in {self.solution_folder}")

        fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True, squeeze=False)
        # Plot durations
        axs[0][0].plot(solve_durations)
        axs[0][0].set_title("Solver Durations")
        axs[0][0].set_xlabel("Chunk Index")
        axs[0][0].set_ylabel("Duration (s)")
        # Plot final MAE
        axs[1][0].plot(residual_mae)
        axs[1][0].set_title("Residual MAE")
        axs[1][0].set_xlabel("Chunk Index")
        axs[1][0].set_ylabel("MAE")

        fig.tight_layout()
        fig.savefig(f"{self.plot_folder}/solver_durations.png")
        plt.close(fig)
        print(f"Plots saved to {self.plot_folder}.")

        return ms

    @partial(jax.jit, static_argnames=['self', 'num_iterations'])
    def _solve_jax(self, key,
                   freqs: jax.Array,
                   times: jax.Array,
                   init_params: jax.Array | None,
                   vis_data: VisibilityData,
                   vis_coords: VisibilityCoords,
                   num_iterations: int) -> Tuple[
        List[jax.Array], Any, jax.Array, jaxopt.OptStep, jax.Array
    ]:
        """
        Solve for the gains using the probabilistic model.

        Args:
            key: an optional random key to use if needed
            freqs: [num_chan] the frequencies
            times: [num_time] the times
            init_params: [...] the initial parameters for the gains or None
            vis_data: [num_row, num_chan[, 4]] the visibility data
            vis_coords: [num_row, ...] the visibility coordinates
            num_iterations: the number of iterations to run the solver for

        Returns:
            gains: list of [num_cal, num_time, num_ant, num_chan[, 2, 2]] the gains
            params: [...] the parameters
            results: [num_iterations, ...] the results of the solver
            final_state: the final state of the solver
            vis_residuals: [num_row, num_chan, 4] the residuals
        """

        probabilistic_model_instances = [
            probabilistic_model.create_model_instance(
                freqs=freqs,
                times=times,
                vis_data=vis_data,
                vis_coords=vis_coords
            ) for probabilistic_model in self.probabilistic_models
        ]

        # Add together the probabilistic model instances into one
        probabilistic_model_instance = probabilistic_model_instances[0]
        for other_model in probabilistic_model_instances[1:]:
            probabilistic_model_instance = probabilistic_model_instance + other_model

        if init_params is None:
            init_params = probabilistic_model_instance.get_init_params()

        def objective_fn(params: List[jax.Array]):
            return -probabilistic_model_instance.log_prob_joint(params)

        def residual_fn(params: List[jax.Array]):
            vis_model, _ = probabilistic_model_instance.forward(params)  # [num_row, num_chan, 4]
            vis_residuals = vis_data.vis - vis_model
            if vis_data.weights is not None:
                vis_residuals *= jnp.sqrt(vis_data.weights)
            if vis_data.flags is not None:
                vis_residuals = jnp.where(vis_data.flags, 0., vis_residuals)
            vis_residuals = lax.reshape(vis_residuals, (np.size(vis_residuals),))
            return jnp.concatenate([jnp.real(vis_residuals), jnp.imag(vis_residuals)], axis=0)

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
        elif self.solver == 'LBFGS':
            solver = LBFGS(
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

        if num_iterations > 0:
            (params, final_state), results = lax.scan(body_fn, carry, xs=jnp.arange(num_iterations))
        else:
            (params, final_state) = carry
            results = jnp.asarray([])

        vis_model, gains = probabilistic_model_instance.forward(params)

        # Subtract the model from the data
        vis_residuals = vis_data.vis - vis_model

        return gains, params, results, final_state, vis_residuals