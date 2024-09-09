import os
import time as time_mod
from dataclasses import dataclass
from functools import partial
from typing import Tuple, List, Any

import astropy.units as au
import jax
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import lax
from jax import numpy as jnp

from dsa2000_cal.calibration.multi_step_lm import MultiStepLevenbergMarquardt, MultiStepLevenbergMarquardtState, \
    MultiStepLevenbergMarquardtDiagnostic
from dsa2000_cal.calibration.probabilistic_models.probabilistic_model import AbstractProbabilisticModel, \
    ProbabilisticModelInstance
from dsa2000_cal.common.jax_utils import create_mesh, tree_device_put, block_until_ready, multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.types import mp_policy
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.measurement_sets.measurement_set import VisibilityData, MeasurementSet
from dsa2000_cal.types import CalibrationSolutions

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
    devices: List[jax.Device] | None = None

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

        mesh = create_mesh((1,), ('chan',))

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
        solve_durations = []
        residual_mae = []
        residual_rmse = []
        t0 = time_mod.time()

        # Inputs
        freqs_jax = quantity_to_jnp(ms.meta.freqs)

        # TODO: Apply UV cutoff to ignore galactic plane
        gen = ms.create_block_generator(
            vis=True, weights=True, flags=True, relative_time_idx=True,
            num_blocks=num_blocks, corrs=[['XX', 'XY'], ['YX', 'YY']]
        )
        gen_response = None
        last_state: MultiStepLevenbergMarquardtState | None = None
        cadence_idx = 0
        key = self.key

        while True:

            # Only solve every cadence_interval
            do_solve = cadence_idx % cadence_interval == 0
            if do_solve:
                print(f"Performing solve for cadence window {cadence_idx}.")
                num_iterations = self.num_iterations
            else:
                print(f"Skipping solve for cadence window {cadence_idx}.")
                num_iterations = 0

            key, solve_key = jax.random.split(key)

            # Get `num_blocks` time integrations of data
            try:
                times, visibility_coords, vis_data = gen.send(gen_response)
            except StopIteration:
                break

            t0_inner = time_mod.time()

            # Prepare distributed data
            times_jax = ms.time_to_jnp(times)  # [num_time]
            vis_data = vis_data._replace(
                vis=mp_policy.cast_to_vis(vis_data.vis),
                weights=mp_policy.cast_to_weight(vis_data.weights),
                flags=mp_policy.cast_to_flag(vis_data.flags)
            )  # [num_row, num_chan[2,2]]
            visibility_coords = visibility_coords._replace(
                uvw=mp_policy.cast_to_length(visibility_coords.uvw),
                time_obs=mp_policy.cast_to_time(visibility_coords.time_obs),
                antenna_1=mp_policy.cast_to_index(visibility_coords.antenna_1),
                antenna_2=mp_policy.cast_to_index(visibility_coords.antenna_2),
                time_idx=mp_policy.cast_to_index(visibility_coords.time_idx)
            )  # [num_row, ...]

            gain_solutions, residual, state, diagnostics = block_until_ready(
                self._solve_jax(
                    key=solve_key,
                    freqs=tree_device_put(freqs_jax, mesh, ('chan',)),
                    times=tree_device_put(times_jax, mesh, ()),
                    init_state=last_state,  # already shard as prior output
                    vis_data=tree_device_put(vis_data, mesh, (None, 'chan')),
                    vis_coords=tree_device_put(visibility_coords, mesh, ()),
                    num_iterations=num_iterations
                )
            )

            cadence_idx += 1
            # Update metrics
            t1_inner = time_mod.time()
            solve_durations.append(t1_inner - t0_inner)
            residual_mae.append(np.mean(np.abs(residual)))
            residual_rmse.append(np.sqrt(np.mean(np.abs(residual) ** 2)))

            # Store subtracted data
            # TODO: store uncertainty estimate in weights.
            gen_response = VisibilityData(
                vis=np.asarray(residual)
            )

            # Pass forward
            last_state = state

            if do_solve:
                print(
                    f"Solved {num_iterations} for cadence window {cadence_idx} in {solve_durations[-1]} seconds "
                    f"({solve_durations[-1] / num_iterations} s/iter)."
                )

                # Plot results
                fig, axs = plt.subplots(2, 1, figsize=(6, 6), squeeze=False, sharex=True)
                axs[0][0].plot(diagnostics.F_norm)
                axs[0][0].set_title(fr"Cadence window: {cadence_idx} residual norm")
                axs[0][0].set_xlabel("Solver Iteration")
                axs[0][0].set_ylabel(r"$|F(x)|^2$")
                axs[1][0].plot(diagnostics.delta_norm)
                axs[1][0].set_title(fr"Cadence window: {cadence_idx} delta norm")
                axs[1][0].set_xlabel("Solver Iteration")
                axs[1][0].set_ylabel(r"$\Delta x$")
                fig.tight_layout()
                fig.savefig(f"{self.plot_folder}/solver_diagnostics_{cadence_idx}.png")
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

        print(f"Completed calibration in {t1 - t0} seconds.")
        print(f"Residuals stored in {ms}. Solutions in {self.solution_folder}")

        fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True, squeeze=False)
        # Plot durations
        axs[0][0].plot(solve_durations)
        axs[0][0].set_title("Solver Durations")
        axs[0][0].set_xlabel("Chunk Index")
        axs[0][0].set_ylabel("Duration (s)")
        # Plot final MAE + rmse error bars
        axs[1][0].plot(residual_mae)
        axs[1][0].errorbar(
            range(len(residual_rmse)),
            residual_mae,
            yerr=np.std(residual_rmse),
            fmt='o'
        )
        axs[1][0].set_title("Residual MAE per iteration")
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
                   init_state: MultiStepLevenbergMarquardtState | None,
                   vis_data: VisibilityData,
                   vis_coords: VisibilityCoords,
                   num_iterations: int) -> Tuple[
        List[jax.Array], jax.Array, MultiStepLevenbergMarquardtState, MultiStepLevenbergMarquardtDiagnostic
    ]:
        """
        Solve for the gains using the probabilistic model.

        Args:
            key: an optional random key to use if needed
            freqs: [num_chan] the frequencies
            times: [num_time] the times
            init_state: the initial state of the solver
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

        # Construct the probabilistic model instance for calibration
        # We average down the data to a single time integration and channel for this
        freqs_cal, times_cal, vis_coords_cal, vis_data_cal = self.compute_average_solint_data(
            freqs=freqs,
            times=times,
            vis_coords=vis_coords,
            vis_data=vis_data
        )

        probabilistic_model_instance = self.create_probabilistic_model_instance(
            freqs=freqs_cal,
            times=times_cal,
            vis_coords=vis_coords_cal,
            vis_data=vis_data_cal
        )

        def residual_fn(params: List[Any]) -> Any:
            vis_model, _ = probabilistic_model_instance.forward(params)  # [num_row, num_chan[, 4]]
            vis_residuals = vis_data.vis - vis_model
            if vis_data.weights is not None:
                vis_residuals *= jnp.sqrt(vis_data.weights)
            if vis_data.flags is not None:
                vis_residuals = jnp.where(vis_data.flags, 0., vis_residuals)
            return vis_residuals

        solver = MultiStepLevenbergMarquardt(
            residual_fn=residual_fn,
            num_iterations=num_iterations,
            num_approx_steps=5
        )

        if init_state is None:
            init_params = probabilistic_model_instance.get_init_params()
            state = solver.create_initial_state(x0=init_params)
        else:
            state = solver.update_initial_state(init_state)

        state, diagnostics = solver.solve(state)

        # Predict at full resolution.
        @partial(
            multi_vmap,
            in_mapping="[c],[t],[t,r],[t,r,c]",
            out_mapping="[t,...,c]",
            verbose=True
        )
        def compute_residuals(freq_chunk: jax.Array,
                              time_chunk: jax.Array,
                              vis_coords_chunk: VisibilityCoords,
                              vis_data_chunk: VisibilityData):
            probabilistic_model_instance = self.create_probabilistic_model_instance(
                freqs=freq_chunk[None],
                times=time_chunk[None],
                vis_coords=vis_coords_chunk,
                vis_data=jax.tree.map(lambda x: x[:, None, ...], vis_data_chunk)  # Put channel back in
            )
            # Predict the model with the same parameters
            vis_model, _ = probabilistic_model_instance.forward(state.x)  # [num_row, 1, 4]
            vis_residuals = vis_data_chunk.vis - vis_model[:, 0, ...]
            return vis_residuals

        _, gains = probabilistic_model_instance.forward(state.x)

        reshape_vis_coords, reshaped_vis_data = Calibration.reshape_solint_data(
            times=times, vis_coords=vis_coords, vis_data=vis_data
        )
        vis_residuals = compute_residuals(freqs, times, reshape_vis_coords, reshaped_vis_data)
        # Stack again
        vis_residuals = lax.reshape(vis_residuals, vis_data.vis.shape)  # [num_row, num_chan, 4]

        return gains, vis_residuals, state, diagnostics

    def create_probabilistic_model_instance(self, freqs: jax.Array, times: jax.Array, vis_coords: VisibilityCoords,
                                            vis_data: VisibilityData) -> ProbabilisticModelInstance:
        """
        Create a probabilistic model instance.

        Args:
            freqs: [num_chan] the frequencies
            times: [num_time] the times
            vis_coords: [num_row, ...] the visibility coordinates
            vis_data: [num_row, num_chan[, 4]] the visibility data

        Returns:
            the probabilistic model instance
        """
        probabilistic_model_instances = [
            probabilistic_model.create_model_instance(
                freqs=freqs,
                times=times,
                vis_data=vis_data,
                vis_coords=vis_coords
            ) for probabilistic_model in self.probabilistic_models
        ]
        probabilistic_model_instance = probabilistic_model_instances[0]
        for other_model in probabilistic_model_instances[1:]:
            probabilistic_model_instance = probabilistic_model_instance + other_model
        return probabilistic_model_instance

    @staticmethod
    def compute_average_solint_data(
            freqs: jax.Array, times: jax.Array, vis_coords: VisibilityCoords, vis_data: VisibilityData
    ) -> Tuple[jax.Array, jax.Array, VisibilityCoords, VisibilityData]:
        """
        Compute the average data over the solution interval.

        Args:
            freqs: [num_chan] the frequencies
            times: [num_time] the times
            vis_coords: [num_row, ...] the visibility coordinates
            vis_data: [num_row, num_chan[, 4]] the visibility data

        Returns:
            freqs_cal: [1] the averaged frequencies
            times_cal: [1] the averaged times
            vis_coords_cal: [num_row // num_time, ...] the averaged visibility coordinates
            vis_data_cal: [num_row // num_time, num_chan[, 4]] the averaged visibility data
        """
        reshape_vis_coords, reshaped_vis_data = Calibration.reshape_solint_data(
            times=times, vis_coords=vis_coords, vis_data=vis_data
        )

        vis_cal = jnp.mean(reshaped_vis_data.vis, axis=(0, 2), keepdims=True)[0]  # [num_row // num_time, 1, [, 4]]
        # Weights are 1/var(data) so we average the reciprocal of the weights, and then take the reciprocal.
        weights_cal = jnp.reciprocal(
            jnp.mean(jnp.reciprocal(reshaped_vis_data.weights), axis=(0, 2), keepdims=True)[0]
        )  # [num_row // num_time, 1, [, 4]]
        # TODO: any/all ambiguity with flags.
        flags_cal = jnp.any(reshaped_vis_data.flags, axis=(0, 2), keepdims=True)[0]  # [num_row // num_time, 1, [, 4]]
        vis_data_cal = VisibilityData(
            vis=vis_cal,
            weights=weights_cal,
            flags=flags_cal
        )

        # Average down the coordinates too
        uvw_cal = jnp.mean(reshape_vis_coords.uvw, axis=0)  # [num_row // num_time, 3]
        time_obs_cal = jnp.mean(reshape_vis_coords.time_obs, axis=0)  # [num_row // num_time]
        antenna_1_cal = reshape_vis_coords.antenna_1[0, :]  # [num_row // num_time]
        antenna_2_cal = reshape_vis_coords.antenna_2[0, :]  # [num_row // num_time]
        time_idx_cal = reshape_vis_coords.time_idx[0, :]  # [num_row // num_time]
        vis_coords_cal = VisibilityCoords(
            uvw=uvw_cal,
            time_obs=time_obs_cal,
            antenna_1=antenna_1_cal,
            antenna_2=antenna_2_cal,
            time_idx=time_idx_cal
        )
        freqs_cal = jnp.mean(freqs, keepdims=True)  # [1]
        times_cal = jnp.mean(times, keepdims=True)  # [1]
        return freqs_cal, times_cal, vis_coords_cal, vis_data_cal

    @staticmethod
    def reshape_solint_data(
            times: jax.Array, vis_coords: VisibilityCoords, vis_data: VisibilityData
    ) -> Tuple[VisibilityCoords, VisibilityData]:
        """
        Reshape the data to unstacked solint data.

        Args:
            times: [num_time] the times
            vis_coords: [num_row, ...] the visibility coordinates
            vis_data: [num_row, num_chan[, 4]] the visibility data

        Returns:
            vis_coords_cal: [num_row // num_time, ...] the averaged visibility coordinates
            vis_data_cal: [num_row // num_time, num_chan[, 4]] the averaged visibility data
        """
        num_row = np.shape(vis_data.vis)[0]
        num_time = len(times)

        def _reshape_vis_data(x):
            return lax.reshape(
                x, (num_time, num_row // num_time) + x.shape[1:]
            )  # [num_time, num_row // num_time, num_chan, [ 4]]

        reshaped_vis_data = jax.tree_map(_reshape_vis_data, vis_data)

        def _reshape_vis_coords(x):
            return lax.reshape(
                x, (num_time, num_row // num_time) + x.shape[1:]
            )  # [num_time, num_row // num_time, ...]

        reshape_vis_coords = jax.tree_map(_reshape_vis_coords, vis_coords)

        return reshape_vis_coords, reshaped_vis_data
