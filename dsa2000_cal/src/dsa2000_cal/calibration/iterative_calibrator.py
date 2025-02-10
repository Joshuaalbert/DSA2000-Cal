import dataclasses
import os
import time
from functools import partial
from typing import Generator, Tuple, List, Any
from typing import NamedTuple

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import jaxns.framework.context as ctx
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map
from jaxns.framework.ops import simulate_prior_model

from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import AbstractGainPriorModel, GainPriorModel
from dsa2000_cal.calibration.solvers.multi_step_lm import MultiStepLevenbergMarquardtState, \
    MultiStepLevenbergMarquardtDiagnostic, MultiStepLevenbergMarquardt
from dsa2000_cal.common.array_types import ComplexArray, FloatArray, BoolArray, IntArray
from dsa2000_cal.common.corr_utils import broadcast_translate_corrs
from dsa2000_cal.common.jax_utils import simple_broadcast, create_mesh
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, time_to_jnp, jnp_to_time
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_fm.forward_models.streaming.distributed.average_utils import average_rule

tfpd = tfp.distributions


@dataclasses.dataclass
class TimerLog:
    msg: str

    def __post_init__(self):
        self.t0 = time.time()

    def __enter__(self):
        print(f"{self.msg}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"... took {time.time() - self.t0:.3f} seconds")
        return False


class Data(NamedTuple):
    """
    Data structure that contains the data that is used in the calibration algorithm.
    D - number of bright sources
    E - number of background sources
    T - number of times at full resolution
    Tm - number of model times, require T = l * Tm for some l
    B - number of baselines
    C - number of channels at full resolution
    Cm - number of model freqs, require C = l * Cm for some l
    num_coh - number of coherence products
    """

    sol_int_time_idx: int
    coherencies: Tuple[str, ...]  # list of coherencies of length num_coh

    # Full resolution data
    vis_data: ComplexArray  # [T, B, C, num_coh]
    weights: FloatArray  # [T, B, C, num_coh]
    flags: BoolArray  # [T, B, C, num_coh]

    # Model
    vis_bright_sources: ComplexArray  # [D, Tm, B, Cm, num_coh]
    vis_background: ComplexArray  # [E, Tm, B, Cm, num_coh]
    model_freqs: au.Quantity  # [Cm]
    model_times: at.Time  # [Tm]
    ref_time: at.Time
    antenna1: IntArray  # [B]
    antenna2: IntArray  # [B]


class ReturnData(NamedTuple):
    vis_residuals: ComplexArray  # [T, B, C, num_coh]
    solver_state: Any


@dataclasses.dataclass(eq=False)
class CalibrationStep:
    """
    Performs a single step of calibration
    """
    gain_probabilistic_model: AbstractGainPriorModel
    full_stokes: bool
    num_ant: int
    verbose: bool = True

    def __call__(self,
                 vis_model: ComplexArray,
                 vis_data: ComplexArray,
                 weights: FloatArray,
                 flags: BoolArray,
                 model_freqs: FloatArray,
                 model_times: FloatArray,
                 antenna1: IntArray,
                 antenna2: IntArray,
                 state: MultiStepLevenbergMarquardtState | None = None
                 ) -> Tuple[ComplexArray, MultiStepLevenbergMarquardtState, MultiStepLevenbergMarquardtDiagnostic]:
        """
        Calibrate and subtract model visibilities from data visibilities.

        Args:
            vis_model: [D, Tm, B, Cm[,2,2]] the model visibilities per direction
            vis_data: [Ts, B, Cs[,2,2]] the data visibilities
            weights: [Ts, B, Cs[,2,2]] the weights
            flags: [Ts, B, Cs[,2,2]] the flags
            model_freqs: [Cm] the frequencies
            model_times: [Tm] the times
            antenna1: [B] the antenna1
            antenna2: [B] the antenna2
            state: MultiStepLevenbergMarquardtState the state of the solver (optional)

        Returns:
            gains: [D, Tm, A, Cm[,2,2]] the gains
            state: MultiStepLevenbergMarquardtState the state of the solver
            diagnostics: the diagnostics of the solver
        """

        # calibrate and subtract
        key = jax.random.PRNGKey(0)

        D, Tm, B, Cm = np.shape(vis_model)[:4]

        # Create gain prior model
        def get_gains():
            prior_model = self.gain_probabilistic_model.build_prior_model(
                num_source=D,
                num_ant=self.num_ant,
                freqs=model_freqs,
                times=model_times
            )
            (gains,), _ = simulate_prior_model(key, prior_model)  # [D, Tm, A, Cm[,2,2]]
            return gains  # [D, Tm, A, Cm[,2,2]]

        get_gains_transformed = ctx.transform(get_gains)

        compute_residuals_fn = self.build_compute_residuals_fn()

        # Create residual_fn
        def residual_fn(params: ComplexArray) -> ComplexArray:
            gains = get_gains_transformed.apply(params, key).fn_val
            return compute_residuals_fn(vis_model, vis_data, weights, flags, gains, antenna1, antenna2)

        solver = MultiStepLevenbergMarquardt(
            residual_fn=residual_fn,
            num_approx_steps=0,
            num_iterations=100,
            verbose=self.verbose,
            gtol=1e-4
        )

        # Get solver state
        if state is None:
            init_params = get_gains_transformed.init(key).params
            state = solver.create_initial_state(init_params)
        else:
            # TODO: EKF forward update on data
            state = solver.update_initial_state(state)
        state, diagnostics = solver.solve(state)

        gains = get_gains_transformed.apply(state.x, key).fn_val

        return gains, state, diagnostics

    def build_compute_residuals_fn(self):
        def compute_residuals_fn(
                vis_model: ComplexArray,
                vis_data: ComplexArray,
                weights: FloatArray,
                flags: BoolArray,
                gains: ComplexArray,
                antenna1: IntArray,
                antenna2: IntArray
        ):
            """
            Compute the residual between the model visibilities and the observed visibilities.

            Args:
                vis_model: [D, Tm, B, Cm[,2,2]] the model visibilities per direction
                vis_data: [Ts, B, Cs[,2,2]] the data visibilities
                weights: [Ts, B, Cs[,2,2]] the data weights
                flags: [Ts, B, Cs[,2,2]] the data flags
                gains: [D, Tm, A, Cm[,2, 2]] the gains
                antenna1: [B] the antenna1
                antenna2: [B] the antenna2

            Returns:
                residuals: [Ts, B, Cs[,2,2]]
            """

            if np.shape(weights) != np.shape(flags):
                raise ValueError(
                    f"Weights and flags must have the same shape, got {np.shape(weights)} and {np.shape(flags)}")
            if np.shape(vis_data) != np.shape(weights):
                raise ValueError(
                    f"Visibilities and weights must have the same shape, got {np.shape(vis_data)} and {np.shape(weights)}")

            residuals = compute_residual(
                vis_model=vis_model,
                vis_data=vis_data,
                gains=gains,
                antenna1=antenna1,
                antenna2=antenna2
            )
            weights *= jnp.logical_not(flags).astype(weights.dtype)  # [Tm, B, Cm[,2,2]]
            residuals *= jnp.sqrt(weights)  # [Tm, B, Cm[,2,2]]
            return residuals.real, residuals.imag

        return compute_residuals_fn


@dataclasses.dataclass(eq=False)
class IterativeCalibrator:
    """
    Performs streaming calibration and subtraction. A chunk of full resolution is shaped [T, B, C, num_coh].

    There are two further shapes corresponding to model and averaging rules:

    1. Model visibilties have shape [D, Tm, B, Cm, num_coh] where T % Tm == 0, and C % Cm == 0, and should be evaluated at
    averaged times, and frequencies rather than averaged down from full resolution. This is to prevent model smearing.

    2. Secondly, the full resolution data can be averaged to lower resolution [Ts, B, Cs, num_coh]. We require
    T % Ts == 0, Ts % Tm == 0 and C % Cs == 0, Cs % Cm == 0. Therefore, the model dimensions must divide both the full
    resolution data, and the averaged down resolution data. Piece-wise constant model interpolation is used. Without
    averaging we have Ts=T and Cs=C.

    Note: averaging decreasing SNR, as it introduces smearing. The SNR boost is associated with the reduced model
    dimensions, Tm and Cm. Averaging should be done if performance is limited by the data size (which may not be the
    case). It is thus suggested to start with Ts=T and Cs=T, and choose Tm and Cm that give suitable solutions, and
    then average data if necessary for performance.
    """
    plot_folder: str
    run_name: str
    gain_probabilistic_model: AbstractGainPriorModel
    full_stokes: bool
    antennas: ac.EarthLocation
    verbose: bool = False
    devices: List[jax.Device] | None = None

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)

    @staticmethod
    def create_simple_calibrator(full_stokes: bool, antennas: ac.EarthLocation, verbose: bool = True,
                                 **kwargs) -> 'IterativeCalibrator':
        gain_prior_model = GainPriorModel(
            gain_stddev=1.,
            dd_dof=1,
            di_dof=1,
            double_differential=True,
            dd_type='unconstrained',
            di_type='unconstrained',
            full_stokes=full_stokes
        )
        return IterativeCalibrator(
            gain_probabilistic_model=gain_prior_model,
            full_stokes=full_stokes,
            antennas=antennas,
            verbose=verbose,
            **kwargs
        )

    def build_compute_residual(self):
        @jax.jit
        def calc_residual(vis_model, vis_data, gains, antenna1, antenna2):
            """
            Do a step of calibration.

            Args:
                vis_model: [D, Tm, B, Cm[,2, 2]]
                vis_data: [T, B, C[,2, 2]]
                gains: [D, Tm, A, Cm[,2,2]] the gains
                antenna1: [B]
                antenna2: [B]


            Returns:
                residual visibilties: [T, B, C[,2, 2]]
            """
            # Performs application of gains to model, then tiles and subtracts from full resolution data.
            # Tm divides T, Cm divides C
            return compute_residual(vis_model, vis_data, gains, antenna1, antenna2)

        return calc_residual

    def build_calibration(self):

        def calibrate(vis_model, vis_data_avg, weights_avg, flags_avg, model_freqs, model_times, antenna1, antenna2,
                      solve_state: MultiStepLevenbergMarquardtState | None) -> Tuple[
            ComplexArray, MultiStepLevenbergMarquardtState, MultiStepLevenbergMarquardtDiagnostic]:
            """
            Do a step of calibration.

            Args:
                vis_model: [D, Tm, B, Cm[,2, 2]]
                vis_data_avg: [Ts, B, Cs[,2, 2]]
                weights_avg: [Ts, B, Cs[,2, 2]]
                flags_avg: [Ts, B, Cs[,2, 2]]
                model_freqs: [Cm]
                model_times: [Tm]
                solve_state: solver state, or None.

            Returns:
                gains: [D, Tm, A, Cm[,2,2]] the gains
                state: MultiStepLevenbergMarquardtState the state of the solver
                diagnostics: the diagnostics of the solver
            """
            calibration_step = CalibrationStep(
                gain_probabilistic_model=self.gain_probabilistic_model,
                full_stokes=self.full_stokes,
                num_ant=len(self.antennas),
                verbose=True
            )
            return calibration_step(
                vis_model=vis_model,
                vis_data=vis_data_avg,
                weights=weights_avg,
                flags=flags_avg,
                model_freqs=model_freqs,
                model_times=model_times,
                antenna1=antenna1,
                antenna2=antenna2,
                state=solve_state
            )

        if self.devices is not None:
            @jax.jit
            def sharded_calibrate(vis_model, vis_data_avg, weights_avg, flags_avg, model_freqs, model_times, antenna1,
                                  antenna2,
                                  solve_state: MultiStepLevenbergMarquardtState | None):
                B = np.shape(antenna1)[0]
                if B % len(self.devices) != 0:
                    # append some baselines with flag=True
                    extra = len(self.devices) - B % len(self.devices)

                    vis_model = jnp.concatenate([vis_model, vis_model[:, :, :extra]], axis=2)
                    vis_data_avg = jnp.concatenate([vis_data_avg, vis_data_avg[:, :extra]], axis=1)
                    weights_avg = jnp.concatenate([weights_avg, jnp.zeros_like(weights_avg[:, :extra])], axis=1)
                    flags_avg = jnp.concatenate([flags_avg, jnp.ones_like(flags_avg[:, :extra])], axis=1)
                    antenna1 = jnp.concatenate([antenna1, antenna1[:extra]], axis=0)
                    antenna2 = jnp.concatenate([antenna2, antenna2[:extra]], axis=0)

                mesh = create_mesh((len(self.devices),), ('B'), self.devices)
                return shard_map(
                    calibrate,
                    mesh=mesh,
                    in_specs=(
                        PartitionSpec(None, None, 'B'),  # vis_model
                        PartitionSpec(None, 'B'),  # vis_data_avg
                        PartitionSpec(None, 'B'),  # weights_avg
                        PartitionSpec(None, 'B'),  # flags_avg
                        PartitionSpec(),  # model_freqs
                        PartitionSpec(),  # model_times
                        PartitionSpec('B'),  # antenna1
                        PartitionSpec('B'),  # antenna2
                        PartitionSpec(),  # solve_state
                    ),
                    out_specs=(
                        PartitionSpec(),  # gains
                        PartitionSpec(),  # state
                        PartitionSpec(),  # diagnostics
                    ),
                    check_rep=False
                )(vis_model, vis_data_avg, weights_avg, flags_avg, model_freqs, model_times, antenna1, antenna2,
                  solve_state)

            return sharded_calibrate
        else:
            calibrate = jax.jit(calibrate)
            return calibrate

    def build_average_rule(self, num_model_times_per_sol_int: int | None, num_model_freqs_per_sol_int: int | None):

        @jax.jit
        def average(vis_data: ComplexArray, weights: FloatArray, flags: BoolArray) -> Tuple[
            ComplexArray, FloatArray, BoolArray]:
            # average data to match model: [Ts, B, Cs[, 2, 2]] -> [Tm, B, Cm[, 2, 2]]
            if num_model_times_per_sol_int is not None:
                time_average_rule = partial(
                    average_rule,
                    num_model_size=num_model_times_per_sol_int,
                    axis=0
                )
            else:
                time_average_rule = lambda x: x
            if num_model_freqs_per_sol_int is not None:
                freq_average_rule = partial(
                    average_rule,
                    num_model_size=num_model_freqs_per_sol_int,
                    axis=2
                )
            else:
                freq_average_rule = lambda x: x
            vis_data_avg = time_average_rule(freq_average_rule(vis_data))
            weights_avg = jnp.reciprocal(time_average_rule(freq_average_rule(jnp.reciprocal(weights))))
            flags_avg = freq_average_rule(time_average_rule(flags.astype(jnp.float16))).astype(jnp.bool_)
            return vis_data_avg, weights_avg, flags_avg

        return average

    def build_main_step(self, Ts: int | None = None, Cs: int | None = None):
        average = self.build_average_rule(
            num_model_times_per_sol_int=Ts,
            num_model_freqs_per_sol_int=Cs
        )
        calibrate = self.build_calibration()
        calc_residual = self.build_compute_residual()

        # Predict data and model

        def _step(data: Data, solver_state=None) -> ReturnData:
            nonlocal Ts, Cs
            vis_model = jnp.concatenate([data.vis_bright_sources, data.vis_background],
                                        axis=0)  # [S + E, Tm, B, Cm, num_coh]

            D, Tm, B, Cm, num_coh = np.shape(vis_model)
            T, B_, C, num_coh_ = np.shape(data.vis_data)
            if B != B_:
                raise ValueError(f"Model and data must have the same number of baselines, got {B} and {B_}")
            if num_coh != num_coh_:
                raise ValueError(
                    f"Model and data must have the same number of coherence products, got {num_coh} and {num_coh_}")

            if T % Tm != 0:
                raise ValueError(f"Model times must divide full resolution times, got {T} and {Tm}")

            if C % Cm != 0:
                raise ValueError(f"Model frequencies must divide full resolution frequencies, got {C} and {Cm}")

            # Print out the possible values for Ts and Cs such that Ts % Tm = 0 and Cs % Cm = 0 and T % Ts == 0 and C % Cs == 0
            print(f"Possible values of Ts={[i for i in range(Tm, T + 1) if T % i == 0 and i % Tm == 0]}")
            print(f"Possible values of Cs={[i for i in range(Cm, C + 1) if C % i == 0 and i % Cm == 0]}")

            if Ts is None:
                Ts = T
            if Cs is None:
                Cs = C
            if not (T % Ts == 0 and Ts % Tm == 0):
                raise ValueError(f"Ts must divide T and Ts % Tm = 0, got {T} and {Ts}")
            if not (C % Cs == 0 and Cs % Cm == 0):
                raise ValueError(f"Cs must divide C and Cs % Cm = 0, got {C} and {Cs}")

            if self.full_stokes:
                vis_model = broadcast_translate_corrs(vis_model, data.coherencies, (('XX', 'XY'), ('YX', 'YY')))
                vis_data = broadcast_translate_corrs(data.vis_data, data.coherencies, (('XX', 'XY'), ('YX', 'YY')))
                weights = broadcast_translate_corrs(data.weights, data.coherencies, (('XX', 'XY'), ('YX', 'YY')))
                flags = broadcast_translate_corrs(data.flags, data.coherencies, (('XX', 'XY'), ('YX', 'YY')))
            else:
                vis_model = broadcast_translate_corrs(vis_model, data.coherencies, ('I',))
                vis_data = broadcast_translate_corrs(data.vis_data, data.coherencies, ('I',))
                weights = broadcast_translate_corrs(data.weights, data.coherencies, ('I',))
                flags = broadcast_translate_corrs(data.flags, data.coherencies, ('I',))

            # Average using average rule to match model: [Ts, B, Cs[, 2, 2]] -> [Tm, B, Cm[, 2, 2]]
            with TimerLog("Averaging data"):
                vis_data_avg, weights_avg, flags_avg = jax.block_until_ready(
                    average(vis_data, weights, flags))

            # Construct calibration

            with TimerLog("Calibrating"):
                model_times = time_to_jnp(data.model_times, data.ref_time)
                model_freqs = quantity_to_jnp(data.model_freqs, 'Hz')
                gains, solver_state, diagnostics = jax.block_until_ready(
                    calibrate(
                        vis_model, vis_data_avg, weights_avg, flags_avg, model_freqs, model_times,
                        data.antenna1,
                        data.antenna2,
                        solver_state
                    )
                )

            with TimerLog("Plotting calibration diagnostics"):

                # plot phase, amp over aperature
                for i in range(np.shape(gains)[0]):
                    fig, axs = plt.subplots(2, 1, figsize=(6, 10))
                    _gain = gains[i]  # [Tm, A, Cm, 2, 2]
                    if self.full_stokes:
                        _gain = _gain[0, :, 0, 0, 0]
                    else:
                        _gain = _gain[0, :, 0]
                    _gain = _gain / _gain[0]
                    _phase = np.angle(_gain)
                    _amplitude = np.abs(_gain)
                    lon = self.antennas.geodetic.lon
                    lat = self.antennas.geodetic.lat
                    sc = axs[0].scatter(lon, lat, c=_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
                    plt.colorbar(sc, ax=axs[0], label='Phase (rad)')
                    axs[0].set_title('Phase')
                    sc = axs[1].scatter(lon, lat, c=_amplitude, cmap='jet')
                    plt.colorbar(sc, ax=axs[1], label='Amplitude')
                    axs[1].set_title('Amplitude')
                    plt.savefig(
                        os.path.join(self.plot_folder,
                                     f'{self.run_name}_calibration_{data.sol_int_time_idx}_dir{i:03d}.png')
                    )
                    plt.close(fig)

                    _gain = gains[i]  # [Tm, A, Cm, 2, 2]
                    if self.full_stokes:
                        _gain = _gain[0, :, 0, 0, 0]
                    else:
                        _gain = _gain[0, :, 0]

                    G = _gain[:, None] * _gain.conj()[None, :]  # [A, A]
                    _phase = np.angle(G)
                    _amplitude = np.abs(G)

                    fig, axs = plt.subplots(2, 1, figsize=(6, 10))
                    sc = axs[0].imshow(_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi, interpolation='nearest',
                                       origin='lower')
                    plt.colorbar(sc, ax=axs[0], label='Phase (rad)')
                    axs[0].set_title('baseline-based G phase')
                    sc = axs[1].imshow(_amplitude, cmap='jet', interpolation='nearest', origin='lower')
                    plt.colorbar(sc, ax=axs[1], label='Amplitude')
                    axs[1].set_title('baseline-based G amplitude')
                    plt.savefig(
                        os.path.join(self.plot_folder,
                                     f'{self.run_name}_calibration_baseline_{data.sol_int_time_idx}_dir{i:03d}.png')
                    )
                    plt.close(fig)

                # row 1: Plot error
                # row 2: Plot r
                # row 3: plot chi-2 (F_norm)
                # row 4: plot damping
                iterations = int(np.max(diagnostics.iteration) + 1)
                fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
                diagnostics: MultiStepLevenbergMarquardtDiagnostic
                axs[0].plot(diagnostics.iteration[:iterations], diagnostics.error[:iterations])
                axs[0].set_title('Error')
                axs[1].plot(diagnostics.iteration[:iterations], diagnostics.r[:iterations])
                axs[1].set_title('r')
                axs[2].plot(diagnostics.iteration[:iterations], diagnostics.F_norm[:iterations])
                axs[2].set_title('|F|')
                axs[3].plot(diagnostics.iteration[:iterations], diagnostics.damping[:iterations])
                axs[3].set_title('Damping')
                axs[3].set_xlabel('Iteration')
                plt.savefig(
                    os.path.join(self.plot_folder, f'{self.run_name}_diagnostics_{data.sol_int_time_idx}.png')
                )
                plt.close(fig)
            # Compute residuals
            with TimerLog("Computing residuals"):
                # Subtract the model for the bright sources only
                num_cals = np.shape(data.vis_bright_sources)[0]
                vis_residuals = jax.block_until_ready(
                    calc_residual(vis_model[:num_cals], vis_data, gains[:num_cals], data.antenna1,
                                  data.antenna2)
                )
                # Convert back to input coherencies
                if self.full_stokes:
                    vis_residuals = broadcast_translate_corrs(vis_residuals, (('XX', 'XY'), ('YX', 'YY')),
                                                              data.coherencies)
                else:
                    vis_residuals = broadcast_translate_corrs(vis_residuals, ('I',), data.coherencies)

            # Send back to generator
            return ReturnData(
                vis_residuals=vis_residuals,
                solver_state=solver_state
            )

        return _step

    def run(self, data_generator: Generator[Data, ReturnData, None], Ts: int | None = None, Cs: int | None = None):

        main_step = self.build_main_step(Ts, Cs)
        # Predict data and model
        solver_state = None
        gen_response: ReturnData | None = None
        while True:
            try:
                data: Data = data_generator.send(gen_response)
            except StopIteration:
                break
            gen_response = main_step(data, solver_state)


def apply_gains_to_model_vis(vis_model, gains, antenna1, antenna2):
    """
    Compute the residual between the model visibilities and the observed visibilities.

    Args:
        vis_model: [D, Tm, B, Cm[,2,2]] the model visibilities per direction
        gains: [D, Tm, A, Cm[,2,2]] the gains
        antenna1: [B] the antenna1
        antenna2: [B] the antenna2

    Returns:
        [Tm, B, Cm[, 2, 2]] the residuals
    """

    def body_fn(accumulate, x):
        vis_model, gains = x

        g1 = gains[:, antenna1, :, ...]  # [Tm, B, Cm[, 2, 2]]
        g2 = gains[:, antenna2, :, ...]  # [Tm, B, Cm[, 2, 2]]

        @partial(
            simple_broadcast,  # [Tm,B,Cm,...]
            leading_dims=3
        )
        def apply_gains(g1, g2, vis):
            if np.shape(g1) != np.shape(g1):
                raise ValueError("Gains must have the same shape.")
            if np.shape(vis) != np.shape(g1):
                raise ValueError("Gains and visibilities must have the same shape.")
            if np.shape(g1) == (2, 2):
                return mp_policy.cast_to_vis(kron_product(g1, vis, g2.conj().T))
            elif np.shape(g1) == ():
                return mp_policy.cast_to_vis(g1 * vis * g2.conj())
            else:
                raise ValueError(f"Invalid shape: {np.shape(g1)}")

        delta_vis = apply_gains(g1, g2, vis_model)  # [Tm, B, Cm[, 2, 2]]
        return accumulate + delta_vis, ()

    if np.shape(vis_model)[0] != np.shape(gains)[0]:
        raise ValueError(
            f"Model visibilities and gains must have the same number of directions, got {np.shape(vis_model)[0]} and {np.shape(gains)[0]}")

    accumulate = jnp.zeros(np.shape(vis_model)[1:], dtype=vis_model.dtype)
    accumulate, _ = jax.lax.scan(body_fn, accumulate, (vis_model, gains))
    return accumulate


def compute_residual(vis_model, vis_data, gains, antenna1, antenna2):
    """
    Compute the residual between the model visibilities and the observed visibilities.

    Args:
        vis_model: [D, Tm, B, Cm[,2,2]] the model visibilities per direction
        vis_data: [Ts, B, Cs[,2,2]] the data visibilities, Ts = 0 mod Tm, Cs = 0 mod Cm i.e. Ts % Tm = 0, Cs % Cm = 0
        gains: [D, Tm, A, Cm[,2,2]] the gains
        antenna1: [B] the antenna1
        antenna2: [B] the antenna2

    Returns:
        [Ts, B, Cs[, 2, 2]] the residuals
    """

    def body_fn(accumulate, x):
        vis_model, gains = x

        g1 = gains[:, antenna1, :, ...]  # [Tm, B, Cm[, 2, 2]]
        g2 = gains[:, antenna2, :, ...]  # [Tm, B, Cm[, 2, 2]]

        @partial(
            simple_broadcast,  # [Tm,B,Cm,...]
            leading_dims=3
        )
        def apply_gains(g1, g2, vis):
            if np.shape(g1) != np.shape(g1):
                raise ValueError(f"Gains must have the same shape, "
                                 f"got {np.shape(g1)} and {np.shape(vis)}.")
            if np.shape(vis) != np.shape(g1):
                raise ValueError(f"Gains and visibilities must have the same shape, "
                                 f"got {np.shape(g1)} and {np.shape(vis)}.")
            if np.shape(g1) == (2, 2):
                return mp_policy.cast_to_vis(kron_product(g1, vis, g2.conj().T))
            elif np.shape(g1) == ():
                return mp_policy.cast_to_vis(g1 * vis * g2.conj())
            else:
                raise ValueError(f"Invalid shape: {np.shape(g1)}")

        delta_vis = apply_gains(g1, g2, vis_model)  # [Tm, B, Cm[, 2, 2]]
        return accumulate + delta_vis, ()

    if np.shape(vis_model)[0] != np.shape(gains)[0]:
        raise ValueError(
            f"Model visibilities and gains must have the same number of directions, "
            f"got {np.shape(vis_model)[0]} and {np.shape(gains)[0]}")

    # num_directions = np.shape(vis_model)[0]

    accumulate = jnp.zeros(np.shape(vis_model)[1:], dtype=vis_model.dtype)
    accumulate, _ = jax.lax.scan(body_fn, accumulate, (vis_model, gains))

    # Invert average rule with tile
    time_rep = np.shape(vis_data)[0] // np.shape(accumulate)[0]  # Ts / Tm
    freq_rep = np.shape(vis_data)[2] // np.shape(accumulate)[2]  # Cs / Cm
    tile_reps = [1] * len(np.shape(accumulate))
    tile_reps[0] = time_rep
    tile_reps[2] = freq_rep
    if np.prod(tile_reps) > 1:
        # print(f"Replicating accumulated model vis {tile_reps}")
        accumulate = jnp.tile(accumulate, tile_reps)
    return vis_data - accumulate


class DataGenInput(NamedTuple):
    sol_int_time_idx: int
    time_idxs: IntArray
    freq_idxs: IntArray
    model_times: at.Time
    model_freqs: au.Quantity
    ref_time: at.Time


def create_data_input_gen(sol_int_freq_idx: int, T: int, C: int, Tm: int, Cm: int, obsfreqs: au.Quantity,
                          obstimes: at.Time, ref_time: at.Time) -> Generator[DataGenInput, None, None]:
    """
    Create a generator that yields input that can be used to fetch input for calibration algorithm.

    Args:
        sol_int_freq_idx: the solution interval frequency index to produce data for.
        T: the solution interval size in time
        C: the solution interval size in frequency
        Tm: the model size in time
        Cm: the model size in frequency
        obsfreqs: the observed frequencies
        obstimes: the observed times
        ref_time: the reference time

    Yields:
        DataGenInput: the input for the generator for calibration algorithm.
    """
    if len(obsfreqs) % C != 0:
        raise ValueError(
            f"Solution interval frequencies must divide full resolution frequencies, got {len(obsfreqs)} and {C}")
    if len(obstimes) % T != 0:
        raise ValueError(f"Solution interval times must divide full resolution times, got {len(obstimes)} and {T}")
    if T % Tm != 0:
        raise ValueError(f"Model times must divide full resolution times, got {T} and {Tm}")
    if C % Cm != 0:
        raise ValueError(f"Model frequencies must divide full resolution frequencies, got {C} and {Cm}")
    if sol_int_freq_idx * C >= len(obsfreqs):
        raise ValueError(
            f"sol_int_freq_idx * C must be less than len(obsfreqs), got {sol_int_freq_idx * C} and {len(obsfreqs)}")

    print(f"Producing data for frequency solution interval {sol_int_freq_idx}")
    freqs = quantity_to_jnp(obsfreqs, 'Hz')
    for sol_int_time_idx in range(len(obstimes) // T):
        time_idxs = np.arange(sol_int_time_idx * T, (sol_int_time_idx + 1) * T)
        freq_idxs = np.arange(sol_int_freq_idx * C, (sol_int_freq_idx + 1) * C)

        sol_int_times = time_to_jnp(obstimes[time_idxs], ref_time)
        sol_int_freqs = freqs[freq_idxs]

        model_times = jnp_to_time(average_rule(sol_int_times, Tm, axis=0), ref_time)
        model_freqs = np.asarray(average_rule(sol_int_freqs, Cm, axis=0)) * au.Hz
        with TimerLog(
                f"Producing data for time solution interval {sol_int_time_idx}"):
            yield DataGenInput(sol_int_time_idx=sol_int_time_idx, time_idxs=time_idxs, freq_idxs=freq_idxs,
                               model_times=model_times, model_freqs=model_freqs,
                               ref_time=ref_time)
