import dataclasses
import os
from functools import partial
from typing import Generator, Tuple, Any, Literal
from typing import NamedTuple

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp

from dsa2000_cal.calibration_step import calibration_step
from dsa2000_cal.probabilistic_models.gain_prior_models import GainPriorModel
from dsa2000_cal.solvers.multi_step_lm import LMDiagnostic
from dsa2000_cal.subtraction_step import subtraction_step
from dsa2000_common.common.array_types import ComplexArray, FloatArray, BoolArray, IntArray
from dsa2000_common.common.corr_utils import broadcast_translate_corrs
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.quantity_utils import quantity_to_jnp, time_to_jnp, jnp_to_time
from dsa2000_common.common.ray_utils import TimerLog
from dsa2000_fm.actors.average_utils import average_rule

tfpd = tfp.distributions


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
    gains: Any  # [D, T, A, C[, 2, 2]]
    params: Any


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
    full_stokes: bool
    antennas: ac.EarthLocation

    gain_stddev: float = 2.
    dd_type: Literal['unconstrained', 'rice', 'phase_only', 'amplitude_only'] = 'unconstrained'
    dd_dof: int = 4
    double_differential: bool = True
    di_dof: int = 4
    di_type: Literal['unconstrained', 'rice', 'phase_only', 'amplitude_only'] = 'unconstrained'

    verbose: bool = False
    num_devices: int = 1
    backend: str = 'cpu'

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)

    def build_average_rule(self):

        @partial(jax.jit, static_argnames=['Ts', 'Cs'])
        def average(vis_data: ComplexArray, weights: FloatArray, Ts: int, Cs: int) -> Tuple[
            ComplexArray, FloatArray]:
            # average data to match model: [Ts, B, Cs[, 2, 2]] -> [Tm, B, Cm[, 2, 2]]
            if Ts is not None:
                time_average_rule = partial(
                    average_rule,
                    num_model_size=Ts,
                    axis=0
                )
            else:
                time_average_rule = lambda x: x
            if Cs is not None:
                freq_average_rule = partial(
                    average_rule,
                    num_model_size=Cs,
                    axis=2
                )
            else:
                freq_average_rule = lambda x: x
            vis_data_avg = time_average_rule(freq_average_rule(vis_data))
            weights_avg = jnp.reciprocal(time_average_rule(freq_average_rule(jnp.reciprocal(weights))))
            return vis_data_avg, weights_avg

        return average

    def build_main_step(self, Ts: int | None = None, Cs: int | None = None):
        """
        Build the main step of the iterative calibration algorithm.

        Args:
            Ts: how many times to average the data to match the model
            Cs: how many frequencies to average the data to match the model

        Returns:
            The main step function
        """
        average = self.build_average_rule()

        # Predict data and model

        def _step(data: Data, params=None) -> ReturnData:
            """
            Perform a single step of calibration.

            Args:
                data: the data to calibrate
                params: the last params

            Returns:
                the residuals and the state of the solver
            """
            nonlocal Ts, Cs
            vis_model = jnp.concatenate(
                [data.vis_bright_sources, data.vis_background],
                axis=0
            )  # [S + E, Tm, B, Cm, num_coh]

            D, Tm, B, Cm, num_coh = np.shape(vis_model)
            T, B_, C, num_coh_ = np.shape(data.vis_data)
            dsa_logger.info(
                f"Model shape: D={D}, Tm={Tm}, B={B}, Cm={Cm}, num_coh={num_coh}. Size={vis_model.nbytes / 2 ** 20:.2f} MB.")
            dsa_logger.info(
                f"Data shape: T={T}, B={B}, C={C}, num_coh={num_coh}. Size={data.vis_data.nbytes / 2 ** 20:.2f} MB.")
            if B != B_:
                raise ValueError(f"Model and data must have the same number of baselines, got {B} and {B_}")
            if num_coh != num_coh_:
                raise ValueError(
                    f"Model and data must have the same number of coherence products, got {num_coh} and {num_coh_}")

            if T % Tm != 0:
                dsa_logger.info(f"Possible values of Tm={[i for i in range(0, T + 1) if T % i == 0]}")
                raise ValueError(f"Model times must divide full resolution times, got {T} and {Tm}")

            if C % Cm != 0:
                dsa_logger.info(f"Possible values of Cm={[i for i in range(0, C + 1) if C % i == 0]}")
                raise ValueError(f"Model frequencies must divide full resolution frequencies, got {C} and {Cm}")

            # Print out the possible values for Ts and Cs such that Ts % Tm = 0 and Cs % Cm = 0 and T % Ts == 0 and C % Cs == 0

            if Ts is None:
                Ts = Tm
            if Cs is None:
                Cs = Cm
            if not (T % Ts == 0 and Ts % Tm == 0):
                dsa_logger.info(f"Possible values of Ts={[i for i in range(Tm, T + 1) if T % i == 0 and i % Tm == 0]}")
                raise ValueError(f"Ts must divide T and Ts % Tm = 0, got {T} and {Ts}")
            if not (C % Cs == 0 and Cs % Cm == 0):
                dsa_logger.info(f"Possible values of Cs={[i for i in range(Cm, C + 1) if C % i == 0 and i % Cm == 0]}")
                raise ValueError(f"Cs must divide C and Cs % Cm = 0, got {C} and {Cs}")

            weights = data.weights * np.logical_not(data.flags)

            if self.full_stokes:
                vis_model = broadcast_translate_corrs(vis_model, data.coherencies, (('XX', 'XY'), ('YX', 'YY')))
                vis_data = broadcast_translate_corrs(data.vis_data, data.coherencies, (('XX', 'XY'), ('YX', 'YY')))
                weights = np.reciprocal(
                    broadcast_translate_corrs(jnp.reciprocal(weights), data.coherencies, (('XX', 'XY'), ('YX', 'YY')))
                )
            else:
                vis_model = broadcast_translate_corrs(vis_model, data.coherencies, ('I',))[..., 0]
                vis_data = broadcast_translate_corrs(data.vis_data, data.coherencies, ('I',))[..., 0]
                weights = jnp.reciprocal(
                    broadcast_translate_corrs(jnp.reciprocal(weights), data.coherencies, ('I',))[..., 0]
                )

            # Average using average rule to match model: [Ts, B, Cs[, 2, 2]] -> [Tm, B, Cm[, 2, 2]]
            with TimerLog("Averaging data"):
                vis_data_avg, weights_avg = jax.block_until_ready(
                    average(vis_data, weights, Ts=Ts, Cs=Cs)
                )
                dsa_logger.info(
                    f"Averaged data down to: Ts={Ts}, B={B}, Cs={Cs}, num_coh={num_coh}. Size={vis_data_avg.nbytes / 2 ** 20:.2f} MB."
                )
            # Construct calibration

            with TimerLog("Calibrating"):
                A = len(self.antennas)
                model_times = time_to_jnp(data.model_times, data.ref_time)
                model_freqs = quantity_to_jnp(data.model_freqs, 'Hz')
                gain_prior_model = GainPriorModel(
                    times=model_times,
                    freqs=model_freqs,
                    num_source=D,
                    num_ant=A,
                    gain_stddev=self.gain_stddev,
                    dd_dof=self.dd_dof,
                    di_dof=self.di_dof,
                    double_differential=self.double_differential,
                    dd_type=self.dd_type,
                    di_type=self.di_type,
                    full_stokes=self.full_stokes
                )
                params, gains, diagnostics = jax.block_until_ready(
                    calibration_step(
                        params,
                        vis_model,
                        vis_data_avg,
                        weights_avg,
                        data.antenna1,
                        data.antenna2,
                        gain_probabilistic_model=gain_prior_model,
                        verbose=self.verbose,
                        num_devices=self.num_devices,
                        backend=self.backend,
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
                diagnostics: LMDiagnostic
                axs[0].plot(diagnostics.iteration[:iterations], diagnostics.g_norm[:iterations])
                axs[0].set_title('Error')
                axs[1].plot(diagnostics.iteration[:iterations], diagnostics.gain_ratio[:iterations])
                axs[1].set_title('r')
                axs[2].plot(diagnostics.iteration[:iterations], diagnostics.Q[:iterations])
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
                    subtraction_step(
                        gains[:num_cals], vis_model[:num_cals], vis_data, data.antenna1,
                        data.antenna2, num_devices=self.num_devices, backend=self.backend
                    )
                )  # [T, B, C[, 2, 2]]
                # Convert back to input coherencies
                if self.full_stokes:
                    vis_residuals = broadcast_translate_corrs(vis_residuals, (('XX', 'XY'), ('YX', 'YY')),
                                                              data.coherencies)
                else:
                    vis_residuals = broadcast_translate_corrs(vis_residuals[..., None], ('I',), data.coherencies)

            # Send back to generator
            return ReturnData(
                vis_residuals=vis_residuals,
                gains=gains,
                params=params
            )

        return _step

    def run(self, data_generator: Generator[Data, ReturnData, None], Ts: int | None = None, Cs: int | None = None):

        main_step = self.build_main_step(Ts, Cs)
        # Predict data and model
        params = None
        gen_response: ReturnData | None = None
        while True:
            try:
                data: Data = data_generator.send(gen_response)
            except StopIteration:
                break
            gen_response = main_step(data, params)


class DataGenInput(NamedTuple):
    sol_int_time_idx: int  # the solution interval time index
    time_idxs: IntArray  # [Ts] the time indices, starting at 0
    freq_idxs: IntArray  # [Cs] the frequency indices, starting at 0
    model_times: at.Time  # [Tm] the model times
    model_freqs: au.Quantity  # [Cm] the model frequencies
    ref_time: at.Time  # the reference time


def create_data_input_gen(sol_int_freq_idx: int, T: int, C: int, Tm: int, Cm: int, obsfreqs: au.Quantity,
                          obstimes: at.Time, ref_time: at.Time) -> Generator[DataGenInput, None, None]:
    """
    Create a generator that yields input that can be used to fetch input for calibration algorithm.

    Args:
        sol_int_freq_idx: the solution interval frequency index to produce data for.
        T: the solution interval size in time, this many times will be averaged to Ts.
        C: the solution interval size in frequency, this many frequencies will be averaged to Cs.
        Tm: the model size in time, must divide Ts. This many times will be used to model the Ts data
        Cm: the model size in frequency, must divide Cs. This many frequencies will be used to model the Cs data.
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

    dsa_logger.info(f"Producing data for frequency solution interval {sol_int_freq_idx}")
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
