from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jaxctx import transform, CtxParams
from jaxctx.priors.prior import Prior

from dsa2000_cal.solvers.multi_step_lm import lm_solver
from dsa2000_common.common.array_types import FloatArray, ComplexArray, IntArray
from dsa2000_common.common.mixed_precision_utils import mp_policy
from dsa2000_common.common.sum_utils import scan_sum

tfpd = tfp.distributions


class StreamData(NamedTuple):
    vis_obs: FloatArray  # [T, B, C, 2, 2] visibility observations
    weights: FloatArray  # [T, B, C, 2, 2] weights for the observations
    vis_model: FloatArray  # [D, T, B, C, 2, 2] model visibilities


class ModelParams(NamedTuple):
    tec: FloatArray  # [D, A] TEC values in mTECU
    lna_amp: FloatArray  # [A] LNA amplitude gains
    lna_phase: FloatArray  # [A] LNA phase gains in radians
    gains: ComplexArray  # [D, A, C] total gains in complex form


def get_gain_params(freqs: FloatArray, antennas_gcrs: FloatArray, directions_radec: FloatArray,
                    tec_stddev_mtecu: FloatArray,
                    lna_amp_mean: FloatArray, lna_amp_stddev: FloatArray, lna_phase_mean_rad: FloatArray,
                    lna_phase_stddev_rad: FloatArray):
    num_directions = directions_radec.shape[0]  # Number of directions
    num_antennas = antennas_gcrs.shape[0]  # Number of antennas
    # Build the total gains from the model
    tec_conv = -8.4479745e6 / freqs  # rad / mTECU [C]
    delay_conv = (2 * np.pi * 1e-9) * freqs  # rad / ns [C]
    tec = Prior(
        tfpd.Normal(
            loc=jnp.zeros((num_directions, num_antennas), dtype=jnp.float32),
            scale=tec_stddev_mtecu * jnp.ones((num_directions, num_antennas), dtype=jnp.float32),
        ),
        name='tec'
    ).parameter()
    g_ones = jnp.ones((num_antennas,))
    lna_amp = Prior(
        tfpd.LogNormal.experimental_from_mean_variance(
            mean=g_ones * lna_amp_mean,
            variance=g_ones * lna_amp_stddev ** 2
        ),
        name="G_amp"
    ).parameter(random_init=True)

    lna_phase = Prior(
        tfpd.Normal(
            loc=g_ones * lna_phase_mean_rad,
            scale=g_ones * lna_phase_stddev_rad
        ),
        name="G_phase"
    ).parameter(random_init=True)
    phase = tec[:, :, None] * tec_conv + lna_phase[:, None]  # [D, A, C]
    gains = lna_amp[:, None] * jax.lax.complex(
        jnp.cos(phase), jnp.sin(phase)
    )
    return ModelParams(
        tec=tec,  # [D, A]
        lna_amp=lna_amp,  # [A]
        lna_phase=lna_phase,  # [A]
        gains=gains  # [D, A, C]
    )


def residual_fn(
        gains,
        vis_obs: ComplexArray, vis_model: ComplexArray, weights: FloatArray,
        antenna1: IntArray, antenna2: IntArray
) -> FloatArray:
    """
    Compute the residual between the observed visibilities and the model visibilities, given the gains.

    Args:
        gains: [D, A, C] total DD gains in complex form
        vis_obs: [T, B, C, 2, 2] visibility observations
        vis_model: [D, T, B, C, 2, 2] model visibilities, with pre-applied bandpass, and polarisation solution.
        weights: [T, B, C, 2, 2] weights for the observations
        antenna1: [B] indices of the first antenna in the baseline
        antenna2: [B] indices of the second antenna in the baseline

    Returns:
        [T, B, C, 2, 2] complex residuals, the difference between the observed visibilities and the model visibilities
    """

    def accum_fn(x):
        vis_model, gains = x

        g1 = gains[antenna1, :, ...]  # [B, C]
        g2 = gains[antenna2, :, ...]  # [B, C]

        mueller_coeff = (g1 * jax.lax.conj(g2))  # [B, C]
        delta_vis = mp_policy.cast_to_vis(mueller_coeff[..., None, None] * vis_model)  # [T, B, C, 2, 2]
        # TODO: consider F16 return type for accumulate, passing real and imaginary parts separately

        return delta_vis

    zeros = jnp.zeros(np.shape(vis_model)[1:], dtype=mp_policy.vis_dtype)
    # TODO: consider prefix scan for performance, or vmapping the scan in chunks
    accumulate = scan_sum(accum_fn, zeros, (vis_model, gains), unroll=2)  # [T, B, C, 2, 2]
    residual = jax.lax.sub(vis_obs, accumulate)  # [T, B, C, 2, 2]
    return residual * jnp.sqrt(weights)


def solve_dd_gains(
        # Initial parameters
        init_params: CtxParams | None,
        # Data for solve
        vis_obs: ComplexArray, vis_model: ComplexArray, weights: FloatArray,
        antenna1: IntArray, antenna2: IntArray,
        # Parameters for evaluating gain model
        freqs: FloatArray, antennas_gcrs: FloatArray, directions_radec: FloatArray,
        # Gain model prior hyperparameters
        tec_stddev_mtecu: FloatArray, lna_amp_mean: FloatArray, lna_amp_stddev: FloatArray,
        lna_phase_mean_rad: FloatArray,
        lna_phase_stddev_rad: FloatArray
):
    """
    Solve the gain parameters using a Levenberg-Marquardt solver.

    Args:
        init_params: if None, the initial parameters will be computed from the gain model.
        vis_obs: [T, B, C, 2, 2] visibility observations
        vis_model: [D, T, B, C, 2, 2] model visibilities, with pre-applied bandpass, and polarisation solution.
        weights: [T, B, C, 2, 2] weights for the observations
        antenna1: [B] indices of the first antenna in the baseline
        antenna2: [B] indices of the second antenna in the baseline
        freqs: [C] frequencies in Hz
        antennas_gcrs: [A, 3] antenna positions in GCRS coordinates
        directions_radec: [D, 2] directions in RA/Dec coordinates
        tec_stddev_mtecu: scalar, or [D, A] TEC standard deviation in mTECU
        lna_amp_mean: scalar, or [A] mean LNA amplitude gain
        lna_amp_stddev:, scalar, or [A] standard deviation of LNA amplitude gain
        lna_phase_mean_rad: scalar, or [A] mean LNA phase gain in radians
        lna_phase_stddev_rad: scalar, or [A] standard deviation of LNA phase gain in radians

    Returns:
        params: the parameters of the gain model, as a CtxParams object.
        gains: [D, A, C] total DD gains in complex form
        diagnostics: diagnostics from the Levenberg-Marquardt solver
    """
    transformed_get_gains = transform(get_gain_params)

    rngs = {'params': jax.random.PRNGKey(0)}
    if init_params is None:
        init_results = transformed_get_gains.init(
            rngs, None,
            # gain model args
            freqs, antennas_gcrs, directions_radec, tec_stddev_mtecu, lna_amp_mean, lna_amp_stddev, lna_phase_mean_rad,
            lna_phase_stddev_rad
        )
        init_params = init_results.collections

    def _residual_fn(params,
                     # residual kwargs
                     vis_obs, vis_model, weights, antenna1, antenna2,
                     # gain model kwargs
                     freqs: FloatArray, antennas_gcrs, directions_radec: FloatArray,
                     tec_stddev_mtecu: FloatArray,
                     lna_amp_mean: FloatArray, lna_amp_stddev: FloatArray, lna_phase_mean_rad: FloatArray,
                     lna_phase_stddev_rad: FloatArray
                     ):
        model = transformed_get_gains.apply(
            {},
            params,
            # gain model args
            freqs, antennas_gcrs, directions_radec, tec_stddev_mtecu, lna_amp_mean, lna_amp_stddev, lna_phase_mean_rad,
            lna_phase_stddev_rad
        ).fn_val
        gains = model.gains
        return residual_fn(
            gains=gains,
            vis_obs=vis_obs,
            vis_model=vis_model,
            weights=weights,
            antenna1=antenna1,
            antenna2=antenna2
        )

    # Solve the least squares problem using the Levenberg-Marquardt algorithm
    params, diagnostics = lm_solver(
        residual_fn=_residual_fn,
        x0=init_params,
        args=(
            # residual args
            vis_obs, vis_model, weights, antenna1, antenna2,
            # gain model args
            freqs, antennas_gcrs, directions_radec, tec_stddev_mtecu, lna_amp_mean, lna_amp_stddev, lna_phase_mean_rad,
            lna_phase_stddev_rad
        )
    )

    model = transformed_get_gains.apply(
        {},
        params,
        # gain model args
        freqs, antennas_gcrs, directions_radec, tec_stddev_mtecu, lna_amp_mean, lna_amp_stddev, lna_phase_mean_rad,
        lna_phase_stddev_rad
    ).fn_val

    return params, model, diagnostics


def test_residual_zero_for_unity_gains():
    """
    residual_fn should return zero residuals when the observed visibilities
    exactly match the model visibilities under unity gains.
    """
    # Dimensions: 1 direction, 1 time, 1 baseline, 1 channel, 2x2 pol
    D, T, B, C, A = 1, 2, 3, 4, 2
    # Model visibilities: all ones (complex)
    vis_model = jnp.ones((D, T, B, C, 2, 2), dtype=jnp.complex64)
    # Unity gains
    gains = jnp.ones((D, A, C), dtype=jnp.complex64)
    # Observations equal the model under unity gains
    vis_obs = vis_model[0]
    # Unit weights
    weights = jnp.ones((T, B, C, 2, 2), dtype=jnp.float32)
    # Single baseline between antenna 0 and antenna 1
    antenna1 = jnp.array([0, 0, 1], dtype=jnp.int32)
    antenna2 = jnp.array([0, 1, 1], dtype=jnp.int32)

    # Compute residuals
    residual = residual_fn(
        gains=gains,
        vis_obs=vis_obs,
        vis_model=vis_model,
        weights=weights,
        antenna1=antenna1,
        antenna2=antenna2
    )

    # Check shapes and values
    assert residual.shape == (T, B, C, 2, 2)
    assert jnp.allclose(residual, 0.0, atol=1e-6)


def test_solve_dd_gains_returns_model_and_params_shapes():
    """
    solve_dd_gains on a trivial unity-gain problem should complete without error,
    return gains of shape [D, A, C], and approximately recover unity gains.
    """
    # Problem dimensions
    D, T, B, C, A = 1, 2, 3, 4, 5
    # Frequency axis
    freqs = jnp.array([1e8]*C, dtype=jnp.float32)
    # One direction at RA/Dec = (0,0)
    directions_radec = jnp.zeros((D, 2), dtype=jnp.float32)
    # Antenna positions (2 antennas)
    antennas_gcrs = jnp.zeros((A, 3), dtype=jnp.float32)
    # Hyperparameters: broad TEC prior, mild LNA priors
    tec_stddev = jnp.ones((D, A), dtype=jnp.float32) * 1e3
    lna_amp_mean = jnp.array(1.0, dtype=jnp.float32)
    lna_amp_stddev = jnp.array(0.1, dtype=jnp.float32)
    lna_phase_mean = jnp.array(0.0, dtype=jnp.float32)
    lna_phase_stddev = jnp.array(0.1, dtype=jnp.float32)
    # Single baseline
    antenna1 = jnp.array([0, 0, 1], dtype=jnp.int32)
    antenna2 = jnp.array([0, 1, 1], dtype=jnp.int32)
    # Model and observed visibilities: unity gains -> model equals obs
    vis_model = jnp.ones((D, T, B, C, 2, 2), dtype=jnp.complex64)
    vis_obs = vis_model[0]
    weights = jnp.ones((T, B, C, 2, 2), dtype=jnp.float32)

    # Call the solver
    params, model, diagnostics = solve_dd_gains(
        init_params=None,
        vis_obs=vis_obs,
        vis_model=vis_model,
        weights=weights,
        antenna1=antenna1,
        antenna2=antenna2,
        freqs=freqs,
        antennas_gcrs=antennas_gcrs,
        directions_radec=directions_radec,
        tec_stddev_mtecu=tec_stddev,
        lna_amp_mean=lna_amp_mean,
        lna_amp_stddev=lna_amp_stddev,
        lna_phase_mean_rad=lna_phase_mean,
        lna_phase_stddev_rad=lna_phase_stddev
    )

    # The returned gains should have shape [D, A, C]
    assert model.gains.shape == (D, A, C)
    # For this trivial problem, the solver should recover near-unity gains
    print(model.gains)

    print(params)
