from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Tuple, List

import astropy.units as au
import jax
import jaxopt
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import lax
from jax import numpy as jnp
from jax._src.typing import SupportsDType

from dsa2000_cal.common.coord_utils import lmn_to_icrs
from dsa2000_cal.common.jax_utils import pytree_unravel, promote_pytree
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords, VisibilityData, MeasurementSet
from dsa2000_cal.predict.fft_stokes_I_predict import FFTStokesIPredict, FFTStokesIModelData
from dsa2000_cal.predict.gaussian_predict import GaussianPredict, GaussianModelData
from dsa2000_cal.predict.point_predict import PointPredict, PointModelData
from dsa2000_cal.predict.vec_utils import kron_product
from dsa2000_cal.source_models.corr_translation import flatten_coherencies, stokes_to_linear
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


@dataclass(eq=False)
class Calibration:
    # models to calibrate based on. Each model gets a gain direction in the flux weighted direction.
    wsclean_source_models: List[WSCleanSourceModel]
    fits_source_models: List[FitsStokesISourceModel]

    preapply_gain_model: GainModel | None

    # Calibration parameters
    inplace_subtract: bool
    num_iterations: int
    residual_ms_folder: str | None = None
    seed: int = 42
    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64

    def __post_init__(self):
        self.num_calibrators = len(self.wsclean_source_models) + len(self.fits_source_models)
        self.key = jax.random.PRNGKey(self.seed)

    def calibrate(self, ms: MeasurementSet):
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

        gen = ms.create_block_generator(vis=True, weights=True, flags=True, relative_time_idx=True)
        gen_response = None

        init_params = self.get_init_params(
            num_source=self.num_calibrators,
            num_time=1,
            num_ant=len(ms.meta.antennas),
            num_chan=len(ms.meta.freqs)
        )  # [num_source, 1, num_ant, num_freqs, 2, 2]

        calibrator_lmn = au.Quantity(
            np.stack(
                [
                    model.flux_weighted_lmn() for model in self.wsclean_source_models
                ] + [
                    model.flux_weighted_lmn() for model in self.fits_source_models
                ],
                axis=0)
        )  # [num_calibrators, 3]

        while True:
            try:
                time, visibility_coords, data = gen.send(gen_response)
            except StopIteration:
                break

            cal_sources = lmn_to_icrs(lmn=calibrator_lmn, phase_tracking=ms.meta.phase_tracking, time=time)

            if self.preapply_gain_model is not None:
                # Since we pass a single `time` we need time_idx to be relative.
                preapply_gains = self.preapply_gain_model.compute_gain(
                    freqs=ms.meta.freqs,
                    time=time,
                    phase_tracking=ms.meta.phase_tracking,
                    array_location=ms.meta.array_location,
                    sources=cal_sources
                )  # [num_calibrators, num_ant, num_chan, 2, 2]
            else:
                preapply_gains = jnp.tile(
                    jnp.eye(2, dtype=self.float_dtype)[None, None, None, ...],
                    reps=(self.num_calibrators, len(ms.meta.antennas), len(ms.meta.freqs), 1, 1)
                )

            self.key, key = jax.random.split(self.key)

            params, results = self.solve(
                freqs=ms.meta.freqs,
                preapply_gains=preapply_gains,
                init_params=init_params,
                vis_data=data,
                vis_coords=visibility_coords
            )

            # Subtract the model from the data and store in subtracted MS

            # Store params
            init_params = params

    def _stokes_I_to_linear(self, image_I: jax.Array) -> jax.Array:
        """
        Convert Stokes I to linear.

        Args:
            image_I: [...]

        Returns:
            image_linear: [..., 2, 2]
        """
        shape = np.shape(image_I)
        image_I = lax.reshape(image_I, (np.size(image_I),))
        zero = jnp.zeros_like(image_I)
        image_stokes = jnp.stack([image_I, zero, zero, image_I], axis=-1)  # [..., 4]
        image_linear = jax.vmap(partial(stokes_to_linear, flat_output=False))(image_stokes)  # [..., 2, 2]
        return lax.reshape(image_linear, shape + (2, 2))

    def _build_log_likelihood(self, freqs: jax.Array, preapply_gains: jax.Array, vis_data: VisibilityData,
                              vis_coords: VisibilityCoords):
        num_rows, num_freqs, _ = vis_data.vis.shape
        vis = jnp.zeros((self.num_calibrators, num_rows, num_freqs, 2, 2),
                        vis_data.vis.dtype)  # [cal_dirs, num_rows, num_chans, 2, 2]
        # Predict the visibilities with pre-applied gains
        dft_predict = PointPredict(convention=self.convention,
                                   dtype=self.dtype)
        gaussian_predict = GaussianPredict(convention=self.convention,
                                           dtype=self.dtype)
        faint_predict = FFTStokesIPredict(convention=self.convention,
                                          dtype=self.dtype)
        # Each calibrator has a source model which is a collection of sources that make up the calibrator.
        cal_idx = 0
        for wsclean_source_model in self.wsclean_source_models:
            preapply_gains_cal = preapply_gains[cal_idx]  # [num_ant, num_chan, 2, 2]
            # Add time dime
            preapply_gains_cal = preapply_gains_cal[None, ...]  # [1, num_ant, num_chan, 2, 2]
            # Points
            l0 = quantity_to_jnp(wsclean_source_model.point_source_model.l0)  # [source]
            m0 = quantity_to_jnp(wsclean_source_model.point_source_model.m0)  # [source]
            n0 = jnp.sqrt(1. - l0 ** 2 - m0 ** 2)  # [source]

            lmn = jnp.stack([l0, m0, n0], axis=-1)  # [source, 3]
            image_I = quantity_to_jnp(wsclean_source_model.point_source_model.A)  # [source, chan]
            image_linear = self._stokes_I_to_linear(image_I)  # [source, chan, 2, 2]

            dft_model_data = PointModelData(
                gains=preapply_gains_cal,  # [1, num_ant, num_chan, 2, 2]
                lmn=lmn,
                image=image_linear
            )
            vis = vis.at[cal_idx].set(
                dft_predict.predict(
                    freqs=freqs, dft_model_data=dft_model_data,
                    visibility_coords=vis_coords
                )  # [num_rows, num_chans, 2, 2]
            )

            # Gaussians
            l0 = quantity_to_jnp(wsclean_source_model.gaussian_source_model.l0)  # [source]
            m0 = quantity_to_jnp(wsclean_source_model.gaussian_source_model.m0)  # [source]
            n0 = jnp.sqrt(1. - l0 ** 2 - m0 ** 2)  # [source]

            lmn = jnp.stack([l0, m0, n0], axis=-1)  # [source, 3]
            image_I = quantity_to_jnp(wsclean_source_model.gaussian_source_model.A)  # [source, chan]
            image_linear = self._stokes_I_to_linear(image_I)  # [source, chan, 2, 2]

            ellipse_params = jnp.stack([
                quantity_to_jnp(wsclean_source_model.gaussian_source_model.major),
                quantity_to_jnp(wsclean_source_model.gaussian_source_model.minor),
                quantity_to_jnp(wsclean_source_model.gaussian_source_model.theta)
            ],
                axis=-1)  # [source, 3]

            gaussian_model_data = GaussianModelData(
                image=image_linear,  # [source, chan, 2, 2]
                gains=preapply_gains_cal,  # [1, num_ant, num_chan, 2, 2]
                ellipse_params=ellipse_params,  # [source, 3]
                lmn=lmn  # [source, 3]
            )

            vis = vis.at[cal_idx].add(
                gaussian_predict.predict(
                    freqs=freqs,  # [chan]
                    gaussian_model_data=gaussian_model_data,  # [source, chan, 2, 2]
                    visibility_coords=vis_coords  # [row, 3]
                )  # [num_rows, num_chans, 2, 2]
            )

            cal_idx += 1

        for fits_source_model in self.fits_source_models:
            preapply_gains_cal = preapply_gains[cal_idx]  # [num_ant, num_chan, 2, 2]
            # Add time dime
            preapply_gains_cal = preapply_gains_cal[None, ...]  # [1, num_ant, num_chan, 2, 2]
            l0 = quantity_to_jnp(fits_source_model.l0)  # [num_chan]
            m0 = quantity_to_jnp(fits_source_model.m0)  # [num_chan]
            dl = quantity_to_jnp(fits_source_model.dl)  # [num_chan]
            dm = quantity_to_jnp(fits_source_model.dm)  # [num_chan]
            image = jnp.stack(
                [
                    quantity_to_jnp(img)
                    for img in fits_source_model.images
                ],
                axis=0
            )  # [num_chan, Nx, Ny]

            faint_model_data = FFTStokesIModelData(
                image=image,  # [num_chan, Nx, Ny]
                gains=preapply_gains_cal,  # [1, num_ant, num_chan, 2, 2]
                l0=l0,  # [num_chan]
                m0=m0,  # [num_chan]
                dl=dl,  # [num_chan]
                dm=dm  # [num_chan]
            )

            vis = vis.at[cal_idx].set(
                faint_predict.predict(
                    freqs=freqs,
                    faint_model_data=faint_model_data,
                    visibility_coords=vis_coords
                )
            )

        # vis now contains the model visibilities for each calibrator

        def _log_likelihood(gains: jax.Array) -> jax.Array:
            """
            Compute the log probability of the data given the gains.

            Args:
                gains: [cal_dirs, num_ant, num_time, num_chans, 2, 2]

            Returns:
                log_prob: scalar
            """

            # V_ij = G_i * V_ij * G_j^H
            g1 = gains[:, vis_coords.antenna_1, vis_coords.time_idx,
                 ...]  # [cal_dirs, num_rows, num_chans, 2, 2]
            g2 = gains[:, vis_coords.antenna_2, vis_coords.time_idx,
                 ...]  # [cal_dirs, num_rows, num_chans, 2, 2]

            @partial(jax.vmap, in_axes=(2, 2, 2), out_axes=2)  # over num_chans
            @partial(jax.vmap, in_axes=(0, 0, 0))  # over cal_dirs
            @partial(jax.vmap, in_axes=(0, 0, 0))  # over num_rows
            def transform(g1, g2, vis):
                return flatten_coherencies(kron_product(g1, vis, g2.T.conj()))  # [4]

            model_vis = transform(g1, g2, vis)  # [cal_dirs, num_rows, num_chan, 4]
            model_vis = jnp.sum(model_vis, axis=0)  # [num_rows, num_chan, 4]

            vis_variance = 1. / vis_data.weights  # Should probably use measurement set SIGMA here
            vis_stddev = jnp.sqrt(vis_variance)
            obs_dist_real = tfpd.Normal(*promote_pytree('vis_real', (vis_data.vis.real, vis_stddev)))
            obs_dist_imag = tfpd.Normal(*promote_pytree('vis_imag', (vis_data.vis.imag, vis_stddev)))
            log_prob = obs_dist_real.log_prob(model_vis.real) + obs_dist_imag.log_prob(
                model_vis.imag)  # [num_rows, num_chan, 4]

            # Mask out flagged data or zero-weighted data.
            log_prob = jnp.where(jnp.bitwise_or(vis_data.weights == 0, vis_data.flags), -jnp.inf, log_prob)

            return jnp.sum(log_prob)

        return _log_likelihood

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
    def solve(self, freqs: jax.Array, preapply_gains: jax.Array, init_params: CalibrationParams,
              vis_data: VisibilityData,
              vis_coords: VisibilityCoords) -> Tuple[CalibrationParams, jaxopt.OptStep]:

        log_prob_fn = self._build_log_likelihood(freqs=freqs,
                                                 preapply_gains=preapply_gains, vis_data=vis_data,
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
            jit=False,
            unroll=False,
            use_gamma=True
        )

        # Unroll ourself
        def body_fn(carry, x):
            params_flat, state = carry
            params_flat, state = solver.update(params=params_flat, state=state)
            return (params_flat, state), state.value

        carry = (init_params_flat, solver.init_state(init_params=init_params_flat))

        (params_flat, _), results = lax.scan(body_fn, carry, xs=jnp.arange(self.num_iterations))

        params = unravel_fn(params_flat)
        return params, results
