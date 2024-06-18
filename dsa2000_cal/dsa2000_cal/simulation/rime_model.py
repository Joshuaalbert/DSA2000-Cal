import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.typing import SupportsDType

from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.forward_model.sky_model import SkyModel
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords
from dsa2000_cal.predict.fft_stokes_I_predict import FFTStokesIPredict, FFTStokesIModelData
from dsa2000_cal.predict.gaussian_predict import GaussianPredict, GaussianModelData
from dsa2000_cal.predict.point_predict import PointPredict, PointModelData
from dsa2000_cal.source_models.corr_translation import stokes_to_linear, flatten_coherencies


@dataclasses.dataclass(eq=False)
class RIMEModel:
    # source models to simulate. Each source gets a gain direction in the flux weighted direction.
    sky_model: SkyModel

    convention: str = 'casa'
    dtype: SupportsDType = jnp.complex64

    def _stokes_I_to_linear_jax(self, image_I: jax.Array) -> jax.Array:
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
        image_stokes = jnp.stack([image_I, zero, zero, zero], axis=-1)  # [_, 4]
        image_linear = jax.vmap(partial(stokes_to_linear, flat_output=False))(image_stokes)  # [_, 2, 2]
        return lax.reshape(image_linear, shape + (2, 2))  # [..., 2, 2]

    def predict_model_visibilities_jax(self, freqs: jax.Array, apply_gains: jax.Array,
                                       vis_coords: VisibilityCoords,
                                       flat_coherencies: bool = False) -> jax.Array:
        """
        Simulate visibilities for a set of source models.

        Args:
            freqs: [num_chans]
            apply_gains: [num_cal, num_time, num_ant, num_chan, 2, 2] in linear units
            vis_coords: [num_row] visibility coordinates
            flat_coherencies: whether to return the visibilities as a flat coherencies

        Returns:
            vis: [num_source_models, num_row, num_chans, 2, 2]
                or  [num_source_models, num_row, num_chans, 4] if flat_output is True
        """
        num_rows, _ = np.shape(vis_coords.uvw)
        num_freqs = np.shape(freqs)[0]
        vis = jnp.zeros((self.sky_model.num_sources, num_rows, num_freqs, 2, 2),
                        self.dtype)  # [cal_dirs, num_rows, num_chans, 2, 2]
        # Predict the visibilities with pre-applied gains
        dft_predict = PointPredict(convention=self.convention,
                                   dtype=self.dtype)
        gaussian_predict = GaussianPredict(convention=self.convention,
                                           dtype=self.dtype)
        faint_predict = FFTStokesIPredict(convention=self.convention,
                                          dtype=self.dtype)

        # Each calibrator has a model which is a collection of sources that make up the calibrator.
        cal_idx = 0
        for wsclean_source_model in self.sky_model.component_models:
            preapply_gains_cal = apply_gains[cal_idx]  # [num_time, num_ant, num_chan, 2, 2]

            if wsclean_source_model.point_source_model is not None:
                # Points
                l0 = quantity_to_jnp(wsclean_source_model.point_source_model.l0)  # [source]
                m0 = quantity_to_jnp(wsclean_source_model.point_source_model.m0)  # [source]
                n0 = jnp.sqrt(1. - l0 ** 2 - m0 ** 2)  # [source]

                lmn = jnp.stack([l0, m0, n0], axis=-1)  # [source, 3]
                image_I = quantity_to_jnp(wsclean_source_model.point_source_model.A, 'Jy')  # [source, chan]
                image_linear = self._stokes_I_to_linear_jax(image_I)  # [source, chan, 2, 2]

                dft_model_data = PointModelData(
                    gains=preapply_gains_cal,  # [num_time, num_ant, num_chan, 2, 2]
                    lmn=lmn,
                    image=image_linear
                )
                vis = vis.at[cal_idx, ...].set(
                    dft_predict.predict(
                        freqs=freqs,
                        dft_model_data=dft_model_data,
                        visibility_coords=vis_coords
                    )  # [num_rows, num_chans, 2, 2]
                )

            if wsclean_source_model.gaussian_source_model is not None:
                # Gaussians
                l0 = quantity_to_jnp(wsclean_source_model.gaussian_source_model.l0)  # [source]
                m0 = quantity_to_jnp(wsclean_source_model.gaussian_source_model.m0)  # [source]
                n0 = jnp.sqrt(1. - l0 ** 2 - m0 ** 2)  # [source]

                lmn = jnp.stack([l0, m0, n0], axis=-1)  # [source, 3]
                image_I = quantity_to_jnp(wsclean_source_model.gaussian_source_model.A, 'Jy')  # [source, chan]
                image_linear = self._stokes_I_to_linear_jax(image_I)  # [source, chan, 2, 2]

                ellipse_params = jnp.stack([
                    quantity_to_jnp(wsclean_source_model.gaussian_source_model.major),
                    quantity_to_jnp(wsclean_source_model.gaussian_source_model.minor),
                    quantity_to_jnp(wsclean_source_model.gaussian_source_model.theta)
                ],
                    axis=-1)  # [source, 3]

                gaussian_model_data = GaussianModelData(
                    image=image_linear,  # [source, chan, 2, 2]
                    gains=preapply_gains_cal,  # [num_time, num_ant, num_chan, 2, 2]
                    ellipse_params=ellipse_params,  # [source, 3]
                    lmn=lmn  # [source, 3]
                )

                vis = vis.at[cal_idx].add(
                    gaussian_predict.predict(
                        freqs=freqs,  # [chan]
                        gaussian_model_data=gaussian_model_data,  # [source, chan, 2, 2]
                        visibility_coords=vis_coords  # [row, 3]
                    ),  # [num_rows, num_chans, 2, 2]
                    indices_are_sorted=True,
                    unique_indices=True
                )

            cal_idx += 1

        for fits_source_model in self.sky_model.fits_models:
            preapply_gains_cal = apply_gains[cal_idx]  # [num_time, num_ant, num_chan, 2, 2]
            l0 = quantity_to_jnp(fits_source_model.l0)  # [num_chan]
            m0 = quantity_to_jnp(fits_source_model.m0)  # [num_chan]
            dl = quantity_to_jnp(fits_source_model.dl)  # [num_chan]
            dm = quantity_to_jnp(fits_source_model.dm)  # [num_chan]
            image = jnp.stack(
                [
                    quantity_to_jnp(img, 'Jy')
                    for img in fits_source_model.images
                ],
                axis=0
            )  # [num_chan, Nl, Nm]

            faint_model_data = FFTStokesIModelData(
                image=image,  # [num_chan, Nl, Nm]
                gains=preapply_gains_cal,  # [num_time, num_ant, num_chan, 2, 2]
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
            cal_idx += 1

        if flat_coherencies:
            vis = jax.vmap(jax.vmap(jax.vmap(flatten_coherencies)))(vis)  # [num_sources, num_rows, num_freqs, 4]
        return vis
