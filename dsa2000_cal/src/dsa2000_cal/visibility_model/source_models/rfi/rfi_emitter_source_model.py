import dataclasses
from functools import partial
from typing import NamedTuple, Tuple

import jax
import numpy as np
import pylab as plt
from astropy import constants as const, units as au
from jax import numpy as jnp

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.assets import RFIEmitterSourceModelParams, AbstractRFIEmitterData
from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.delay_models.near_field import NearFieldDelayEngine
from dsa2000_cal.visibility_model.source_models.rfi.parametric_rfi_emitter import ParametricDelayACF


class RFIEmitterModelData(NamedTuple):
    freqs: jax.Array  # [chan]
    position_enu: jax.Array  # [E, 3]
    delay_acf: InterpolatedArray | ParametricDelayACF  # [E,chan,[2,2]]
    gains: jax.Array | None  # [[E,] time, ant, chan[, 2, 2]]


@dataclasses.dataclass(eq=False)
class RFIEmitterSourceModel(AbstractSourceModel):
    """
    Predict vis for point source.
    """
    params: RFIEmitterSourceModelParams

    @property
    def num_emitters(self):
        return self.params.delay_acf.shape[0]

    @staticmethod
    def from_rfi_model(rfi_model: AbstractRFIEmitterData, freqs: au.Quantity, central_freq: au.Quantity | None = None,
                       full_stokes: bool = True) -> 'RFIEmitterSourceModel':
        """
        Create a source model from an RFI model.

        Args:
            rfi_model: the RFI model
            freqs: the frequencies to evaluate the model at
            central_freq: the central frequency of the model
            full_stokes: whether to create a full stokes model

        Returns:
            source_model: the source model
        """
        return RFIEmitterSourceModel(
            params=rfi_model.make_source_params(freqs=freqs, full_stokes=full_stokes, central_freq=central_freq)
        )

    def is_full_stokes(self) -> bool:
        return len(self.params.delay_acf.shape) == 4 and self.params.delay_acf.shape[-2:] == (
            2, 2)

    def get_model_data(self, gains: jax.Array | None) -> RFIEmitterModelData:
        """
        Get the model data for the source models. Optionally pre-apply gains in model.

        Args:
            gains: [[E,] time, ant, chan[, 2, 2]] the gains to apply to the source model

        Returns:
            model_data: the model data
        """
        return RFIEmitterModelData(
            freqs=mp_policy.cast_to_freq(quantity_to_jnp(self.params.freqs)),
            position_enu=mp_policy.cast_to_length(quantity_to_jnp(self.params.position_enu)),
            delay_acf=self.params.delay_acf,
            gains=mp_policy.cast_to_gain(gains)
        )

    def get_source_positions_enu(self) -> jax.Array:
        return quantity_to_jnp(self.params.position_enu)

    def plot(self, save_file: str = None):
        """
        Plot the source model.

        Args:
            save_file: the file to save the plot to
        """
        # Plot each emitter in ENU coordinates
        freq_idx = len(self.params.freqs) // 2
        fig, axs = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
        # Plot array centre at 0,0 in red
        axs[0, 0].scatter(0, 0, c='r', label='Array Centre')
        for emitter in range(self.num_emitters):
            sc = axs[0, 0].scatter(self.params.position_enu[emitter, 0], self.params.position_enu[emitter, 1],
                                   marker='*')
        plt.colorbar(sc, ax=axs[0, 0], label='Luminosity [Jy km^2]')
        axs[0, 0].set_xlabel('East [m]')
        axs[0, 0].set_ylabel('North [m]')
        axs[0, 0].set_title('RFI Emitter Source Model')
        axs[0, 0].legend()
        if save_file is not None:
            plt.savefig(save_file)
        plt.show()


@dataclasses.dataclass(eq=False)
class RFIEmitterPredict:
    delay_engine: NearFieldDelayEngine
    convention: str = 'physical'

    def check_predict_inputs(self, model_data: RFIEmitterModelData
                             ) -> Tuple[bool, bool, bool]:
        """
        Check the inputs for predict.

        Args:
            model_data: data, see above for shape info.

        Returns:
            full_stokes: bool
            is_gains: bool
            direction_dependent_gains: bool
        """
        full_stokes = len(model_data.delay_acf.shape) == 4 and model_data.delay_acf.shape[-2:] == (2, 2)
        E, _ = np.shape(model_data.position_enu)
        num_freqs = len(model_data.freqs)
        is_gains = model_data.gains is not None
        if full_stokes:
            if model_data.delay_acf.shape != (E, num_freqs, 2, 2):
                raise ValueError(f"ACF must be [E, num_chans, 2, 2], got {model_data.delay_acf.shape}")
            if is_gains:  # [[E,] time, ant, chan[, 2, 2]]
                if np.shape(model_data.gains)[-3] != len(model_data.freqs):
                    raise ValueError(
                        f"Gains must have the same number of channels as freqs {len(model_data.freqs)}, got {np.shape(model_data.gains)}")
                if len(np.shape(model_data.gains)) == 6 and np.shape(model_data.gains)[-2:] == (2, 2):
                    direction_dependent_gains = True
                elif len(np.shape(model_data.gains)) == 5 and np.shape(model_data.gains)[-2:] == (2, 2):
                    direction_dependent_gains = False
                else:
                    raise ValueError(
                        f"Gains must be [[E,] time, ant, chan, 2, 2], got {np.shape(model_data.gains)}"
                    )
            else:
                direction_dependent_gains = False
        else:
            if model_data.delay_acf.shape != (E, num_freqs):
                raise ValueError(f"ACF must be [E, num_chans], got {model_data.delay_acf.shape}")
            if is_gains:  # [[E,] time, ant, chan]
                if np.shape(model_data.gains)[-1] != len(model_data.freqs):
                    raise ValueError(
                        f"Gains must have the same number of channels as freqs {len(model_data.freqs)}, got {np.shape(model_data.gains)}")
                if len(np.shape(model_data.gains)) == 4:
                    direction_dependent_gains = True
                elif len(np.shape(model_data.gains)) == 3:
                    direction_dependent_gains = False
                else:
                    raise ValueError(
                        f"Gains must be [[E,] time, ant, chan], got {np.shape(model_data.gains)}"
                    )

            else:
                direction_dependent_gains = False

        return full_stokes, is_gains, direction_dependent_gains

    def predict(self, model_data: RFIEmitterModelData, visibility_coords: VisibilityCoords) -> jax.Array:
        """
        Predict visibilities from DFT model data.

        Args:
            model_data: data, see above for shape info.
            visibility_coords: visibility coordinates.

        Returns:
            visibilities: [row, chan[, 2, 2]] in linear correlation basis.
        """
        full_stokes, is_gains, direction_dependent_gains = self.check_predict_inputs(model_data)
        if full_stokes:
            out_mapping = "[e,r,c,~p,~q]"
            acf_values_mapping = "[x,e,c,p,q]"
            spectral_power_mapping = "[e,p,q]"
        else:
            out_mapping = "[e,r,c]"
            acf_values_mapping = "[x,e,c]"
            spectral_power_mapping = "[e]"

        if is_gains:

            _t = visibility_coords.time_idx
            _a1 = visibility_coords.antenna_1
            _a2 = visibility_coords.antenna_2

            if direction_dependent_gains:
                if full_stokes:
                    g1 = model_data.gains[:, _t, _a1, :, :, :]
                    g2 = model_data.gains[:, _t, _a2, :, :, :]
                    g_mapping = "[e,r,c,p,q]"
                else:
                    g1 = model_data.gains[:, _t, _a1, :]
                    g2 = model_data.gains[:, _t, _a2, :]
                    g_mapping = "[e,r,c]"
            else:
                if full_stokes:
                    g1 = model_data.gains[_t, _a1, :, :, :]
                    g2 = model_data.gains[_t, _a2, :, :, :]
                    g_mapping = "[r,c,p,q]"
                else:
                    g1 = model_data.gains[_t, _a1, :]
                    g2 = model_data.gains[_t, _a2, :]
                    g_mapping = "[r,c]"
        else:
            g1 = None
            g2 = None
            g_mapping = "[]"

        @partial(multi_vmap,
                 in_mapping=f"[c],[r],[r],[r],[r],{g_mapping},{g_mapping},[e,3],{acf_values_mapping}",
                 out_mapping=out_mapping,
                 verbose=True
                 )
        def compute_phase_from_projection_jax_from_interpolated_array(freq, t1, i1, i2, w, g1, g2, position_enu,
                                                                      acf_values):
            """
            Compute the delay from the projection.

            Args:
                t1: time index
                i1: antenna 1 index
                i2: antenna 2 index
                w: w coordinate
                g1: [] or [2,2]
                g2: [] or [2,2]
                position_enu: [3]
                acf_values: [num_x]
            """
            # propagation delay
            delay, dist20, dist10 = self.delay_engine.compute_delay_from_projection_jax(
                a_east=position_enu[0],
                a_north=position_enu[1],
                a_up=position_enu[2],
                t1=t1,
                i1=i1,
                i2=i2
            )  # [], [], []

            # jax.debug.print("delay={delay}", delay=delay)
            # jax.debug.print("dist20={dist20}", dist20=dist20)
            # jax.debug.print("dist10={dist10}", dist10)

            # ACF delay -- rebuild from sharded data
            delay_acf = InterpolatedArray(
                x=model_data.delay_acf.x,
                values=acf_values,
                axis=0,
                regular_grid=model_data.delay_acf.regular_grid
            )  # [[2,2]]
            delay_s = mp_policy.cast_to_time(delay / quantity_to_jnp(const.c))
            delay_acf_val = delay_acf(x=delay_s)  # [[2,2]]
            return apply_delay(
                freq, delay, w, delay_acf_val, dist10, dist20, g1, g2
            )

        @partial(
            multi_vmap,
            in_mapping=f"[c],[r],[r],[r],[r],{g_mapping},{g_mapping},[e,3],[e],[e],{spectral_power_mapping},[c],[c]",
            out_mapping=out_mapping,
            verbose=True
        )
        def compute_phase_from_projection_jax_from_parametric_acf(freq, t1, i1, i2, w, g1, g2, position_enu,
                                                                  mu, fwhp, spectral_power, channel_lower,
                                                                  channel_upper):
            """
            Compute the delay from the projection.

            Args:
                t1: time index
                i1: antenna 1 index
                i2: antenna 2 index
                w: w coordinate
                g1: [] or [2,2]
                g2: [] or [2,2]
                position_enu: [3]
                mu: []
                fwhp: []
                spectral_power: [[2,2]]
                channel_lower: []
                channel_upper: []
            """
            # propagation delay
            delay, dist20, dist10 = self.delay_engine.compute_delay_from_projection_jax(
                a_east=position_enu[0],
                a_north=position_enu[1],
                a_up=position_enu[2],
                t1=t1,
                i1=i1,
                i2=i2
            )  # [], [], []

            # jax.debug.print("delay={delay}", delay=delay)
            # jax.debug.print("dist20={dist20}", dist20=dist20)
            # jax.debug.print("dist10={dist10}", dist10)

            # ACF delay -- rebuild from sharded data
            delay_acf = ParametricDelayACF(
                mu=mu[None],
                fwhp=fwhp[None],
                spectral_power=spectral_power[None],
                channel_lower=channel_lower[None],
                channel_upper=channel_upper[None],
                resolution=model_data.delay_acf.resolution,
                convention=model_data.delay_acf.convention
            )  # [E=1,c=1,[2,2]]
            delay_s = mp_policy.cast_to_time(delay / quantity_to_jnp(const.c))
            delay_acf_val = delay_acf(delay_s)[0, 0]  # [[2,2]]
            return apply_delay(
                freq, delay, w, delay_acf_val, dist10, dist20, g1, g2
            )

        def apply_delay(freq, delay, w, delay_acf_val, dist10, dist20, g1, g2):
            wavelength = quantity_to_jnp(const.c) / freq  # []
            # delay ~ l*u + m*v + n*w
            # -2j pi delay / wavelength + 2j pi w / wavelength = -2j pi (delay - w) / wavelength
            # = -2j pi (l*u + m*v + n*w - w) / wavelength
            # = -2j pi (l*u + m*v + (n-1)*w) / wavelength
            phase = -2j * jnp.pi * delay / wavelength  # []

            # e^(-2j pi w) so that get -2j pi w (n-1) term
            tracking_delay = 2j * jnp.pi * w / wavelength  # []
            phase += tracking_delay

            if self.convention == 'engineering':
                phase = jnp.negative(phase)

            # fields decrease with 1/r
            visibilities = (
                    (delay_acf_val * jnp.reciprocal(dist10) * jnp.reciprocal(dist20)) * jnp.exp(phase)
            )  # [[2, 2]]

            if is_gains:
                if full_stokes:
                    visibilities = kron_product(g1, visibilities, g2.conj().T)  # [2, 2]
                else:
                    visibilities = g1 * visibilities * g2.conj().T  # []

            return mp_policy.cast_to_vis(visibilities)  # [[2,2]]

        if isinstance(model_data.delay_acf, InterpolatedArray):
            vis = compute_phase_from_projection_jax_from_interpolated_array(
                model_data.freqs,
                visibility_coords.time_obs,
                visibility_coords.antenna_1,
                visibility_coords.antenna_2,
                visibility_coords.uvw[:, 2],
                g1,
                g2,
                model_data.position_enu,
                model_data.delay_acf.values
            )  # [E, num_chans[,2,2]]
        elif isinstance(model_data.delay_acf, ParametricDelayACF):
            vis = compute_phase_from_projection_jax_from_parametric_acf(
                model_data.freqs,
                visibility_coords.time_obs,
                visibility_coords.antenna_1,
                visibility_coords.antenna_2,
                visibility_coords.uvw[:, 2],
                g1,
                g2,
                model_data.position_enu,
                model_data.delay_acf.mu,
                model_data.delay_acf.fwhp,
                model_data.delay_acf.spectral_power,
                model_data.delay_acf.channel_lower,
                model_data.delay_acf.channel_upper
            )  # [E, num_chans[,2,2]]
        else:
            raise ValueError(f"Invalid delay_acf type {type(model_data.delay_acf)}")
        return jnp.sum(vis, axis=0)  # [num_rows, num_chans[, 2, 2]]
