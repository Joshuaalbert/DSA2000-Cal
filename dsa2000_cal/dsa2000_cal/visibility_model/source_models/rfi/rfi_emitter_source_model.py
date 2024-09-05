import dataclasses
from functools import partial
from typing import NamedTuple, Tuple

import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
from astropy import constants as const
from jax._src.typing import SupportsDType

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.assets.rfi.rfi_emitter_model import RFIEmitterSourceModelParams, AbstractRFIEmitterData
from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.types import complex_type
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.delay_models.near_field import NearFieldDelayEngine


class RFIEmitterModelData(NamedTuple):
    freqs: jax.Array  # [num_chans]
    position_enu: jax.Array  # [E, 3]
    luminosity: jax.Array  # [E, num_chans[,2,2]]
    delay_acf: InterpolatedArray  # [E]
    gains: jax.Array | None  # [[E,] time, ant, chan[, 2, 2]]


@dataclasses.dataclass(eq=False)
class RFIEmitterSourceModel(AbstractSourceModel):
    """
    Predict vis for point source.
    """
    params: RFIEmitterSourceModelParams

    def __getitem__(self, item):
        params = RFIEmitterSourceModelParams(
            freqs=self.params.freqs,
            position_enu=self.params.position_enu[item],
            spectral_flux_density=self.params.spectral_flux_density[item],
            delay_acf=InterpolatedArray(
                x=self.params.delay_acf.x,
                values=self.params.delay_acf.values[item],
                axis=self.params.delay_acf.axis,
                regular_grid=self.params.delay_acf.regular_grid
            )
        )
        return RFIEmitterSourceModel(params)

    @property
    def num_emitters(self):
        return self.params.spectral_flux_density.shape[0]

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
        return len(self.params.spectral_flux_density.shape) == 4 and self.params.spectral_flux_density.shape[-2:] == (
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
            freqs=quantity_to_jnp(self.params.freqs),
            position_enu=quantity_to_jnp(self.params.position_enu),
            luminosity=quantity_to_jnp(self.params.spectral_flux_density, 'Jy*m^2'),
            delay_acf=self.params.delay_acf,
            gains=gains
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
            if self.is_full_stokes():
                c = self.params.spectral_flux_density[emitter, freq_idx, 0, 0].to('Jy*km^2').value
            else:
                c = self.params.spectral_flux_density[emitter, freq_idx].to('Jy*km^2').value
            sc = axs[0, 0].scatter(self.params.position_enu[emitter, 0], self.params.position_enu[emitter, 1], c=c,
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
    dtype: SupportsDType = complex_type

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
        full_stokes = len(model_data.luminosity.shape) == 4 and model_data.luminosity.shape[-2:] == (2, 2)
        E, _ = np.shape(model_data.position_enu)
        num_freqs = len(model_data.freqs)
        is_gains = model_data.gains is not None
        if full_stokes:
            if np.shape(model_data.luminosity) != (E, num_freqs, 2, 2):
                raise ValueError(f"Luminosity must be [E, num_chans, 2, 2], got {np.shape(model_data.luminosity)}")
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
            if np.shape(model_data.luminosity) != (E, num_freqs):
                raise ValueError(f"Luminosity must be [E, num_chans], got {np.shape(model_data.luminosity)}")
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
            luminosity_mapping = "[e,c,2,2]"
            out_mapping = "[e,r,c,...]"
        else:
            luminosity_mapping = "[e,c]"
            out_mapping = "[e,r,c]"

        if is_gains:

            _t = visibility_coords.time_idx
            _a1 = visibility_coords.antenna_1
            _a2 = visibility_coords.antenna_2

            if direction_dependent_gains:
                if full_stokes:
                    g1 = model_data.gains[:, _t, _a1, :, :, :]
                    g2 = model_data.gains[:, _t, _a2, :, :, :]
                    g_mapping = "[e,r,c,2,2]"
                else:
                    g1 = model_data.gains[:, _t, _a1, :]
                    g2 = model_data.gains[:, _t, _a2, :]
                    g_mapping = "[e,r,c]"
            else:
                if full_stokes:
                    g1 = model_data.gains[_t, _a1, :, :, :]
                    g2 = model_data.gains[_t, _a2, :, :, :]
                    g_mapping = "[r,c,2,2]"
                else:
                    g1 = model_data.gains[_t, _a1, :]
                    g2 = model_data.gains[_t, _a2, :]
                    g_mapping = "[r,c]"
        else:
            g1 = None
            g2 = None
            g_mapping = "[]"

        @partial(multi_vmap,
                 in_mapping=f"[c],[r],[r],[r],[r],{g_mapping},{g_mapping},{luminosity_mapping},[e,3],[x,e]",
                 out_mapping=out_mapping,
                 verbose=True
                 )
        def compute_phase_from_projection_jax(freq, t1, i1, i2, w, g1, g2, luminosity, position_enu,
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
                luminosity: [] or [2,2]
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
            )
            delay_s = delay / quantity_to_jnp(const.c)
            delay_acf_val = delay_acf(time=delay_s)  # []

            wavelength = quantity_to_jnp(const.c) / freq  # []
            # delay ~ l*u + m*v + n*w
            # -2j pi delay / wavelength + 2j pi w / wavelength = -2j pi (delay - w) / wavelength
            # = -2j pi (l*u + m*v + n*w - w) / wavelength
            # = -2j pi (l*u + m*v + (n-1)*w) / wavelength
            phase = -2j * jnp.pi * delay / wavelength  # []

            # e^(-2j pi w) so that get -2j pi w (n-1) term
            tracking_delay = 2j * jnp.pi * w / wavelength  # []
            phase += tracking_delay

            if self.convention == 'casa':
                phase = jnp.negative(phase)

            phase = phase.astype(self.dtype)

            if full_stokes:
                if is_gains:
                    luminosity = kron_product(g1, luminosity, g2.conj().T)  # [2, 2]
                # fields decrease with 1/r, and sqrt(luminosity)
                e_1 = jnp.sqrt(luminosity) * jnp.reciprocal(dist10)  # [2, 2]
                e_2 = jnp.sqrt(luminosity) * jnp.reciprocal(dist20)  # [2, 2]
                visibilities = (
                        (e_1 * e_2) * jnp.exp(phase)
                )  # [2, 2]
            else:
                if is_gains:
                    luminosity = g1 * luminosity * g2.conj().T  # []
                e_1 = jnp.sqrt(luminosity) * jnp.reciprocal(dist10)  # []
                e_2 = jnp.sqrt(luminosity) * jnp.reciprocal(dist20)  # []
                visibilities = (e_1 * e_2) * jnp.exp(phase)  # []
            visibilities *= delay_acf_val  # []
            return visibilities.astype(self.dtype)  # [num_chan[,2,2]]

        vis = compute_phase_from_projection_jax(
            model_data.freqs,
            visibility_coords.time_obs,
            visibility_coords.antenna_1,
            visibility_coords.antenna_2,
            visibility_coords.uvw[:, 2],
            g1,
            g2,
            model_data.luminosity,
            model_data.position_enu,
            model_data.delay_acf.values
        )  # [E, num_chans[,2,2]]
        return jnp.sum(vis, axis=0)  # [num_rows, num_chans[, 2, 2]]
