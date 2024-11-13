import dataclasses
from functools import partial
from typing import NamedTuple

import jax
import numpy as np
import pylab as plt
from astropy import constants as const
from jax import numpy as jnp

from dsa2000_cal.common.array_types import ComplexArray, FloatArray
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.types import VisibilityCoords
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_cal.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_cal.visibility_model.source_models.abc import AbstractSourceModel
from dsa2000_cal.visibility_model.source_models.rfi.abc import AbstractRFIAutoCorrelationFunction


class RFIEmitterModelData(NamedTuple):
    position_enu: FloatArray  # [E, 3]
    delay_acf: AbstractRFIAutoCorrelationFunction  # [E[,2, 2]]


@dataclasses.dataclass(eq=False)
class RFIEmitterSourceModel(AbstractSourceModel[RFIEmitterModelData]):
    """
    Predict vis for point source.
    """
    position_enu: FloatArray  # [E, 3]
    delay_acf: AbstractRFIAutoCorrelationFunction  # [E,[2,2]]

    convention: str = 'physical'
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return

        if len(np.shape(self.position_enu)) != 2:
            raise ValueError(f"position_enu must be 2D, got {np.shape(self.position_enu)}")

        if np.shape(self.position_enu)[0] != self.delay_acf.shape()[0]:
            raise ValueError("position_enu and delay_acf must have the same number of sources")

    def plot(self, save_file: str = None):
        """
        Plot the source model.

        Args:
            save_file: the file to save the plot to
        """
        # Plot each emitter in ENU coordinates
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # Plot array centre at 0,0 in red
        ax.scatter(0, 0, c='r', label='Array Centre')
        for emitter in range(np.shape(self.position_enu)[0]):
            ax.scatter(self.position_enu[emitter, 0], self.position_enu[emitter, 1], marker='*')
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_title('RFI Emitter Source Model')
        ax.legend()
        if save_file is not None:
            plt.savefig(save_file)
        plt.show()

    def is_full_stokes(self) -> bool:
        return len(self.delay_acf.shape()) == 3 and self.delay_acf.shape[-2:] == (2, 2)

    def get_model_slice(self, freq: FloatArray, time: FloatArray,
                        geodesic_model: BaseGeodesicModel) -> RFIEmitterModelData:
        return RFIEmitterModelData(
            position_enu=self.position_enu,
            delay_acf=self.delay_acf
        )

    def predict(self, visibility_coords: VisibilityCoords, gain_model: GainModel,
                near_field_delay_engine: BaseNearFieldDelayEngine, far_field_delay_engine: BaseFarFieldDelayEngine,
                geodesic_model: BaseGeodesicModel) -> ComplexArray:

        _a1 = visibility_coords.antenna_1  # [B]
        _a2 = visibility_coords.antenna_2  # [B]

        if self.is_full_stokes():
            out_mapping = "[T,~B,C,~P,~Q]"
        else:
            out_mapping = "[T,~B,C]"

        @partial(
            multi_vmap,
            in_mapping=f"[T,B,3],[C],[T]",
            out_mapping=out_mapping,
            scan_dims={'C'},
            verbose=True
        )
        def compute_baseline_visibilities_point(uvw, freq, time):
            """
            Compute visibilities for a single row, channel, accumulating over sources.

            Args:
                uvw: [B, 3]
                freq: []
                time: []

            Returns:
                vis_accumulation: [B, 2, 2] visibility for given baseline, accumulated over all provided directions.
            """

            model_data = self.get_model_slice(
                freq=freq,
                time=time,
                geodesic_model=geodesic_model
            )  # [num_sources, 2, 2]

            if gain_model is not None:
                lmn_geodesic = geodesic_model.compute_near_field_geodesics(
                    times=time[None],
                    source_positions_enu=model_data.position_enu
                )  # [1, num_ant, num_sources, 3]
                # Compute the gains
                gains = gain_model.compute_gain(
                    freqs=freq[None],
                    times=time[None],
                    lmn_geodesic=lmn_geodesic,
                )  # [1, num_ant, 1, num_sources,[, 2, 2]]
                g1 = gains[0, visibility_coords.antenna_1, 0, :, ...]  # [B, num_sources[, 2, 2]]
                g2 = gains[0, visibility_coords.antenna_2, 0, :, ...]  # [B, num_sources[, 2, 2]]
            else:
                g1 = g2 = None

            if self.is_full_stokes():
                gain_mapping = "[B,S,2,2]"
                out_mapping = "[B,~P,~Q]"
            else:
                gain_mapping = "[B,S]"
                out_mapping = "[B]"

            @partial(
                multi_vmap,
                in_mapping=f"[S,3],[B,3],{gain_mapping},{gain_mapping}",
                out_mapping=out_mapping,
                verbose=True
            )
            def compute_visibilities_rfi_over_sources(uvw, g1, g2):
                """
                Compute visibilities for a single direction, accumulating over sources.

                Args:
                    uvw: [3]
                    g1: [S[, 2, 2]]
                    g2: [S[, 2, 2]]

                Returns:
                    vis_accumulation: [B[, 2, 2]] visibility for given baseline, accumulated over all provided directions.
                """
                x_0_gcrs = near_field_delay_engine.construct_x_0_gcrs_from_projection(
                    a_east=model_data.position_enu[:, 0],
                    a_north=model_data.position_enu[:, 1],
                    a_up=model_data.position_enu[:, 2]
                )  # [S, 3]

                delay, dist20, dist10 = near_field_delay_engine.compute_delay(
                    x_0_gcrs=x_0_gcrs,
                    t1=time,
                    i1=_a1,
                    i2=_a2
                )  # [S], [S], [S]

                vis = self._single_compute_visibilty(
                    freq=freq,
                    delay=delay,
                    w=uvw[2],
                    delay_acf=model_data.delay_acf,
                    dist10=dist10,
                    dist20=dist20,
                    g1=g1,
                    g2=g2
                )  # [S[,2,2]]

                return jnp.sum(vis, axis=0)  # [[2,2]]

            return compute_visibilities_rfi_over_sources(model_data.lmn, uvw, g1, g2, )

        visibilities = compute_baseline_visibilities_point(
            visibility_coords.uvw,
            visibility_coords.freqs,
            visibility_coords.times
        )  # [num_times, num_baselines, num_freqs[,2, 2]]
        return visibilities

    def _single_compute_visibilty(self, freq, delay, w, delay_acf: AbstractRFIAutoCorrelationFunction, dist10, dist20,
                                  g1, g2):
        """
        Compute the visibility from a single direction for a single baseline.

        Args:
            freq: []
            delay: []
            w: []
            delay_acf: [[2, 2]]
            dist10: []
            dist20: []
            g1: [[2, 2]]
            g2: [[2, 2]]

        Returns:
            [[2, 2]] visibility in given direction for given baseline.
        """

        delay_s = mp_policy.cast_to_time(delay / quantity_to_jnp(const.c))

        delay_acf_val = delay_acf.eval(freq=freq, tau=delay_s)

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
            (delay_acf_val * jnp.exp(phase) * jnp.reciprocal(dist10) * jnp.reciprocal(dist20))
        )  # [[2, 2]]

        if g1 is not None and g2 is not None:
            if self.is_full_stokes():
                visibilities = kron_product(g1, visibilities, g2.conj().T)  # [2, 2]
            else:
                visibilities = g1 * visibilities * g2.conj().T  # []

        return mp_policy.cast_to_vis(visibilities)  # [[2,2]]


def RFIEmitterSourceModel_flatten(m: RFIEmitterSourceModel):
    return (
        [m.position_enu, m.delay_acf], (
            m.convention, m.skip_post_init
        )
    )


def RFIEmitterSourceModel_unflatten(aux_data, children):
    convention, skip_post_init = aux_data
    position_enu, delay_acf = children
    return RFIEmitterSourceModel(
        position_enu=position_enu,
        delay_acf=delay_acf,
        convention=convention,
        skip_post_init=skip_post_init
    )


jax.tree_util.register_pytree_node(
    RFIEmitterSourceModel,
    RFIEmitterSourceModel_flatten,
    RFIEmitterSourceModel_unflatten
)
