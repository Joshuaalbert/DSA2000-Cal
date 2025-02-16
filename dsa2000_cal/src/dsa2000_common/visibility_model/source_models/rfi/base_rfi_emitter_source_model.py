import dataclasses
import pickle
import warnings
from functools import partial
from typing import NamedTuple, Tuple, List, Any

import jax
import numpy as np
import pylab as plt
from astropy import constants as const
from jax import numpy as jnp

from dsa2000_common.common.array_types import ComplexArray, FloatArray
from dsa2000_common.common.jax_utils import multi_vmap
from dsa2000_common.common.mixed_precision_utils import mp_policy
from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_common.common.types import VisibilityCoords
from dsa2000_common.common.vec_utils import kron_product
from dsa2000_common.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_common.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_common.gain_models.gain_model import GainModel
from dsa2000_common.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_common.visibility_model.source_models.abc import AbstractSourceModel
from dsa2000_common.visibility_model.source_models.rfi.abc import AbstractRFIAutoCorrelationFunction


class RFIEmitterModelData(NamedTuple):
    position_enu: FloatArray  # [E, 3]
    delay_acf: AbstractRFIAutoCorrelationFunction  # [E[,2, 2]]


@dataclasses.dataclass(eq=False)
class RFIEmitterSourceModel(AbstractSourceModel):
    """
    Predict vis for point source.

    Args:
        position_enu: [E, 3] ENU coordinates of the emitters
        delay_acf: (t) -> [E,[2,2]] Delay ACF for each emitter
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

    def predict(self, visibility_coords: VisibilityCoords, gain_model: GainModel | None,
                near_field_delay_engine: BaseNearFieldDelayEngine, far_field_delay_engine: BaseFarFieldDelayEngine,
                geodesic_model: BaseGeodesicModel) -> ComplexArray:

        _a1 = visibility_coords.antenna1  # [B]
        _a2 = visibility_coords.antenna2  # [B]

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

            if gain_model is not None:
                lmn_geodesic = geodesic_model.compute_near_field_geodesics(
                    times=time[None],
                    source_positions_enu=self.position_enu
                )  # [1, num_ant, num_sources, 3]
                # Compute the gains
                gains = gain_model.compute_gain(
                    freqs=freq[None],
                    times=time[None],
                    lmn_geodesic=lmn_geodesic,
                )  # [1, num_ant, 1, num_sources,[, 2, 2]]
                g1 = gains[0, visibility_coords.antenna1, 0, :, ...]  # [B, num_sources[, 2, 2]]
                g2 = gains[0, visibility_coords.antenna2, 0, :, ...]  # [B, num_sources[, 2, 2]]
                if self.is_full_stokes():
                    gain_mapping = "[B,S,2,2]"
                else:
                    gain_mapping = "[B,S]"
            else:
                g1 = g2 = None
                gain_mapping = "[]"

            if self.is_full_stokes():
                out_mapping = "[B,~P,~Q]"
            else:
                out_mapping = "[B]"

            @partial(
                multi_vmap,
                in_mapping=f"[B,3],{gain_mapping},{gain_mapping},[B],[B]",
                out_mapping=out_mapping,
                verbose=True
            )
            def compute_visibilities_rfi_over_sources(uvw, g1, g2, i1, i2):
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
                    a_east=self.position_enu[:, 0],
                    a_north=self.position_enu[:, 1],
                    a_up=self.position_enu[:, 2]
                )  # [S, 3]

                delay, dist20, dist10 = near_field_delay_engine.compute_delay(
                    x_0_gcrs=x_0_gcrs,
                    t1=time,
                    i1=i1,
                    i2=i2
                )  # [S], [S], [S]
                c = quantity_to_jnp(const.c)

                delay_acf_val = self.delay_acf.eval(freq=freq, tau=mp_policy.cast_to_time(delay / c))  # [S,[2,2]]

                def body(accumulate, x):
                    (delay, delay_acf_val, dist10, dist20, g1, g2) = x

                    vis = self._compute_visibilty(
                        freq=freq,
                        delay=delay,
                        w=uvw[2],
                        delay_acf_val=delay_acf_val,
                        dist10=dist10,
                        dist20=dist20,
                        g1=g1,
                        g2=g2
                    )  # [[,2,2]]
                    return accumulate + vis, None

                accumulate = jnp.zeros(
                    (2, 2) if self.is_full_stokes() else (),
                    dtype=mp_policy.vis_dtype
                )

                vis_accumulation, _ = jax.lax.scan(
                    body,
                    accumulate,
                    (delay, delay_acf_val, dist10, dist20, g1, g2)
                )
                return vis_accumulation

            return compute_visibilities_rfi_over_sources(uvw, g1, g2, _a1, _a2)

        visibilities = compute_baseline_visibilities_point(
            visibility_coords.uvw,
            visibility_coords.freqs,
            visibility_coords.times
        )  # [num_times, num_baselines, num_freqs[,2, 2]]
        return visibilities

    def _compute_visibilty(self, freq, delay, w, delay_acf_val: FloatArray, dist10: FloatArray, dist20: FloatArray,
                           g1, g2):
        """
        Compute the visibility from a single direction for a single baseline.

        Args:
            freq: []
            delay: []
            w: []
            delay_acf_val: [[,2, 2]]
            dist10: []
            dist20: []
            g1: [[2, 2]]
            g2: [[2, 2]]

        Returns:
            [[,2, 2]] visibility in given direction for given baseline.
        """

        c = quantity_to_jnp(const.c)

        wavelength = c / freq  # []
        # delay ~ l*u + m*v + n*w
        # 2j pi delay / wavelength + 2j pi w / wavelength = -2j pi (delay - w) / wavelength
        # = 2j pi (l*u + m*v + n*w - w) / wavelength
        # = 2j pi (l*u + m*v + (n-1)*w) / wavelength
        phase = -2j * jnp.pi * delay / wavelength  # []

        # e^(2j pi w) so that get -2j pi w (n-1) term
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

    def save(self, filename: str):
        """
        Serialise the model to file.

        Args:
            filename: the filename
        """
        if not filename.endswith('.pkl'):
            warnings.warn(f"Filename {filename} does not end with .pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        """
        Load the model from file.

        Args:
            filename: the filename

        Returns:
            the model
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        children, aux_data = self.flatten(self)
        children_np = jax.tree.map(np.asarray, children)
        serialised = (aux_data, children_np)
        return (self._deserialise, (serialised,))

    @classmethod
    def _deserialise(cls, serialised):
        # Create a new instance, bypassing __init__ and setting the actor directly
        (aux_data, children_np) = serialised
        children_jax = jax.tree.map(jnp.asarray, children_np)
        return cls.unflatten(aux_data, children_jax)

    @classmethod
    def register_pytree(cls):
        jax.tree_util.register_pytree_node(cls, cls.flatten, cls.unflatten)

    # an abstract classmethod
    @classmethod
    def flatten(cls, this: "RFIEmitterSourceModel") -> Tuple[List[Any], Tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        return (
            [this.position_enu, this.delay_acf], (
                this.convention, this.skip_post_init
            )
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "RFIEmitterSourceModel":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        convention, skip_post_init = aux_data
        position_enu, delay_acf = children
        return RFIEmitterSourceModel(
            position_enu=position_enu,
            delay_acf=delay_acf,
            convention=convention,
            skip_post_init=skip_post_init
        )


RFIEmitterSourceModel.register_pytree()
