import dataclasses
from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.geodesics.geodesic_model import GeodesicModel
from dsa2000_cal.delay_models.far_field import VisibilityCoords, FarFieldDelayEngine
from dsa2000_cal.delay_models.near_field import NearFieldDelayEngine
from dsa2000_cal.visibility_model.facet_model import FacetModel, FacetModelData


@dataclasses.dataclass(eq=False)
class RIMEModel:
    # source models to simulate. Each source gets a gain direction in the flux weighted direction.
    facet_models: List[FacetModel]

    def __post_init__(self):
        if len(self.facet_models) == 0:
            raise ValueError("At least one source model must be provided.")

        # Assumes all source models have the same engines
        self.near_field_delay_engine: NearFieldDelayEngine = self.facet_models[0].near_field_delay_engine
        self.far_field_delay_engine: FarFieldDelayEngine = self.facet_models[0].far_field_delay_engine
        self.geodesic_model: GeodesicModel = self.facet_models[0].geodesic_model
        self.dtype = self.facet_models[0].dtype
        self.convention = self.facet_models[0].convention

    @property
    def num_facets(self) -> int:
        return len(self.facet_models)

    @property
    def num_antennas(self) -> int:
        return len(self.geodesic_model.antennas)

    def predict_visibilities(self, model_data: List[FacetModelData], visibility_coords: VisibilityCoords) -> jax.Array:
        """
        Predict visibilities for a set of source models.

        Args:
            model_data: list of model data for each facet
            visibility_coords: visibility coordinates

        Returns:
            vis: [num_source_models, num_row, num_chans[, 2, 2]] depending on the predict data stokes type.
        """
        vis = []
        for model, model_data in zip(self.facet_models, model_data):
            vis.append(model.predict(model_data=model_data, visibility_coords=visibility_coords))
        return jnp.stack(vis, axis=0)

    def get_model_data(self, times: jax.Array) -> List[FacetModelData]:
        """
        Get the model data for the source models. Optionally pre-apply gains in model.

        Args:
            times: [num_time] the times to compute the model data, in TT since start of observation.

        Returns:
            model_data: the model data
        """
        model_data = []
        for facet_model in self.facet_models:
            model_data.append(facet_model.get_model_data(times=times))
        return model_data

    def predict_facets_model_visibilities(self, times: jax.Array, with_autocorr: bool = True) -> Tuple[
        jax.Array, VisibilityCoords]:
        """
        Simulate visibilities for a set of source models, creating one row of visibilities per data model.
        These correspond to facets. Order is first celestial, then RFI facets.

        Args:
            times: [num_times] times to predict visibilities for
            with_autocorr: whether to include autocorrelations

        Returns:
            vis: [num_source_models, num_row, num_chans[, 2, 2]] depending on the predict data stokes type.
            visibility_coords: [num_row] visibility coordinates
        """
        # Predict the visibilities with pre-applied gains
        visibility_coords = self.facet_models[0].far_field_delay_engine.compute_visibility_coords(
            times=times, with_autocorr=with_autocorr)
        vis = []
        for facet_model in self.facet_models:
            facet_model_data = facet_model.get_model_data(times=times)
            vis.append(facet_model.predict(model_data=facet_model_data, visibility_coords=visibility_coords))
        return jnp.stack(vis, axis=0), visibility_coords

    @staticmethod
    def apply_gains(gains: jax.Array, vis: jax.Array, visibility_coords: VisibilityCoords) -> jax.Array:
        """
        Apply gains to the source models.

        Args:
            gains: [num_facets, num_time, num_ant, num_chan[, 2, 2]] gains to apply
            vis: [num_facets, num_row, num_chans[, 2, 2]] visibilities to apply gains to
            visibility_coords: [num_row] visibility coordinates

        Returns:
            vis: [num_row, num_chans[, 2, 2]] visibilities with gains applied
        """
        g1 = gains[:, visibility_coords.antenna_1, visibility_coords.time_idx, :, ...]
        g2 = gains[:, visibility_coords.antenna_2, visibility_coords.time_idx, :, ...]
        if len(np.shape(gains)) == 6:
            gains_mapping = "[s,r,f,2,2]"
            vis_mapping = "[s,r,f,2,2]"
            is_full_stokes = True
        elif len(np.shape(gains)) == 4:
            gains_mapping = "[s,r,f]"
            vis_mapping = "[s,r,f]"
            is_full_stokes = False
        else:
            raise ValueError("Gains must be of shape [num_facets, num_time, num_ant, num_chan[, 2, 2]] or "
                             "[num_facets, num_time, num_ant, num_chan]")

        @partial(
            multi_vmap,
            in_mapping=f"{gains_mapping},{gains_mapping},{vis_mapping}",
            out_mapping="[r,f,...]",
            verbose=True
        )
        def apply(g1: jax.Array, g2: jax.Array, vis: jax.Array) -> jax.Array:

            def body_fn(accumulate, x):
                g1, g2, vis = x
                if is_full_stokes:
                    delta = kron_product(g1, vis, g2.conj().T)
                else:
                    delta = g1 * vis * g2.conj()
                return accumulate + delta, ()

            if is_full_stokes:
                init = jnp.zeros((2, 2), vis.dtype)
            else:
                init = jnp.zeros((), vis.dtype)
            vis_accumulate, _ = lax.scan(body_fn, init, (g1, g2, vis), unroll=2)
            return vis_accumulate

        return apply(g1, g2, vis)
