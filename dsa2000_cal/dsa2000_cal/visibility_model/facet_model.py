import dataclasses
from typing import NamedTuple

import jax

from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.delay_models.far_field import FarFieldDelayEngine, VisibilityCoords
from dsa2000_cal.delay_models.near_field import NearFieldDelayEngine
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.geodesic_model import GeodesicModel
from dsa2000_cal.visibility_model.source_models.celestial.fits_source_model import FITSSourceModel, \
    FITSModelData, FITSPredict
from dsa2000_cal.visibility_model.source_models.celestial.gaussian_source_model import \
    GaussianSourceModel, GaussianModelData, GaussianPredict
from dsa2000_cal.visibility_model.source_models.celestial.point_source_model import \
    PointSourceModel, PointModelData, PointPredict
from dsa2000_cal.visibility_model.source_models.rfi.rfi_emitter_source_model import \
    RFIEmitterSourceModel, RFIEmitterModelData, \
    RFIEmitterPredict


class FacetModelData(NamedTuple):
    point_model_data: PointModelData | None
    gaussian_model_data: GaussianModelData | None
    fits_model_data: FITSModelData | None
    rfi_emitter_model_data: RFIEmitterModelData | None


@dataclasses.dataclass(eq=False)
class FacetModel:
    """
    Represents a flux from a subset of the celestial sphere.
    In the case that the flux is highly localised it can be used to construct calibrators for calibration.
    Otherwise, the notion of a flux-weighted direction is not of much value.
    """
    near_field_delay_engine: NearFieldDelayEngine
    far_field_delay_engine: FarFieldDelayEngine
    geodesic_model: GeodesicModel

    point_source_model: PointSourceModel | None = None
    gaussian_source_model: GaussianSourceModel | None = None
    fits_source_model: FITSSourceModel | None = None
    rfi_emitter_source_model: RFIEmitterSourceModel | None = None
    gain_model: GainModel | None = None

    convention: str = "physical"

    def __post_init__(self):
        # Ensure at least one source model is provided
        if (
                self.point_source_model is None
                and self.gaussian_source_model is None
                and self.fits_source_model is None
                and self.rfi_emitter_source_model is None
        ):
            raise ValueError(
                "At least one of point_source_model, gaussian_source_model, "
                "fits_source_model, or lte_source_model must be provided"
            )
        # Ensure all is_full_stokes agree with each other
        is_full_stokes = []
        if self.point_source_model is not None:
            is_full_stokes.append(self.point_source_model.is_full_stokes())
        if self.gaussian_source_model is not None:
            is_full_stokes.append(self.gaussian_source_model.is_full_stokes())
        if self.fits_source_model is not None:
            is_full_stokes.append(self.fits_source_model.is_full_stokes())
        if self.rfi_emitter_source_model is not None:
            is_full_stokes.append(self.rfi_emitter_source_model.is_full_stokes())
        if self.gain_model is not None:
            is_full_stokes.append(self.gain_model.is_full_stokes())
        if len(set(is_full_stokes)) > 1:
            raise ValueError(f"All source models must be the same Stokes type, got {is_full_stokes}")
        self._is_full_stokes = is_full_stokes[0]

    def is_full_stokes(self) -> bool:
        return self._is_full_stokes

    def get_model_data(self, times: jax.Array) -> FacetModelData:
        """
        Get the model data for the source models. Optionally pre-apply gains in model.

        Args:
            times: [num_time] the times to compute the model data, in TT since start of observation.


        Returns:
            model_data: the model data
        """
        # 1. Gets the geodesics for each source model, and gets the gains for them, to use in the predict.
        # Sum up all visibilities from each source model.
        point_model_data = None
        gaussian_model_data = None
        fits_model_data = None
        rfi_emitter_model_data = None

        if self.point_source_model is not None:
            gains = None
            if self.gain_model is not None:
                # Get RA, DEC for each point source
                lmn_sources = self.point_source_model.get_lmn_sources()  # [num_sources, 3]
                geodesics = self.geodesic_model.compute_far_field_geodesic(
                    times=times,
                    lmn_sources=lmn_sources
                )  # [num_sources, num_times, num_ant, 3]
                freqs = quantity_to_jnp(self.point_source_model.freqs)
                gains = self.gain_model.compute_gain(times=times,
                                                     geodesics=geodesics,
                                                     freqs=freqs
                                                     )  # [[num_sources,] time, ant, chan[, 2, 2]]
            point_model_data = self.point_source_model.get_model_data(gains)
        if self.gaussian_source_model is not None:
            gains = None
            if self.gain_model is not None:
                lmn_sources = self.gaussian_source_model.get_lmn_sources()
                geodesics = self.geodesic_model.compute_far_field_geodesic(
                    times=times,
                    lmn_sources=lmn_sources
                )  # [num_sources, num_times, num_ant, 3]
                freqs = quantity_to_jnp(self.gaussian_source_model.freqs)
                gains = self.gain_model.compute_gain(times=times, geodesics=geodesics,
                                                     freqs=freqs)
            gaussian_model_data = self.gaussian_source_model.get_model_data(
                gains)  # [[source,] time, ant, chan[, 2, 2]]
        if self.fits_source_model is not None:
            gains = None
            if self.gain_model is not None:
                lmn_sources = self.fits_source_model.get_lmn_sources()
                geodesics = self.geodesic_model.compute_far_field_geodesic(
                    times=times,
                    lmn_sources=lmn_sources
                )  # [num_sources=1, num_times, num_ant, 3]
                freqs = quantity_to_jnp(self.fits_source_model.freqs)
                gains = self.gain_model.compute_gain(times=times,
                                                     geodesics=geodesics,
                                                     freqs=freqs)  # [num_sources=1, time, ant, chan[, 2, 2]]
                gains = gains[0]  # [time, ant, chan[, 2, 2]]
            fits_model_data = self.fits_source_model.get_model_data(gains)
        if self.rfi_emitter_source_model is not None:
            gains = None
            if self.gain_model is not None:
                source_positions_enu = self.rfi_emitter_source_model.get_source_positions_enu()
                geodesics = self.geodesic_model.compute_near_field_geodesics(
                    times=times,
                    source_positions_enu=source_positions_enu
                )  # [num_sources, num_times, num_ant, 3]
                # jax.debug.print("geodesics={geodesics}", geodesics=geodesics)
                freqs = quantity_to_jnp(self.rfi_emitter_source_model.params.freqs)
                gains = self.gain_model.compute_gain(
                    times=times,
                    geodesics=geodesics,
                    freqs=freqs
                )
                # jax.debug.print("gains={gains}", gains=gains)
            rfi_emitter_model_data = self.rfi_emitter_source_model.get_model_data(gains)
        return FacetModelData(
            point_model_data=point_model_data,
            gaussian_model_data=gaussian_model_data,
            fits_model_data=fits_model_data,
            rfi_emitter_model_data=rfi_emitter_model_data
        )

    def predict(self, model_data: FacetModelData, visibility_coords: VisibilityCoords) -> jax.Array:
        """
        Predict visibilities for all source models in facet.

        Args:
            model_data: data for predicting visibilities
            visibility_coords: visibility coordinates

        Returns:
            vis: [num_row, num_chans, [2, 2]]
        """

        # TODO: Not using far field engine yet because UVW are provided, but we could for precision,
        # e.g. in point, gaussian, etc.
        point_predict = PointPredict(
            convention=self.convention
        )
        gaussian_predict = GaussianPredict(
            order_approx=1,
            convention=self.convention
        )
        faint_predict = FITSPredict(
            epsilon=1e-6,
            convention=self.convention
        )
        lte_predict = RFIEmitterPredict(
            delay_engine=self.near_field_delay_engine,
            convention=self.convention
        )

        vis = None
        if model_data.point_model_data is not None:
            point_vis = point_predict.predict(model_data.point_model_data, visibility_coords)
            if vis is None:
                vis = point_vis
            else:
                vis += point_vis
        if model_data.gaussian_model_data is not None:
            gaussian_vis = gaussian_predict.predict(model_data.gaussian_model_data, visibility_coords)
            if vis is None:
                vis = gaussian_vis
            else:
                vis += gaussian_vis
        if model_data.fits_model_data is not None:
            fits_vis = faint_predict.predict(model_data.fits_model_data, visibility_coords)
            if vis is None:
                vis = fits_vis
            else:
                vis += fits_vis
        if model_data.rfi_emitter_model_data is not None:
            lte_vis = lte_predict.predict(model_data.rfi_emitter_model_data, visibility_coords)
            if vis is None:
                vis = lte_vis
            else:
                vis += lte_vis
        if vis is None:
            raise ValueError("No source models provided")
        return vis
