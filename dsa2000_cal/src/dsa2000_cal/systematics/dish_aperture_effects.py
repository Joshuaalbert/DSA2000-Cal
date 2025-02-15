import dataclasses
import pickle
import warnings
from typing import Tuple, List, Any

import astropy.units as au
import jax.lax
import jax.numpy as jnp
import numpy as np

from dsa2000_cal.common.array_types import FloatArray
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_common.gain_models import BaseSphericalInterpolatorGainModel
from dsa2000_common.geodesics.base_geodesic_model import BaseGeodesicModel


@dataclasses.dataclass(eq=False)
class DishApertureEffects:
    """
    Applies a perturbation to the dish aperture, and computes the resulting far-field beam.
    """
    dish_diameter: FloatArray  # in m
    focal_length: FloatArray  # in m

    elevation_pointing_error_stddev: FloatArray | None = None  # in rad
    cross_elevation_pointing_error_stddev: FloatArray | None = None  # in rad
    axial_focus_error_stddev: FloatArray | None = None  # in m
    elevation_feed_offset_stddev: FloatArray | None = None  # in m
    cross_elevation_feed_offset_stddev: FloatArray | None = None  # in m
    horizon_peak_astigmatism_stddev: FloatArray | None = None  # in m
    surface_error_mean: FloatArray | None = None  # in m
    surface_error_stddev: FloatArray | None = None  # in m

    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return


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
    def flatten(cls, this: "DishApertureEffects") -> Tuple[List[Any], Tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        return (
            [
                this.dish_diameter,
                this.focal_length,
                this.elevation_pointing_error_stddev,
                this.cross_elevation_pointing_error_stddev,
                this.axial_focus_error_stddev,
                this.elevation_feed_offset_stddev,
                this.cross_elevation_feed_offset_stddev,
                this.horizon_peak_astigmatism_stddev,
                this.surface_error_mean,
                this.surface_error_stddev
            ], (

            )
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "DishApertureEffects":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        dish_diameter, focal_length, elevation_pointing_error_stddev, cross_elevation_pointing_error_stddev, \
        axial_focus_error_stddev, elevation_feed_offset_stddev, cross_elevation_feed_offset_stddev, \
        horizon_peak_astigmatism_stddev, surface_error_mean, surface_error_stddev = children
        # _ = aux_data
        return DishApertureEffects(
            dish_diameter=dish_diameter,
            focal_length=focal_length,
            elevation_pointing_error_stddev=elevation_pointing_error_stddev,
            cross_elevation_pointing_error_stddev=cross_elevation_pointing_error_stddev,
            axial_focus_error_stddev=axial_focus_error_stddev,
            elevation_feed_offset_stddev=elevation_feed_offset_stddev,
            cross_elevation_feed_offset_stddev=cross_elevation_feed_offset_stddev,
            horizon_peak_astigmatism_stddev=horizon_peak_astigmatism_stddev,
            surface_error_mean=surface_error_mean,
            surface_error_stddev=surface_error_stddev,
            skip_post_init=True
        )

    def compute_pointing_error(self, x: FloatArray, y: FloatArray, elevation_point_error: FloatArray,
                               cross_elevation_point_error: FloatArray) -> FloatArray:
        # small angle approximation
        return elevation_point_error * x - cross_elevation_point_error * y

    def compute_feed_shift_error(self, x: FloatArray, y: FloatArray, r: FloatArray,
                                 cross_elevation_feed_offset: FloatArray, elevation_feed_offset: FloatArray,
                                 axial_focus_error: FloatArray) -> FloatArray:
        focal_ratio = r / self.focal_length
        sin_theta_p = focal_ratio / (1. + 0.25 * focal_ratio ** 2)
        cos_theta_p = (1. - 0.25 * focal_ratio ** 2) / (1. + 0.25 * focal_ratio ** 2)
        cos_phi = jnp.where(r == 0., 1., x / r)
        sin_phi = jnp.where(r == 0., 0., y / r)
        feed_shift_error = (
                axial_focus_error * cos_theta_p
                - elevation_feed_offset * sin_theta_p * cos_phi
                - cross_elevation_feed_offset * sin_theta_p * sin_phi
        )
        return feed_shift_error

    def compute_astigmatism_error(self, x: FloatArray, r: FloatArray, elevation: FloatArray,
                                  horizon_peak_astigmatism: FloatArray):
        cos_phi = jnp.where(r == 0., 1., x / r)
        cos_2phi = 2. * cos_phi ** 2 - 1.
        cos_elevation = jnp.cos(elevation)  # [num_times, 1, 1, num_ant, 1]
        peak_astigmatism = horizon_peak_astigmatism * cos_elevation
        R = 0.5 * self.dish_diameter
        return peak_astigmatism * (r / R) ** 2 * cos_2phi

    def apply_dish_aperture_effects(self, key, beam_model: BaseSphericalInterpolatorGainModel,
                                    geodesic_model: BaseGeodesicModel) -> BaseSphericalInterpolatorGainModel:
        # Compute path length errors for each model time and model frequency
        aperture_model = beam_model.to_aperture()
        xvec = aperture_model.lvec  # in units of lambda
        yvec = aperture_model.mvec
        model_times = beam_model.model_times
        model_freqs = beam_model.model_freqs
        c = 299792458.0
        model_wavelengths = c / model_freqs
        X, Y = jnp.meshgrid(xvec, yvec, indexing='ij')
        X = X[None, :, :, None, None] * model_wavelengths  # [1, lres, mres, 1, num_model_freqs]
        Y = Y[None, :, :, None, None] * model_wavelengths  # [1, lres, mres, 1, num_model_freqs]
        R = jnp.sqrt(X ** 2 + Y ** 2)  # [1, lres, mres, 1, num_model_freqs]

        path_length_errors = []  # [num_model_times/1, lres, mres, num_ant/1, num_model_freqs]

        def sample(key, per_time: bool, per_ant: bool):
            shape = [1, np.shape(xvec)[0], np.shape(xvec)[0], 1, 1]
            if per_time:
                shape[0] = np.shape(model_times)[0]
            if per_ant:
                shape[3] = geodesic_model.num_antennas
            return jax.random.normal(key, shape=shape)

        if self.elevation_pointing_error_stddev is not None:
            key, subkey = jax.random.split(key)
            elevation_pointing_error = sample(subkey, per_time=False,
                                              per_ant=True) * self.elevation_pointing_error_stddev
            cross_elevation_pointing_error = sample(subkey, per_time=False,
                                                    per_ant=True) * self.cross_elevation_pointing_error_stddev
            path_length_errors.append(
                self.compute_pointing_error(X, Y, elevation_pointing_error, cross_elevation_pointing_error)
            )
        if self.axial_focus_error_stddev is not None:
            key, subkey = jax.random.split(key)
            axial_focus_error = sample(subkey, per_time=False, per_ant=True) * self.axial_focus_error_stddev
            elevation_feed_offset = sample(subkey, per_time=False, per_ant=True) * self.elevation_feed_offset_stddev
            cross_elevation_feed_offset = sample(subkey, per_time=False,
                                                 per_ant=True) * self.cross_elevation_feed_offset_stddev
            path_length_errors.append(
                self.compute_feed_shift_error(X, Y, R, cross_elevation_feed_offset, elevation_feed_offset,
                                              axial_focus_error)
            )
        if self.horizon_peak_astigmatism_stddev is not None:
            key, subkey = jax.random.split(key)
            horizon_peak_astigmatism = sample(subkey, per_time=False,
                                              per_ant=True) * self.horizon_peak_astigmatism_stddev
            model_time_elevation = jax.vmap(
                lambda t: geodesic_model.compute_elevation_from_lmn(jnp.asarray([0., 0., 1.]), t)
            )(model_times)  # [num_model_times]
            # add dims to broadcast
            model_time_elevation = model_time_elevation[:, None, None, None, None]  # [num_model_times, 1, 1, 1, 1]
            path_length_errors.append(
                self.compute_astigmatism_error(X, R, model_time_elevation, horizon_peak_astigmatism)
            )
        if self.surface_error_mean is not None:
            key, subkey = jax.random.split(key)
            surface_error = sample(subkey, per_time=False,
                                   per_ant=True) * self.surface_error_stddev + self.surface_error_mean
            path_length_errors.append(surface_error)

        aperture_field = aperture_model.model_gains  # [num_model_times, lres, mres, [num_ant,] num_model_freqs[, 2, 2]]
        if len(path_length_errors) == 0:
            return beam_model

        path_length_error = sum(path_length_errors[1:],
                                start=path_length_errors[0])  # [num_model_times, lres, mres, num_ant]
        phase_error = (
                (2 * np.pi) * path_length_error / model_wavelengths
        )  # [num_model_times, lres, mres, num_ant, num_model_freqs]

        aperture_gains = jax.lax.complex(jnp.cos(phase_error), jnp.sin(phase_error))
        if beam_model.tile_antennas:
            aperture_field = aperture_field[:, :, :, None,
                             ...]  # [num_model_times, lres, mres, 1, num_model_freqs[, 2, 2]]
        if beam_model.full_stokes:
            aperture_field *= aperture_gains[
                ..., None, None]  # [num_model_times, lres, mres, num_ant, num_model_freqs[, 2, 2]]
        else:
            aperture_field *= aperture_gains
        aperture_model = BaseSphericalInterpolatorGainModel(
            model_times=model_times,
            model_freqs=model_freqs,
            model_gains=aperture_field,
            lvec=xvec,
            mvec=yvec,
            full_stokes=beam_model.full_stokes,
            tile_antennas=False
        )
        return aperture_model.to_image()

DishApertureEffects.register_pytree()

def build_dish_aperture_effects(
        # dish parameters
        dish_diameter: au.Quantity,
        focal_length: au.Quantity,

        # Dish effect parameters
        elevation_pointing_error_stddev: au.Quantity | None = None,
        cross_elevation_pointing_error_stddev: au.Quantity | None = None,
        axial_focus_error_stddev: au.Quantity | None = None,
        elevation_feed_offset_stddev: au.Quantity | None = None,
        cross_elevation_feed_offset_stddev: au.Quantity | None = None,
        horizon_peak_astigmatism_stddev: au.Quantity | None = None,
        surface_error_mean: au.Quantity | None = None,
        surface_error_stddev: au.Quantity | None = None,
) -> DishApertureEffects:
    return DishApertureEffects(
        dish_diameter=quantity_to_jnp(dish_diameter, 'm'),
        focal_length=quantity_to_jnp(focal_length, 'm'),
        elevation_pointing_error_stddev=quantity_to_jnp(elevation_pointing_error_stddev,
                                                        'rad') if elevation_pointing_error_stddev is not None else None,
        cross_elevation_pointing_error_stddev=quantity_to_jnp(cross_elevation_pointing_error_stddev,
                                                              'rad') if cross_elevation_pointing_error_stddev is not None else None,
        axial_focus_error_stddev=quantity_to_jnp(axial_focus_error_stddev,
                                                 'm') if axial_focus_error_stddev is not None else None,
        elevation_feed_offset_stddev=quantity_to_jnp(elevation_feed_offset_stddev,
                                                     'm') if elevation_feed_offset_stddev is not None else None,
        cross_elevation_feed_offset_stddev=quantity_to_jnp(cross_elevation_feed_offset_stddev,
                                                           'm') if cross_elevation_feed_offset_stddev is not None else None,
        horizon_peak_astigmatism_stddev=quantity_to_jnp(horizon_peak_astigmatism_stddev,
                                                        'm') if horizon_peak_astigmatism_stddev is not None else None,
        surface_error_mean=quantity_to_jnp(surface_error_mean, 'm') if surface_error_mean is not None else None,
        surface_error_stddev=quantity_to_jnp(surface_error_stddev, 'm') if surface_error_stddev is not None else None,
    )
