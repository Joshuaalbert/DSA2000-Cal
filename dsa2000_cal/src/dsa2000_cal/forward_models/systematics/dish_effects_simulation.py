import dataclasses
import os
import time as time_mod
from functools import partial
from typing import Tuple, Literal, NamedTuple

import jax
import numpy as np
from astropy import units as au, coordinates as ac, time as at, constants
from jax import numpy as jnp
from jax._src.typing import SupportsDType
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.cache_utils import check_cache
from dsa2000_cal.common.fourier_utils import ApertureTransform
from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights, apply_interp
from dsa2000_cal.common.mixed_precision_utils import complex_type
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.gain_models.gain_model import GainModel


def assert_congruent_unit(x: au.Quantity, unit: au.Unit):
    if not isinstance(x, au.Quantity):
        raise ValueError(f"Expected {x} to be an astropy quantity")
    if not x.unit.is_equivalent(unit):
        raise ValueError(f"Expected {x} to be in {unit} units but got {x.unit}")


def assert_same_shapes(*x: au.Quantity, expected_shape: Tuple[int, ...] | None = None):
    if len(x) == 0:
        return
    for xi in x:
        if not isinstance(xi, au.Quantity):
            raise ValueError(f"Expected {xi} to be an astropy quantity")
    if expected_shape is None:
        expected_shape = x[0].shape
    for xi in x:
        if xi.shape != expected_shape:
            raise ValueError(f"Expected {xi} to have shape {expected_shape} but got {xi.shape}")


def assert_scalar(*x: au.Quantity):
    for xi in x:
        if not isinstance(xi, au.Quantity):
            raise ValueError(f"Expected {xi} to be an astropy quantity")
        if not xi.isscalar:
            raise ValueError(f"Expected {xi} to be a scalar but got {xi}")


class DishEffectsParams(SerialisableBaseModel):
    # dish parameters
    dish_diameter: au.Quantity = 5. * au.m
    focal_length: au.Quantity = 2. * au.m

    # Dish effect parameters
    elevation_pointing_error_stddev: au.Quantity = 2. * au.arcmin
    cross_elevation_pointing_error_stddev: au.Quantity = 2. * au.arcmin
    axial_focus_error_stddev: au.Quantity = 3. * au.mm
    elevation_feed_offset_stddev: au.Quantity = 3. * au.mm
    cross_elevation_feed_offset_stddev: au.Quantity = 3. * au.mm
    horizon_peak_astigmatism_stddev: au.Quantity = 5. * au.mm
    surface_error_mean: au.Quantity = 3. * au.mm
    surface_error_stddev: au.Quantity = 1. * au.mm

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(DishEffectsParams, self).__init__(**data)
        # Use _check_measurement_set_meta_v0 as instance-wise validator
        _check_dish_effect_params(self)


def _check_dish_effect_params(dish_effect_params: DishEffectsParams):
    # Check units
    assert_congruent_unit(dish_effect_params.dish_diameter, au.m)
    assert_congruent_unit(dish_effect_params.focal_length, au.m)
    assert_congruent_unit(dish_effect_params.elevation_pointing_error_stddev, au.rad)
    assert_congruent_unit(dish_effect_params.cross_elevation_pointing_error_stddev, au.rad)
    assert_congruent_unit(dish_effect_params.axial_focus_error_stddev, au.m)
    assert_congruent_unit(dish_effect_params.elevation_feed_offset_stddev, au.m)
    assert_congruent_unit(dish_effect_params.cross_elevation_feed_offset_stddev, au.m)
    assert_congruent_unit(dish_effect_params.horizon_peak_astigmatism_stddev, au.m)
    assert_congruent_unit(dish_effect_params.surface_error_mean, au.m)
    assert_congruent_unit(dish_effect_params.surface_error_stddev, au.m)
    # Check shapes
    assert_scalar(
        dish_effect_params.dish_diameter,
        dish_effect_params.focal_length,
        dish_effect_params.elevation_pointing_error_stddev,
        dish_effect_params.cross_elevation_pointing_error_stddev,
        dish_effect_params.axial_focus_error_stddev,
        dish_effect_params.elevation_feed_offset_stddev,
        dish_effect_params.cross_elevation_feed_offset_stddev,
        dish_effect_params.horizon_peak_astigmatism_stddev,
        dish_effect_params.surface_error_mean,
        dish_effect_params.surface_error_stddev
    )


class SystemParams(NamedTuple):
    elevation_point_error: jax.Array  # [num_time, num_ant, 1]
    cross_elevation_point_error: jax.Array  # [num_time, num_ant, 1]
    axial_focus_error: jax.Array
    elevation_feed_offset: jax.Array
    cross_elevation_feed_offset: jax.Array
    horizon_peak_astigmatism: jax.Array
    surface_error: jax.Array


class DishEffectsSimulationCache(SerialisableBaseModel):
    seed: int

    antennas: ac.EarthLocation  # [num_ant]
    model_lmn: au.Quantity  # [Nm, Nl, 3]
    model_freqs: au.Quantity  # [num_model_freq]
    model_times: at.Time  # [num_time]

    dish_effects_params: DishEffectsParams

    model_gains: au.Quantity  # [num_time, Nm*Nl, num_ant, num_model_freq, 2, 2]

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(DishEffectsSimulationCache, self).__init__(**data)
        # Use _check_measurement_set_meta_v0 as instance-wise validator
        _check_dish_effects_gain_model_cache(self)


def _check_dish_effects_gain_model_cache(cache: DishEffectsSimulationCache):
    # Check units are congruent
    assert_congruent_unit(cache.model_freqs, au.Hz)
    assert_congruent_unit(cache.model_lmn, au.dimensionless_unscaled)
    assert_congruent_unit(cache.model_gains, au.dimensionless_unscaled)

    # Check shapes
    Nm, Nl, _ = np.shape(cache.model_lmn)
    assert_same_shapes(
        cache.model_gains,
        expected_shape=(len(cache.model_times), Nm, Nl, len(cache.antennas), len(cache.model_freqs), 2, 2)
    )


@dataclasses.dataclass(eq=False)
class DishEffectsSimulation:
    pointings: ac.ICRS | None
    dish_effect_params: DishEffectsParams

    # Beam model
    beam_gain_model: GainModel

    plot_folder: str
    cache_folder: str
    seed: int = 42
    convention: Literal['physical', 'engineering'] = 'physical'
    dtype: SupportsDType = complex_type

    def __post_init__(self):
        os.makedirs(self.cache_folder, exist_ok=True)
        os.makedirs(self.plot_folder, exist_ok=True)

        self.model_freqs = self.beam_gain_model.model_freqs
        self.model_times = self.beam_gain_model.model_times
        self.ref_time = self.model_times[0]
        self.antennas = self.beam_gain_model.antennas

        self.cache_file = os.path.join(self.cache_folder, f"cache_dish_effects_{self.seed}.json")

        # Set up fourier
        wavelengths = (constants.c / self.model_freqs).to('m')
        self.aperture_sampling_interval = np.min(wavelengths)
        self.dx = self.dy = self.aperture_sampling_interval / 2.
        # 2*R = sampling_interval * (2 * n + 1)
        n = int(self.dish_effect_params.dish_diameter / self.aperture_sampling_interval) + 1
        print(f"Using n={n} for dish effects simulation.")

        yvec = np.arange(-n, n + 1) * self.dy
        xvec = np.arange(-n, n + 1) * self.dx
        self.X, self.Y = np.meshgrid(xvec, yvec, indexing='ij')

        self.dl = self.dm = (1. / n) * au.dimensionless_unscaled
        self.lvec = np.arange(-n, n + 1) * self.dl
        self.mvec = np.arange(-n, n + 1) * self.dm
        M, L = np.meshgrid(self.mvec, self.lvec, indexing='ij')
        N = np.sqrt(1. - L ** 2 - M ** 2)
        self.model_lmn = au.Quantity(jnp.stack([L, M, N], axis=-1))  # [Nm, Nl, 3]

    def simulate_dish_effects(self) -> DishEffectsSimulationCache:
        if os.path.exists(self.cache_file):
            cache = DishEffectsSimulationCache.parse_file(self.cache_file)
            check_cache(
                cache_model=cache,
                model_freqs=self.model_freqs,
                model_times=self.model_times,
                dish_effects_params=self.dish_effect_params,
                antennas=self.antennas,
                seed=self.seed
            )
            print(f"Successfully loaded cache {self.cache_file}.")
            return cache

        # Compute aperture amplitude
        t0 = time_mod.time()
        model_gains = self.compute_model_gains()  # [Nm, Nl, num_ant, num_model_freq, 2, 2]
        t1 = time_mod.time()
        print(f"Successfully computed dish effects model gains in {t1 - t0:.2f} seconds.")

        cache = DishEffectsSimulationCache(
            seed=self.seed,
            model_freqs=self.model_freqs,
            model_times=self.model_times,
            model_lmn=self.model_lmn,
            antennas=self.antennas,
            dish_effects_params=self.dish_effect_params,
            model_gains=model_gains
        )
        with open(self.cache_file, 'w') as f:
            f.write(cache.json(indent=2))
        print(f"Successfully saved dish effects model cache {self.cache_file}.")
        return cache

    def compute_model_gains(self) -> au.Quantity:
        """
        Compute the model gains for the dish effects.

        Returns:
            [num_time, Nm, Nl, num_ant, num_model_freq, 2, 2]
        """
        freqs_jax = quantity_to_jnp(self.model_freqs)
        model_lmn_jax = jnp.reshape(quantity_to_jnp(self.model_lmn), (-1, 3))
        geodesics = jnp.tile(
            model_lmn_jax[:, None, None, :], (1, 1, len(self.antennas), 1)
        )  # #[num_sources, num_time, num_ant, 3]

        array_location = self.antennas[0]
        model_gains = []

        for i, time in enumerate(self.model_times):
            if self.pointings is None:
                pointing = zenith = ENU(
                    east=0, north=0, up=1, location=array_location, obstime=time).transform_to(ac.ICRS())
            else:
                pointing = self.pointings  # [[num_ant]]

            # Get beam model gains in image space
            rel_time = jnp.asarray((time.tt - self.ref_time.tt).sec)

            beam_model_gains_image = self.beam_gain_model.compute_gain(
                freqs=freqs_jax,
                times=rel_time,
                geodesics=geodesics
            )  # [num_sources, num_time=1, num_ant, num_model_freqs[, 2, 2]]

            altaz_frame = ac.AltAz(location=array_location, obstime=time)
            elevation = quantity_to_jnp(pointing.transform_to(altaz_frame).alt)

            _model_gains = np.asarray(
                self._compute_model_gains(
                    beam_model_gains_image=beam_model_gains_image,
                    rel_time=rel_time,
                    elevation_rad=elevation
                )
            )  # [Nm, Nl, num_ant, num_model_freq, 2, 2]
            model_gains.append(_model_gains)

        model_gains = np.stack(model_gains,
                               axis=0) * au.dimensionless_unscaled  # [num_time, Nm, Nl, num_ant, num_model_freq, 2, 2]
        return model_gains

    @partial(jax.jit, static_argnames=['self'])
    def _compute_model_gains(self, beam_model_gains_image: jax.Array,
                             rel_time: jax.Array,
                             elevation_rad: jax.Array) -> jax.Array:
        """
        Compute the model gains for the dish effects.

        Args:
            beam_model_gains_image: [num_model_times, resolution, resolution, [num_ant,] num_model_freqs, [2, 2]]
            rel_time: the relative time in seconds
            elevation_rad: the elevation in radians

        Returns:
            [Nm, Nl, num_ant, num_model_freq, 2, 2]
        """
        # Compute the beam model aperture field
        beam_model_gains_aperture = self._compute_beam_model_aperature_jax(
            beam_model_gains_image)  # [Nm, Nl, num_ant, num_model_freq, 2, 2]
        # Compute the aperture field
        model_gains_aperture = self._compute_model_gains_aperture_jax(
            beam_model_gains_aperture=beam_model_gains_aperture,
            rel_time=rel_time,
            elevation_rad=elevation_rad
        )  # [Nm, Nl, num_ant, num_model_freq, 2, 2]
        model_gains_image = self._compute_model_field_image_jax(
            model_gains_aperture=model_gains_aperture
        )  # [Nm, Nl, num_ant, num_model_freq, 2, 2]
        return model_gains_image

    def _compute_beam_model_aperature_jax(self, beam_model_gains_image: jax.Array) -> jax.Array:
        """
        Compute the beam model gains at the aperture of the dish.

        Args:
            beam_model_gains_image: [Nm, Nl, [num_ant,] num_model_freqs, [2, 2]]

        Returns:
            [Nm, Nl, num_ant, num_model_freq, 2, 2]
        """
        lmn_data = quantity_to_jnp(self.model_lmn)
        evanescent_mask = jnp.isnan(lmn_data[..., 2])  # [Nm, Nl]
        beam_model_gains_image = jnp.where(
            evanescent_mask[:, :, None, None, None, None],
            0., beam_model_gains_image
        )  # [Nm, Nl, num_ant, num_model_freq, 2, 2]
        am = ApertureTransform(convention=self.convention)
        dnu = quantity_to_jnp(self.dl * self.dm / self.aperture_sampling_interval ** 2)
        beam_model_gains_aperture = am.to_aperture(
            f_image=beam_model_gains_image, axes=(0, 1), dldm=dnu
        )  # [Nm, Nl, num_ant, num_model_freq, 2, 2]
        return beam_model_gains_aperture

    def _compute_model_field_image_jax(self, model_gains_aperture: jax.Array) -> jax.Array:
        """
        Compute the model field in the image plane.

        Args:
            model_gains_aperture: [Nm, Nl, num_ant, num_freq[, 2, 2]]

        Returns:
            [Nm, Nl, num_ant, num_freq[, 2, 2]]
        """
        am = ApertureTransform(convention=self.convention)
        dx = quantity_to_jnp(self.dx * self.dy)
        model_gains_image = am.to_image(
            f_aperture=model_gains_aperture, axes=(0, 1), dxdy=dx
        )  # [Nm, Nl, num_ant, num_freq[, 2, 2]]
        return model_gains_image

    def _get_system_params(self, key) -> SystemParams:
        keys = jax.random.split(key, 7)
        elevation_point_error = quantity_to_jnp(
            self.dish_effect_params.elevation_pointing_error_stddev) * jax.random.normal(
            key=keys[0],
            shape=(len(self.model_times), len(self.antennas), 1)
        )

        cross_elevation_point_error = quantity_to_jnp(
            self.dish_effect_params.cross_elevation_pointing_error_stddev) * jax.random.normal(
            key=keys[1],
            shape=(len(self.model_times), len(self.antennas), 1)
        )

        axial_focus_error = quantity_to_jnp(self.dish_effect_params.axial_focus_error_stddev) * jax.random.normal(
            key=keys[2],
            shape=(len(self.model_times), len(self.antennas), 1)
        )

        elevation_feed_offset = quantity_to_jnp(
            self.dish_effect_params.elevation_feed_offset_stddev) * jax.random.normal(
            key=keys[3],
            shape=(len(self.antennas), 1)
        )

        cross_elevation_feed_offset = quantity_to_jnp(
            self.dish_effect_params.cross_elevation_feed_offset_stddev) * jax.random.normal(
            key=keys[4],
            shape=(len(self.antennas), 1)
        )

        horizon_peak_astigmatism = quantity_to_jnp(
            self.dish_effect_params.horizon_peak_astigmatism_stddev) * jax.random.normal(
            key=keys[5],
            shape=(len(self.antennas), 1)
        )

        surface_error = quantity_to_jnp(self.dish_effect_params.surface_error_mean) + quantity_to_jnp(
            self.dish_effect_params.surface_error_stddev) * jax.random.normal(
            key=keys[6],
            shape=(len(self.antennas), 1)
        )

        return SystemParams(
            elevation_point_error=elevation_point_error,
            cross_elevation_point_error=cross_elevation_point_error,
            axial_focus_error=axial_focus_error,
            elevation_feed_offset=elevation_feed_offset,
            cross_elevation_feed_offset=cross_elevation_feed_offset,
            horizon_peak_astigmatism=horizon_peak_astigmatism,
            surface_error=surface_error
        )

    def _compute_model_gains_aperture_jax(self, beam_model_gains_aperture: jax.Array,
                                          rel_time: jax.Array,
                                          elevation_rad: jax.Array) -> jax.Array:
        """
        Computes the E-field at the aperture of the dish.

        Args:
            beam_model_gains_aperture: [Nm, Nl, num_ant, num_freq, 2, 2]
            rel_time: relative time in seconds
            elevation_rad: the elevation

        Returns:
            [Nm, Nl, num_ant, num_freq, 2, 2]
        """

        system_params = self._get_system_params(jax.random.PRNGKey(self.seed))

        X = quantity_to_jnp(self.X)
        Y = quantity_to_jnp(self.Y)
        focal_length = quantity_to_jnp(self.dish_effect_params.focal_length)
        dish_diameter = quantity_to_jnp(self.dish_effect_params.dish_diameter)  # [1]
        R = 0.5 * dish_diameter
        wavelengths = quantity_to_jnp(constants.c) / quantity_to_jnp(self.model_freqs)

        cross_elevation_point_error = system_params.cross_elevation_point_error  # [num_time, num_ant, 1]
        elevation_point_error = system_params.elevation_point_error  # [num_time, num_ant, 1]
        axial_focus_error = system_params.axial_focus_error  # [num_time, num_ant, 1]
        elevation_feed_offset = system_params.elevation_feed_offset  # [num_ant, 1]
        cross_elevation_feed_offset = system_params.cross_elevation_feed_offset  # [num_ant, 1]
        horizon_peak_astigmatism = system_params.horizon_peak_astigmatism  # [num_ant, 1]
        surface_error = system_params.surface_error  # [num_ant, 1]

        (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(
            rel_time,
            jnp.asarray((self.model_times - self.ref_time).sec)
        )

        def _interp_time(x):
            return apply_interp(x, i0, alpha0, i1, alpha1, axis=0)

        X = X[:, :, None, None]  # [Nm, Nl, 1, 1]
        Y = Y[:, :, None, None]  # [Nm, Nl, 1, 1]

        pointing_error = _interp_time(elevation_point_error) * X - _interp_time(
            cross_elevation_point_error) * Y

        r = jnp.sqrt(X ** 2 + Y ** 2)
        focal_ratio = r / focal_length  # [Nm, Nl, 1, 1]

        sin_theta_p = focal_ratio / (1. + 0.25 * focal_ratio ** 2)
        cos_theta_p = (1. - 0.25 * focal_ratio ** 2) / (1. + 0.25 * focal_ratio ** 2)

        cos_phi = jnp.where(r == 0., 1., X / r)
        sin_phi = jnp.where(r == 0., 0., Y / r)

        feed_shift_error = (
                _interp_time(axial_focus_error) * cos_theta_p
                - elevation_feed_offset * sin_theta_p * cos_phi
                - cross_elevation_feed_offset * sin_theta_p * sin_phi
        )
        cos_2phi = 2. * cos_phi ** 2 - 1.
        cos_elevation = jnp.cos(elevation_rad)
        peak_astigmatism = horizon_peak_astigmatism * cos_elevation
        astigmatism_error = peak_astigmatism * (r / R) ** 2 * cos_2phi

        total_path_length_error = pointing_error + feed_shift_error + astigmatism_error + surface_error  # [Nm, Nl, num_ant, 1]

        if self.convention == 'engineering':
            constant = jnp.asarray(2j * jnp.pi, self.dtype)  # [num_freqs]
        elif self.convention == 'physical':
            constant = jnp.asarray(-2j * jnp.pi, self.dtype)  # [num_freqs]
        else:
            raise ValueError(f"Unknown convention {self.convention}")

        # Multiple in aperature
        aperture_field = jnp.exp(constant * total_path_length_error / wavelengths)  # [Nm, Nl, num_ant, num_freq]
        model_gains_aperture = aperture_field[
                                   ..., None, None] * beam_model_gains_aperture  # [Nm, Nl, num_ant, num_freq, 2, 2]
        return model_gains_aperture
