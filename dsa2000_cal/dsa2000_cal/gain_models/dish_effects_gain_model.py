import dataclasses
import os.path
from functools import partial
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as au, coordinates as ac, time as at, constants

from dsa2000_cal.common.coord_utils import lmn_to_icrs, icrs_to_lmn
from dsa2000_cal.common.fourier_utils import ApertureTransform
from dsa2000_cal.common.interp_utils import multilinear_interp_2d, get_interp_indices_and_weights
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.gain_models.beam_gain_model import BeamGainModel
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


class DishEffectsGainModelParams(SerialisableBaseModel):
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
        super(DishEffectsGainModelParams, self).__init__(**data)
        # Use _check_measurement_set_meta_v0 as instance-wise validator
        _check_dish_effect_gain_model_params(self)


def _check_dish_effect_gain_model_params(dish_effect_params: DishEffectsGainModelParams):
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


class DishEffectsGainModelCache(SerialisableBaseModel):
    seed: int

    model_freqs: au.Quantity  # [num_model_freq]
    model_times: at.Time  # [num_time]

    dish_effects_params: DishEffectsGainModelParams

    elevation_point_error: au.Quantity  # [num_time, num_ant, 1]
    cross_elevation_point_error: au.Quantity  # [num_time, num_ant, 1]
    axial_focus_error: au.Quantity  # [num_time, num_ant, 1]
    elevation_feed_offset: au.Quantity  # [num_ant, 1]
    cross_elevation_feed_offset: au.Quantity  # [num_ant, 1]
    horizon_peak_astigmatism: au.Quantity  # [num_ant, 1]
    surface_error: au.Quantity  # [num_ant, 1]
    antennas: ac.EarthLocation  # [num_ant]

    sampling_interval: au.Quantity
    dx: au.Quantity
    dy: au.Quantity
    X: au.Quantity  # [Nm, Nl]
    Y: au.Quantity  # [Nm, Nl]
    dl: au.Quantity
    dm: au.Quantity
    lvec: au.Quantity  # [2n+1]
    mvec: au.Quantity  # [2n+1]
    lmn_data: au.Quantity  # [Nm, Nl, 3]

    aperture_gains: au.Quantity  # [Nm, Nl, num_model_freq, 2, 2]
    convention: str

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(DishEffectsGainModelCache, self).__init__(**data)
        # Use _check_measurement_set_meta_v0 as instance-wise validator
        _check_dish_effects_gain_model_cache(self)


def _check_dish_effects_gain_model_cache(cache: DishEffectsGainModelCache):
    # Check units are congruent
    assert_congruent_unit(cache.model_freqs, au.Hz)
    assert_congruent_unit(cache.elevation_point_error, au.rad)
    assert_congruent_unit(cache.cross_elevation_point_error, au.rad)
    assert_congruent_unit(cache.axial_focus_error, au.m)
    assert_congruent_unit(cache.elevation_feed_offset, au.m)
    assert_congruent_unit(cache.cross_elevation_feed_offset, au.m)
    assert_congruent_unit(cache.horizon_peak_astigmatism, au.m)
    assert_congruent_unit(cache.surface_error, au.m)
    assert_congruent_unit(cache.sampling_interval, au.m)
    assert_congruent_unit(cache.dx, au.m)
    assert_congruent_unit(cache.dy, au.m)
    assert_congruent_unit(cache.X, au.m)
    assert_congruent_unit(cache.Y, au.m)
    assert_congruent_unit(cache.dl, au.dimensionless_unscaled)
    assert_congruent_unit(cache.dm, au.dimensionless_unscaled)
    assert_congruent_unit(cache.lvec, au.dimensionless_unscaled)
    assert_congruent_unit(cache.mvec, au.dimensionless_unscaled)
    assert_congruent_unit(cache.lmn_data, au.dimensionless_unscaled)
    assert_congruent_unit(cache.aperture_gains, au.dimensionless_unscaled)

    # Check shapes
    assert_same_shapes(
        cache.elevation_point_error,
        cache.cross_elevation_point_error,
        cache.axial_focus_error
    )  # [num_time, num_ant, 1]
    assert_same_shapes(
        cache.elevation_feed_offset,
        cache.cross_elevation_feed_offset,
        cache.horizon_peak_astigmatism,
        cache.surface_error
    )  # [num_ant, 1]
    assert_scalar(
        cache.sampling_interval,
        cache.dx,
        cache.dy,
        cache.dl,
        cache.dm
    )
    N = cache.lvec.shape[0]
    assert_same_shapes(
        cache.X,
        cache.Y,
        expected_shape=(N, N)
    )
    assert_same_shapes(
        cache.lvec,
        cache.mvec
    )
    assert_same_shapes(
        cache.lmn_data,
        expected_shape=(N, N, 3)
    )
    assert_same_shapes(
        cache.aperture_gains,
        expected_shape=(N, N, len(cache.model_freqs), 2, 2)
    )


@dataclasses.dataclass(eq=False)
class DishEffectsGainModel(GainModel):
    """
    Uses nearest neighbour interpolation to compute the gain model.

    The antennas have attenuation models in frame of antenna, call this the X-Y frame (see below).
    X points up, Y points to the right, Z points towards the source (along bore).
    """
    dish_effect_params: DishEffectsGainModelParams

    # Beam model
    beam_gain_model: BeamGainModel

    # The time axis to precompute the dish effects
    model_times: at.Time  # [num_times]

    plot_folder: str
    cache_folder: str
    seed: int = 42
    convention: Literal['fourier', 'casa'] = 'fourier'
    dtype: jnp.dtype = jnp.complex64

    def __post_init__(self):
        os.makedirs(self.cache_folder, exist_ok=True)
        os.makedirs(self.plot_folder, exist_ok=True)
        self.model_freqs = self.beam_gain_model.model_freqs

        # make sure all 1D
        if self.model_times.isscalar:
            self.model_times = self.model_times.reshape((1,))

        self.ref_time = self.model_times[0]

        self.num_model_times = len(self.model_times)
        self.num_model_freq = len(self.model_freqs)
        self.model_wavelengths = (constants.c / self.model_freqs).to('m')

        # Check shapes
        if len(self.model_freqs.shape) != 1:
            raise ValueError(f"Expected freqs to have 1 dimension but got {len(self.model_freqs.shape)}")
        if len(self.model_times.shape) != 1:
            raise ValueError(f"Expected times to have 1 dimension but got {len(self.model_times.shape)}")

        # Compute everything that needs to be computed
        self._prepare_system()

    def _prepare_system(self):
        cache_file = os.path.join(self.cache_folder, f"dish_effects_gain_model_cache_{self.seed}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as fp:
                cache = DishEffectsGainModelCache.parse_raw(fp.read())
            if not np.allclose(cache.model_freqs.value, self.model_freqs.value):
                raise ValueError(
                    f"Model freqs in cache {cache.model_freqs} does not match model freqs {self.model_freqs}")
            if not np.all((cache.model_times - self.model_times).sec < 1e-3):
                raise ValueError(
                    f"Model times in cache {cache.model_times} does not match model times {self.model_times}")
            if cache.dish_effects_params != self.dish_effect_params:
                raise ValueError(
                    f"Dish effect params in cache {cache.dish_effects_params} does not match "
                    f"dish effect params {self.dish_effect_params}"
                )
            if len(cache.antennas) != len(self.antennas):
                raise ValueError(f"Number of antennas in cache {len(cache.antennas)} does not match number of antennas")
            if cache.seed != self.seed:
                raise ValueError(f"Seed in cache {cache.seed} does not match seed {self.seed}")
            if cache.convention != self.convention:
                raise ValueError(f"Convention in cache {cache.convention} does not match convention {self.convention}")
            print(f"Successfully loaded cache {cache_file}.")
            # Assign the cache values that need to be (some are just for checking consistency)
            self.elevation_point_error = cache.elevation_point_error
            self.cross_elevation_point_error = cache.cross_elevation_point_error
            self.axial_focus_error = cache.axial_focus_error
            self.elevation_feed_offset = cache.elevation_feed_offset
            self.cross_elevation_feed_offset = cache.cross_elevation_feed_offset
            self.horizon_peak_astigmatism = cache.horizon_peak_astigmatism
            self.surface_error = cache.surface_error
            self.sampling_interval = cache.sampling_interval
            self.dx = cache.dx
            self.dy = cache.dy
            self.X = cache.X
            self.Y = cache.Y
            self.dl = cache.dl
            self.dm = cache.dm
            self.lvec = cache.lvec
            self.mvec = cache.mvec
            self.lmn_data = cache.lmn_data
            self.aperture_gains = cache.aperture_gains
        else:

            # Generate the dish effects, broadcasts with [num_antenna, num_freq] ater time interpolation
            keys = jax.random.split(jax.random.PRNGKey(self.seed), 7)
            self.elevation_point_error = self.dish_effect_params.elevation_pointing_error_stddev * np.asarray(
                jax.random.normal(
                    key=keys[0],
                    shape=(self.num_model_times, len(self.antennas), 1)
                )
            )
            self.cross_elevation_point_error = self.dish_effect_params.cross_elevation_pointing_error_stddev * np.asarray(
                jax.random.normal(
                    key=keys[1],
                    shape=(self.num_model_times, len(self.antennas), 1)
                )
            )
            self.axial_focus_error = self.dish_effect_params.axial_focus_error_stddev * np.asarray(
                jax.random.normal(
                    key=keys[2],
                    shape=(self.num_model_times, len(self.antennas), 1)
                )
            )
            self.elevation_feed_offset = self.dish_effect_params.elevation_feed_offset_stddev * np.asarray(
                jax.random.normal(
                    key=keys[3],
                    shape=(len(self.antennas), 1)
                )
            )
            self.cross_elevation_feed_offset = self.dish_effect_params.cross_elevation_feed_offset_stddev * np.asarray(
                jax.random.normal(
                    key=keys[4],
                    shape=(len(self.antennas), 1)
                )
            )
            self.horizon_peak_astigmatism = self.dish_effect_params.horizon_peak_astigmatism_stddev * np.asarray(
                jax.random.normal(
                    key=keys[5],
                    shape=(len(self.antennas), 1)
                )
            )
            self.surface_error = self.dish_effect_params.surface_error_mean + self.dish_effect_params.surface_error_stddev * np.asarray(
                jax.random.normal(
                    key=keys[6],
                    shape=(len(self.antennas), 1)
                )
            )
            # Compute aperture amplitude
            self.sampling_interval = np.min(self.model_wavelengths)
            self.dx = self.dy = self.sampling_interval / 2.
            # 2*R = sampling_interval * (2 * n + 1)
            n = int(self.dish_effect_params.dish_diameter / self.sampling_interval) + 1

            yvec = np.arange(-n, n + 1) * self.dy
            xvec = np.arange(-n, n + 1) * self.dx
            self.X, self.Y = np.meshgrid(xvec, yvec, indexing='ij')

            self.dl = self.dm = (1. / n) * au.dimensionless_unscaled
            self.lvec = np.arange(-n, n + 1) * self.dl
            self.mvec = np.arange(-n, n + 1) * self.dm
            M, L = np.meshgrid(self.mvec, self.lvec, indexing='ij')
            N = np.sqrt(1. - L ** 2 - M ** 2)
            self.lmn_data = au.Quantity(jnp.stack([L, M, N], axis=-1))  # [Nm, Nl, 3]
            self.aperture_gains = self.compute_aperture_amplitude()  # [Nm, Nl, num_model_freq, 2, 2]

            # Plot abs and phase of aperture field (X=M, Y=-L)
            for elevation in [45, 90] * au.deg:
                aperature_field = self.compute_aperture_field_model(freqs=self.model_freqs[:1],
                                                                    time=self.ref_time,
                                                                    elevation=elevation)  # [Nm, Nl, num_ant, 1, 2, 2]
                aperature_field = aperature_field[..., 0, 0]  # [Nm, Nl, num_ant, 1]
                fig, axs = plt.subplots(2, 1, figsize=(8, 8), squeeze=False, sharex=True, sharey=True)
                im = axs[0, 0].imshow(
                    np.abs(aperature_field[:, :, 0]),  # rows are X, columns are Y
                    origin='lower',
                    extent=(self.Y.min().value, self.Y.max().value,
                            self.X.min().value, self.X.max().value
                            ),
                    cmap='PuOr'
                )
                fig.colorbar(im, ax=axs[0, 0])
                axs[0, 0].set_xlabel('Y [m]')
                axs[0, 0].set_ylabel('X [m]')
                axs[0, 0].set_title(f'Aperture amplitude {elevation} elevation')
                im = axs[1, 0].imshow(
                    np.angle(aperature_field[:, :, 0]),  # rows are X, columns are Y
                    origin='lower',
                    extent=(self.Y.min().value, self.Y.max().value,
                            self.X.min().value, self.X.max().value,
                            ),
                    cmap='hsv',
                    vmin=-np.pi,
                    vmax=np.pi
                )
                fig.colorbar(im, ax=axs[1, 0])
                axs[1, 0].set_xlabel('Y [m]')
                axs[1, 0].set_ylabel('X [m]')
                axs[1, 0].set_title(f'Aperture phase {elevation} elevation')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plot_folder, f"aperture_field_{self.seed}_{elevation.value}.png"))
                plt.close(fig)

            # Plot the image plane effects
            for elevation in [45, 90] * au.deg:
                phase_tracking = ac.AltAz(alt=elevation, az=0 * au.deg,
                                          location=ac.EarthLocation(lat=0, lon=0, height=0),
                                          obstime=self.ref_time).transform_to(ac.ICRS())
                sources = lmn_to_icrs(self.lmn_data, time=self.ref_time, phase_tracking=phase_tracking)
                gain = self.compute_gain(freqs=self.model_freqs[:1], sources=sources,
                                         array_location=ac.EarthLocation(lat=0, lon=0, height=0), time=self.ref_time,
                                         pointing=phase_tracking,
                                         mode='fft')  # [Nm, Nl, num_ant, num_model_freq, 2, 2]
                gain = gain[:, :, 0, 0, 0, 0]  # [Nm, Nl]
                fig, axs = plt.subplots(2, 1, figsize=(8, 8), squeeze=False, sharex=True, sharey=True)
                im = axs[0, 0].imshow(
                    np.abs(gain),  # rows are M, columns are L
                    origin='lower',
                    extent=(self.lvec.min().value, self.lvec.max().value,
                            self.mvec.min().value, self.mvec.max().value),
                    cmap='PuOr'
                )
                fig.colorbar(im, ax=axs[0, 0])
                axs[0, 0].set_xlabel('l')
                axs[0, 0].set_ylabel('m')
                axs[0, 0].set_title(f'Beam gain amplitude {elevation} elevation')
                im = axs[1, 0].imshow(
                    np.angle(gain) * 180 / np.pi,  # rows are M, columns are L
                    origin='lower',
                    extent=(self.lvec.min().value, self.lvec.max().value,
                            self.mvec.min().value, self.mvec.max().value),
                    cmap='coolwarm',
                    # vmin=-np.pi,
                    # vmax=np.pi
                )
                fig.colorbar(im, ax=axs[1, 0], label='degrees')
                axs[1, 0].set_xlabel('l')
                axs[1, 0].set_ylabel('m')
                axs[1, 0].set_title(f'Beam gain phase {elevation} elevation')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plot_folder, f"beam_gain_{self.seed}_{elevation.value}.png"))
                plt.close(fig)

            cache = DishEffectsGainModelCache(
                seed=self.seed,
                model_freqs=self.model_freqs,
                model_times=self.model_times,
                antennas=self.antennas,
                dish_effects_params=self.dish_effect_params,
                elevation_point_error=self.elevation_point_error,
                cross_elevation_point_error=self.cross_elevation_point_error,
                axial_focus_error=self.axial_focus_error,
                elevation_feed_offset=self.elevation_feed_offset,
                cross_elevation_feed_offset=self.cross_elevation_feed_offset,
                horizon_peak_astigmatism=self.horizon_peak_astigmatism,
                surface_error=self.surface_error,
                sampling_interval=self.sampling_interval,
                dx=self.dx,
                dy=self.dy,
                X=self.X,
                Y=self.Y,
                dl=self.dl,
                dm=self.dm,
                lvec=self.lvec,
                mvec=self.mvec,
                lmn_data=self.lmn_data,
                aperture_gains=self.aperture_gains,
                convention=self.convention
            )
            with open(cache_file, 'w') as f:
                f.write(cache.json(indent=2))
            print(f"Successfully saved dish effects model cache {cache_file}.")

    @partial(jax.jit, static_argnames=['self'])
    def _compute_aperture_gains_jax(self, beam_gains: jax.Array) -> jax.Array:
        evanescent_mask = jnp.asarray(np.isnan(self.lmn_data[..., 2]))  # [Nm, Nl]
        beam_gains = jnp.where(evanescent_mask[:, :, None, None, None], 0.,
                               beam_gains)  # [Nm, Nl, num_model_freq, 2, 2]
        am = ApertureTransform(convention=self.convention)
        dnu = quantity_to_jnp(self.dl * self.dm / self.sampling_interval ** 2)
        aperture_gains = am.to_aperture(
            f_image=beam_gains, axes=(0, 1), dnu=dnu
        )  # [Nm, Nl, num_model_freq, 2, 2]
        return aperture_gains

    def compute_aperture_amplitude(self) -> au.Quantity:
        # Aribtrary location and time and phase_tracking will do.
        array_location = ac.EarthLocation(lat=0, lon=0, height=0)
        time = at.Time('2021-01-01T00:00:00', scale='utc')
        phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
        sources = lmn_to_icrs(self.lmn_data, time=time, phase_tracking=phase_tracking)

        beam_gains = self.beam_gain_model.compute_gain(freqs=self.model_freqs, sources=sources,
                                                       array_location=array_location, time=time,
                                                       pointing=phase_tracking)  # [Nm, Nl, num_ant, num_model_freq, 2, 2]
        beam_gains = beam_gains[..., 0, :, :, :]  # [Nm, Nl, num_model_freq, 2, 2]
        return np.asarray(
            self._compute_aperture_gains_jax(beam_gains=beam_gains)) * au.dimensionless_unscaled

    def _compute_aperture_field_model_jax(self, freqs: jax.Array, rel_time: jax.Array,
                                          elevation_rad: jax.Array) -> jax.Array:
        """
        Computes the E-field at the aperture of the dish.

        Args:
            freqs: the frequency values
            rel_time: relative time in seconds
            elevation_rad: the elevation

        Returns:
            (shape) + path length distortion in meters
        """

        X = quantity_to_jnp(self.X)
        Y = quantity_to_jnp(self.Y)
        focal_length = quantity_to_jnp(self.dish_effect_params.focal_length)
        cross_elevation_point_error = quantity_to_jnp(self.cross_elevation_point_error)
        elevation_point_error = quantity_to_jnp(self.elevation_point_error)
        axial_focus_error = quantity_to_jnp(self.axial_focus_error)
        elevation_feed_offset = quantity_to_jnp(self.elevation_feed_offset)
        cross_elevation_feed_offset = quantity_to_jnp(self.cross_elevation_feed_offset)
        horizon_peak_astigmatism = quantity_to_jnp(self.horizon_peak_astigmatism)
        surface_error = quantity_to_jnp(self.surface_error)
        dish_diameter = quantity_to_jnp(self.dish_effect_params.dish_diameter)
        R = 0.5 * dish_diameter
        model_freqs = quantity_to_jnp(self.model_freqs)
        wavelengths = quantity_to_jnp(constants.c) / freqs
        aperture_gains = quantity_to_jnp(self.aperture_gains)  # [Nm, Nl, num_freqs, 2, 2]

        # Interp freq
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(freqs, model_freqs)
        aperture_gains = aperture_gains[..., i0, :, :] * alpha0 + aperture_gains[
                                                                  ..., i1, :,
                                                                  :] * alpha1  # [Nm, Nl, num_freqs, 2, 2]

        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(rel_time,
                                                                    jnp.asarray((self.model_times - self.ref_time).sec))

        def _interp_time(x):
            output = x[i0] * alpha0 + x[i1] * alpha1
            return output

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

        if self.convention == 'casa':
            constant = jnp.asarray(2j * jnp.pi, self.dtype)  # [num_freqs]
        elif self.convention == 'fourier':
            constant = jnp.asarray(-2j * jnp.pi, self.dtype)  # [num_freqs]
        else:
            raise ValueError(f"Unknown convention {self.convention}")

        aperture_field = jnp.exp(constant * total_path_length_error / wavelengths)  # [Nm, Nl, num_ant, num_freq]
        aperture_field = aperture_field[..., None, None] * aperture_gains[..., None, :, :,
                                                           :]  # [Nm, Nl, num_ant, num_freq, 2, 2]
        return aperture_field

    def compute_aperture_field_model(self, freqs: au.Quantity, time: at.Time, elevation: au.Quantity) -> jax.Array:
        """
        Computes the E-field at the aperture of the dish.

        Args:
            freqs: the frequency values
            time: the time
            elevation: the elevation

        Returns:
            [Nm, Nl, num_ant, num_freq] aperture field
        """
        if freqs.isscalar:
            freqs = freqs.reshape((1,))
        if len(freqs.shape) != 1:
            raise ValueError(f"Expected freqs to have 1 dimension but got {len(freqs.shape)}")
        if not freqs.unit.is_equivalent(au.Hz):
            raise ValueError(f"Expected freqs to be in Hz but got {freqs.unit}")
        if not elevation.isscalar:
            raise ValueError(f"Expected elevation to be a scalar but got {elevation}")
        if not elevation.unit.is_equivalent(au.rad):
            raise ValueError(f"Expected elevation to be in radians but got {elevation.unit}")

        return self._compute_aperture_field_model_jax(
            freqs=quantity_to_jnp(freqs),
            rel_time=jnp.asarray((time - self.ref_time).sec),
            elevation_rad=quantity_to_jnp(elevation)
        )  # [Nm, Nl, num_ant, num_freq, 2, 2]

    def _compute_beam_fft_jax(self, lmn_sources: jax.Array, aperture_field: jax.Array) -> jax.Array:
        mvec = quantity_to_jnp(self.mvec)
        lvec = quantity_to_jnp(self.lvec)
        # Fourier to fair-field using FFT
        am = ApertureTransform(convention=self.convention)
        dx = quantity_to_jnp(self.dx * self.dy)
        image_field = am.to_image(
            f_aperture=aperture_field, axes=(0, 1), dx=dx
        )  # [Nm, Nl, num_ant, num_freq, 2, 2]

        image_field = multilinear_interp_2d(
            x=lmn_sources[:, 1], y=lmn_sources[:, 0],
            xp=mvec, yp=lvec, z=image_field,
        )  # [num_sources, num_ant, num_freq, 2, 2]
        return image_field

    def _compute_beam_dft_jax(self, freqs: jax.Array, lmn_sources: jax.Array, aperture_field: jax.Array):
        wavelengths = quantity_to_jnp(constants.c) / freqs  # [num_freqs]
        Y = quantity_to_jnp(self.Y)
        X = quantity_to_jnp(self.X)
        # Opposing constant
        if self.convention == 'casa':
            constant = jnp.asarray(-2j * jnp.pi, self.dtype)  # [num_freqs]
        elif self.convention == 'fourier':
            constant = jnp.asarray(2j * jnp.pi, self.dtype)  # [num_freqs]
        else:
            raise ValueError(f"Unknown convention {self.convention}")

        Y_lambda = Y[:, :, None, None, None] / wavelengths  # [Nm, Nl, 1, 1, num_freqs]
        X_lambda = X[:, :, None, None, None] / wavelengths  # [Nm, Nl, 1, 1, num_freqs]
        l_sources = lmn_sources[:, None, None, 0]  # [num_sources, 1, 1]
        m_sources = lmn_sources[:, None, None, 1]  # [num_sources, 1, 1]

        aperture_field = aperture_field[:, :, None, :, :]  # [Nm, Nl, 1, num_ant, num_freq, 2, 2]

        # L = -Y axis, M = X axis
        unity_root = jnp.exp(
            constant * (-Y_lambda * l_sources + X_lambda * m_sources)
        )  # [Nm, Nl, num_sources, 1, num_freqs]
        image_field = (
                unity_root[..., None, None] * aperture_field)  # [Nm, Nl, num_sources, num_ant, num_freq, 2, 2]

        dx = quantity_to_jnp(self.dx * self.dy)
        image_field = jnp.sum(image_field, axis=(0, 1)) * dx  # [num_sources, num_ant, num_freqs, 2, 2]
        return image_field

    @partial(jax.jit, static_argnames=['self', 'mode'])
    def _compute_gain_jax(self, freqs: jax.Array, lmn_sources: jax.Array, rel_time: jax.Array, elevation_rad: jax.Array,
                          mode: str):
        shape = np.shape(lmn_sources)[:-1]
        lmn_sources = lmn_sources.reshape((-1, 3))

        aperture_field = self._compute_aperture_field_model_jax(
            freqs=freqs,
            rel_time=rel_time, elevation_rad=elevation_rad
        )  # [Nm, Nl, num_antenna, num_freq, 2, 2]
        if mode == 'fft':
            image_field = self._compute_beam_fft_jax(
                lmn_sources=lmn_sources,
                aperture_field=aperture_field
            )  # [num_sources, num_ant, num_freqs, 2, 2]
        elif mode == 'dft':
            image_field = self._compute_beam_dft_jax(
                freqs=freqs,
                lmn_sources=lmn_sources,
                aperture_field=aperture_field
            )  # [num_sources, num_ant, num_freqs, 2, 2]
        else:
            raise ValueError(f"Unknown mode {mode}")

        evanescent_mask = jnp.isnan(lmn_sources[..., 2])  # [num_sources]
        image_field = jnp.where(evanescent_mask[:, None, None, None, None], np.nan,
                                image_field)  # [num_sources, num_ant, num_freqs, 2, 2]

        image_field = image_field.reshape(shape + image_field.shape[1:])  # (source_shape) + [num_ant, num_freqs, 2, 2]
        gains = jnp.asarray(image_field, self.dtype)
        return gains

    def compute_gain(self, freqs: au.Quantity, sources: ac.ICRS, pointing: ac.ICRS, array_location: ac.EarthLocation,
                     time: at.Time, **kwargs: str):
        mode = kwargs.get('mode', 'fft')

        print(
            f"Computing dish effects gain model for {len(sources)} sources and {len(self.antennas)} antennas "
            f"at {time} at {len(freqs)} freqs using {mode} mode."
        )

        if freqs.isscalar:
            freqs = freqs.reshape((1,))
        if len(freqs.shape) != 1:
            raise ValueError(f"Expected freqs to have 1 dimension but got {len(freqs.shape)}")
        if not freqs.unit.is_equivalent(au.Hz):
            raise ValueError(f"Expected freqs to be in Hz but got {freqs.unit}")

        altaz_frame = ac.AltAz(location=array_location, obstime=time)
        elevation = pointing.transform_to(altaz_frame).alt
        lmn_sources = icrs_to_lmn(sources=sources, time=time, phase_tracking=pointing)  # (source_shape) + [3]

        return self._compute_gain_jax(
            freqs=quantity_to_jnp(freqs),
            lmn_sources=quantity_to_jnp(lmn_sources),
            rel_time=jnp.asarray((time - self.ref_time).sec),
            elevation_rad=quantity_to_jnp(elevation),
            mode=mode
        )
