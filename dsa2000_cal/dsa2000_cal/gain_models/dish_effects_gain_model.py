import dataclasses
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
import pytest
from astropy import units as au, coordinates as ac, time as at, constants

from dsa2000_cal.common.coord_utils import lmn_to_icrs, icrs_to_lmn
from dsa2000_cal.common.fourier_utils import ApertureTransform
from dsa2000_cal.common.interp_utils import multilinear_interp_2d, get_interp_indices_and_weights
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.beam_gain_model import BeamGainModel, beam_gain_model_factor
from dsa2000_cal.gain_models.gain_model import GainModel


@dataclasses.dataclass(eq=False)
class DishEffectsGainModel(GainModel):
    """
    Uses nearest neighbour interpolation to compute the gain model.

    The antennas have attenuation models in frame of antenna, call this the X-Y frame (see below).
    X points up, Y points to the right, Z points towards the source (along bore).
    """

    # Beam model
    beam_gain_model: BeamGainModel

    # The time axis to precompute the dish effects
    model_times: at.Time  # [num_times]

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

    seed: int = 42
    convention: Literal['fourier', 'casa'] = 'fourier'
    dtype: jnp.dtype = jnp.complex64

    def __post_init__(self):
        self.freqs = self.beam_gain_model.freqs
        self.num_antenna = self.beam_gain_model.num_antenna

        # make sure all 1D
        if self.model_times.isscalar:
            self.model_times = self.model_times.reshape((1,))

        self.num_time = len(self.model_times)
        self.num_freq = len(self.freqs)
        self.wavelengths = constants.c / self.freqs

        # Check shapes
        if len(self.freqs.shape) != 1:
            raise ValueError(f"Expected freqs to have 1 dimension but got {len(self.freqs.shape)}")
        if len(self.model_times.shape) != 1:
            raise ValueError(f"Expected times to have 1 dimension but got {len(self.model_times.shape)}")

        # Ensure phi,theta,freq units congrutent

        if not self.dish_diameter.unit.is_equivalent(au.m):
            raise ValueError(f"Expected dish_diameter to be in length units but got {self.dish_diameter.unit}")
        if not self.focal_length.unit.is_equivalent(au.m):
            raise ValueError(f"Expected focal_length to be in length units but got {self.focal_length.unit}")
        if not self.freqs.unit.is_equivalent(au.Hz):
            raise ValueError(f"Expected freqs to be in Hz but got {self.freqs.unit}")
        if not self.elevation_pointing_error_stddev.unit.is_equivalent(au.rad):
            raise ValueError(
                f"Expected elevation_pointing_error_stddev to be in angular units but got {self.elevation_pointing_error_stddev.unit}")
        if not self.cross_elevation_pointing_error_stddev.unit.is_equivalent(au.rad):
            raise ValueError(
                f"Expected cross_elevation_pointing_error_stddev to be in angular units but got {self.cross_elevation_pointing_error_stddev.unit}")
        if not self.axial_focus_error_stddev.unit.is_equivalent(au.m):
            raise ValueError(
                f"Expected axial_focus_error_stddev to be in length units but got {self.axial_focus_error_stddev.unit}")
        if not self.elevation_feed_offset_stddev.unit.is_equivalent(au.m):
            raise ValueError(
                f"Expected elevation_feed_offset_stddev to be in length units but got {self.elevation_feed_offset_stddev.unit}")
        if not self.cross_elevation_feed_offset_stddev.unit.is_equivalent(au.m):
            raise ValueError(
                f"Expected cross_elevation_feed_offset_stddev to be in length units but got {self.cross_elevation_feed_offset_stddev.unit}")
        if not self.horizon_peak_astigmatism_stddev.unit.is_equivalent(au.m):
            raise ValueError(
                f"Expected horizon_peak_astigmatism_stddev to be in length units but got {self.horizon_peak_astigmatism_stddev.unit}")
        if not self.surface_error_mean.unit.is_equivalent(au.m):
            raise ValueError(
                f"Expected surface_error_mean to be in length units but got {self.surface_error_mean.unit}")
        if not self.surface_error_stddev.unit.is_equivalent(au.m):
            raise ValueError(
                f"Expected surface_error_stddev to be in length units but got {self.surface_error_stddev.unit}")

        # Generate the dish effects, broadcasts with [num_antenna, num_freq] ater time interpolation
        keys = jax.random.split(jax.random.PRNGKey(self.seed), 7)
        self.elevation_point_error = self.elevation_pointing_error_stddev * jax.random.normal(
            key=keys[0],
            shape=(self.num_time, self.num_antenna, 1)
        )
        self.cross_elevation_point_error = self.cross_elevation_pointing_error_stddev * jax.random.normal(
            key=keys[1],
            shape=(self.num_time, self.num_antenna, 1)
        )
        self.axial_focus_error = self.axial_focus_error_stddev * jax.random.normal(
            key=keys[2],
            shape=(self.num_time, self.num_antenna, 1)
        )
        self.elevation_feed_offset = self.elevation_feed_offset_stddev * jax.random.normal(
            key=keys[3],
            shape=(self.num_antenna, 1)
        )
        self.cross_elevation_feed_offset = self.cross_elevation_feed_offset_stddev * jax.random.normal(
            key=keys[4],
            shape=(self.num_antenna, 1)
        )
        self.horizon_peak_astigmatism = self.horizon_peak_astigmatism_stddev * jax.random.normal(
            key=keys[5],
            shape=(self.num_antenna, 1)
        )
        self.surface_error = self.surface_error_mean + self.surface_error_stddev * jax.random.normal(
            key=keys[6],
            shape=(self.num_antenna, 1)
        )
        # Compute aperture amplitude
        self.sampling_interval = au.Quantity(jnp.min(self.wavelengths), unit=self.wavelengths.unit)
        self.dx = self.dy = self.sampling_interval / 2.
        # 2*R = sampling_interval * (2 * n + 1)
        n = int(self.dish_diameter / self.sampling_interval) + 1

        yvec = np.arange(-n, n + 1) * self.dy
        xvec = np.arange(-n, n + 1) * self.dx
        self.X, self.Y = np.meshgrid(xvec, yvec, indexing='ij')

        self.dl = self.dm = (1. / n) * au.dimensionless_unscaled
        self.lvec = np.arange(-n, n + 1) * self.dl
        self.mvec = np.arange(-n, n + 1) * self.dm
        M, L = np.meshgrid(self.mvec, self.lvec, indexing='ij')
        N = np.sqrt(1. - L ** 2 - M ** 2)
        self.lmn_data = au.Quantity(jnp.stack([L, M, N], axis=-1))  # [2n+1, 2n+1, 3]
        self.evanescent_mask = np.isnan(N)  # [2n+1, 2n+1]
        self.aperture_amplitude = self.compute_aperture_amplitude()  # [2n+1, 2n+1, num_freq]

    @partial(jax.jit, static_argnames=['self'])
    def _compute_aperture_amplitude_jax(self, image_amplitude: jax.Array) -> jax.Array:
        evanescent_mask = jnp.asarray(self.evanescent_mask)
        image_amplitude = jnp.where(evanescent_mask[:, :, None], 0., image_amplitude)  # [2n+1, 2n+1, num_freq]
        am = ApertureTransform(convention=self.convention)
        dnu = quantity_to_jnp(self.dl * self.dm / self.sampling_interval ** 2)
        aperture_amplitude = am.to_aperture(
            f_image=image_amplitude, axes=(0, 1), dnu=dnu
        )  # [2n+1, 2n+1, num_freq]
        return aperture_amplitude

    def compute_aperture_amplitude(self) -> jax.Array:
        # Aribtrary location and time and phase_tracking will do.
        array_location = ac.EarthLocation(lat=0, lon=0, height=0)
        time = at.Time('2021-01-01T00:00:00', scale='utc')
        phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
        sources = lmn_to_icrs(self.lmn_data, array_location=array_location, time=time, phase_tracking=phase_tracking)

        gain_amplitude = self.beam_gain_model.compute_beam(
            sources=sources,
            phase_tracking=phase_tracking,
            array_location=array_location,
            time=time
        )  # [2n+1, 2n+1, num_ant, num_freq, 2, 2]
        # TODO: assumes identical antennas.
        # TODO: assumes scalar amplitud, so use [0,0]
        image_amplitude = gain_amplitude[..., 0, :, 0, 0]  # [2n+1, 2n+1, num_freq]
        return self._compute_aperture_amplitude_jax(image_amplitude=image_amplitude)

    def _compute_aperture_field_model_jax(self, time_jd, elevation_rad) -> jax.Array:
        """
        Computes the E-field at the aperture of the dish.

        Args:
            elevation: the elevation

        Returns:
            (shape) + path length distortion in meters
        """

        X = quantity_to_jnp(self.X)
        Y = quantity_to_jnp(self.Y)
        focal_length = quantity_to_jnp(self.focal_length)
        cross_elevation_point_error = quantity_to_jnp(self.cross_elevation_point_error)
        elevation_point_error = quantity_to_jnp(self.elevation_point_error)
        axial_focus_error = quantity_to_jnp(self.axial_focus_error)
        elevation_feed_offset = quantity_to_jnp(self.elevation_feed_offset)
        cross_elevation_feed_offset = quantity_to_jnp(self.cross_elevation_feed_offset)
        horizon_peak_astigmatism = quantity_to_jnp(self.horizon_peak_astigmatism)
        surface_error = quantity_to_jnp(self.surface_error)
        dish_diameter = quantity_to_jnp(self.dish_diameter)
        R = 0.5 * dish_diameter
        wavelengths = quantity_to_jnp(self.wavelengths)

        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(time_jd, jnp.asarray(self.model_times.jd))

        def _interp_time(x):
            output = x[i0] * alpha0 + x[i1] * alpha1
            return output

        X = X[:, :, None, None]  # [2n+1, 2n+1, 1, 1]
        Y = Y[:, :, None, None]  # [2n+1, 2n+1, 1, 1]

        pointing_error = _interp_time(elevation_point_error) * X - _interp_time(
            cross_elevation_point_error) * Y

        r = jnp.sqrt(X ** 2 + Y ** 2)
        focal_ratio = r / focal_length  # [2n+1, 2n+1, 1, 1]

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

        total_path_length_error = pointing_error + feed_shift_error + astigmatism_error + surface_error  # [2n+1, 2n+1, num_ant, 1]

        if self.convention == 'casa':
            constant = jnp.asarray(2j * jnp.pi, self.dtype)  # [num_freqs]
        elif self.convention == 'fourier':
            constant = jnp.asarray(-2j * jnp.pi, self.dtype)  # [num_freqs]
        else:
            raise ValueError(f"Unknown convention {self.convention}")

        aperture_field = jnp.exp(constant * total_path_length_error / wavelengths)
        aperture_field *= self.aperture_amplitude[..., None, :]  # [2n+1, 2n+1, num_ant, num_freq]
        return aperture_field

    def compute_aperture_field_model(self, time: at.Time, elevation: au.Quantity) -> au.Quantity:
        """
        Computes the E-field at the aperture of the dish.

        Args:
            elevation: the elevation

        Returns:
            (shape) + path length distortion in meters
        """
        return self._compute_aperture_field_model_jax(
            time_jd=time.jd,
            elevation_rad=quantity_to_jnp(elevation)
        )

    def _compute_beam_fft_jax(self, lmn_sources: jax.Array, aperture_field: jax.Array) -> jax.Array:
        mvec = quantity_to_jnp(self.mvec)
        lvec = quantity_to_jnp(self.lvec)
        # Fourier to fair-field using FFT
        am = ApertureTransform(convention=self.convention)
        dx = quantity_to_jnp(self.dx * self.dy)
        image_field = am.to_image(
            f_aperture=aperture_field, axes=(0, 1), dx=dx
        )  # [2n+1, 2n+1, num_ant, num_freq]

        image_field = multilinear_interp_2d(
            x=lmn_sources[:, 1], y=lmn_sources[:, 0],
            xp=mvec, yp=lvec, z=image_field,
        )  # [num_sources, num_ant, num_freq]
        return image_field

    def _compute_beam_dft_jax(self, lmn_sources: jax.Array, aperture_field: jax.Array):
        wavelengths = quantity_to_jnp(self.wavelengths)
        Y = quantity_to_jnp(self.Y)
        X = quantity_to_jnp(self.X)
        # Opposing constant
        if self.convention == 'casa':
            constant = jnp.asarray(-2j * jnp.pi, self.dtype)  # [num_freqs]
        elif self.convention == 'fourier':
            constant = jnp.asarray(2j * jnp.pi, self.dtype)  # [num_freqs]
        else:
            raise ValueError(f"Unknown convention {self.convention}")

        Y_lambda = Y[:, :, None, None, None] / wavelengths  # [2n+1, 2n+1, 1, 1, num_freqs]
        X_lambda = X[:, :, None, None, None] / wavelengths  # [2n+1, 2n+1, 1, 1, num_freqs]
        l_sources = lmn_sources[:, None, None, 0]  # [num_sources, 1, 1]
        m_sources = lmn_sources[:, None, None, 1]  # [num_sources, 1, 1]

        aperture_field = aperture_field[:, :, None, :, :]  # [2n+1, 2n+1, 1, num_ant, num_freq]

        # L = -Y axis, M = X axis
        unity_root = jnp.exp(
            constant * (-Y_lambda * l_sources + X_lambda * m_sources)
        )  # [2n+1, 2n+1, num_sources, 1, num_freqs]
        image_field = (unity_root * aperture_field)  # [2n+1, 2n+1, num_sources, num_ant, num_freq]

        dx = quantity_to_jnp(self.dx * self.dy)
        image_field = jnp.sum(image_field, axis=(0, 1)) * dx  # [num_sources, num_ant, num_freqs]
        return image_field

    @partial(jax.jit, static_argnames=['self', 'mode'])
    def _compute_beam_jax(self, lmn_sources: jax.Array, time_jd: jax.Array, elevation_rad: jax.Array, mode: str):
        shape = np.shape(lmn_sources)[:-1]
        lmn_sources = lmn_sources.reshape((-1, 3))

        aperture_field = self._compute_aperture_field_model_jax(
            time_jd=time_jd, elevation_rad=elevation_rad
        )  # [2n+1, 2n+1, num_antenna, num_freq]
        if mode == 'fft':
            image_field = self._compute_beam_fft_jax(
                lmn_sources=lmn_sources,
                aperture_field=aperture_field
            )
        elif mode == 'dft':
            image_field = self._compute_beam_dft_jax(
                lmn_sources=lmn_sources,
                aperture_field=aperture_field
            )
        else:
            raise ValueError(f"Unknown mode {mode}")

        image_field = image_field.reshape(shape + image_field.shape[1:])  # (source_shape) + [num_ant, num_freqs]
        # set diagonal
        gains = jnp.zeros(image_field.shape + (2, 2), self.dtype)
        gains = gains.at[..., 0, 0].set(image_field)
        gains = gains.at[..., 1, 1].set(image_field)
        return gains

    def compute_beam(self, sources: ac.ICRS, phase_tracking: ac.ICRS, array_location: ac.EarthLocation, time: at.Time,
                     mode: str = 'fft'):

        altaz_frame = ac.AltAz(location=array_location, obstime=time)
        elevation = phase_tracking.transform_to(altaz_frame).alt
        lmn_sources = icrs_to_lmn(
            sources=sources,
            array_location=array_location,
            time=time,
            phase_tracking=phase_tracking
        )  # (source_shape) + [3]

        return self._compute_beam_jax(
            lmn_sources=quantity_to_jnp(lmn_sources),
            time_jd=jnp.asarray(time.jd),
            elevation_rad=quantity_to_jnp(elevation),
            mode=mode
        )


@pytest.mark.parametrize('mode', ['fft', 'dft'])
def test_dish_effects_gain_model_real_data(mode):
    freqs = au.Quantity([700e6, 2000e6], unit=au.Hz)
    beam_gain_model = beam_gain_model_factor(freqs=freqs, array_name='dsa2000W')

    dish_effects_gain_model = DishEffectsGainModel(
        beam_gain_model=beam_gain_model,
        model_times=at.Time(['2021-01-01T00:00:00', '2021-01-01T00:01:00'], scale='utc')
    )

    assert jnp.allclose(dish_effects_gain_model.dy / dish_effects_gain_model.dl, 2.5 * au.m, atol=0.1)
    assert jnp.all(jnp.isfinite(dish_effects_gain_model.aperture_amplitude))
    assert dish_effects_gain_model.dx.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.dy.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.dl.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.dm.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.lmn_data.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.X.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.Y.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.lvec.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.mvec.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.wavelengths.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.sampling_interval.unit.is_equivalent(au.m)

    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg, )
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:30', scale='utc')

    if mode == 'dft':
        sources = ac.ICRS(ra=[0, 0.] * au.deg, dec=[0., 1.] * au.deg)
    elif mode == 'fft':
        sources = lmn_to_icrs(dish_effects_gain_model.lmn_data, array_location=array_location, time=time,
                              phase_tracking=phase_tracking)
    else:
        raise ValueError(f"Unknown mode {mode}")

    aperture_field = dish_effects_gain_model.compute_aperture_field_model(
        time=time,
        elevation=90. * au.deg
    )  # [2n+1, 2n+1, num_ant, num_freq]

    plt.imshow(
        jnp.abs(aperture_field[:, :, 0, 0]),
        origin='lower',
        extent=(dish_effects_gain_model.X.min().value, dish_effects_gain_model.X.max().value,
                dish_effects_gain_model.Y.min().value, dish_effects_gain_model.Y.max().value)
    )
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.colorbar()
    plt.show()

    plt.imshow(
        jnp.angle(aperture_field[:, :, 0, 0]),
        origin='lower',
        extent=(dish_effects_gain_model.X.min().value, dish_effects_gain_model.X.max().value,
                dish_effects_gain_model.Y.min().value, dish_effects_gain_model.Y.max().value)
    )
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.colorbar()
    plt.show()

    gains = dish_effects_gain_model.compute_beam(
        sources=sources,
        phase_tracking=phase_tracking,
        array_location=array_location,
        time=time,
        mode=mode
    )
    if mode == 'fft':
        plt.imshow(
            jnp.abs(gains[:, :, 0, 0, 0, 0]),
            origin='lower',
            extent=(
                dish_effects_gain_model.mvec.min().value,
                dish_effects_gain_model.mvec.max().value,
                dish_effects_gain_model.lvec.min().value,
                dish_effects_gain_model.lvec.max().value)
        )
        plt.colorbar()
        plt.show()
    else:
        print(gains[..., 0:2, :, 0, 0])
    assert gains.shape == sources.shape + (
        dish_effects_gain_model.num_antenna, len(dish_effects_gain_model.freqs), 2, 2)
