import dataclasses
from typing import Literal

import numpy as np
import pylab as plt
import pytest
from astropy import units as au, coordinates as ac, time as at, constants

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.coord_utils import lmn_to_icrs, icrs_to_lmn
from dsa2000_cal.common.uniform_interp import multilinear_interp_2d
from dsa2000_cal.gain_models.beam_gain_model import BeamGainModel
from dsa2000_cal.gain_models.gain_model import GainModel, get_interp_indices_and_weights


@dataclasses.dataclass(eq=False)
class ApertureTransform:
    """
    A class to transform between aperture and image planes.

    For fourier convention, the transform is defined as:

    .. math::

            f_image = int f_aperture(x) e^{2i pi x nu} dx
            f_aperture = int f_image(nu) e^{-2i pi x nu} dnu

    For casa convention, the transform is defined as:

    .. math::

            f_image = int f_aperture(x) e^{-2i pi x nu} dx
            f_aperture = int f_image(nu) e^{2i pi x nu} dnu

    """
    convention: str = 'fourier'

    def to_image(self, f_aperture, axes, dx):
        if self.convention == 'fourier':
            return self._to_image_fourier(f_aperture, axes, dx)
        elif self.convention == 'casa':
            return self._to_image_casa(f_aperture, axes, dx)
        else:
            raise ValueError(f"Unknown convention {self.convention}")

    def to_aperture(self, f_image, axes, dnu):
        if self.convention == 'fourier':
            return self._to_aperture_fourier(f_image, axes, dnu)
        elif self.convention == 'casa':
            return self._to_aperture_casa(f_image, axes, dnu)
        else:
            raise ValueError(f"Unknown convention {self.convention}")

    def _to_aperture_fourier(self, f_image, axes, dnu):
        # undo uses -2pi convention so fft is used
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f_image, axes=axes), axes=axes), axes=axes) * dnu

    def _to_image_fourier(self, f_aperture, axes, dx):
        factor = np.prod([f_aperture.shape[axis] for axis in axes])
        # uses -2pi convention so ifft is used
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f_aperture, axes=axes), axes=axes),
                               axes=axes) * dx * factor

    def _to_aperture_casa(self, f_image, axes, dnu):
        # uses +2pi convention so ifft is used
        factor = np.prod([f_image.shape[axis] for axis in axes])
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f_image, axes=axes), axes=axes), axes=axes) * dnu * factor

    def _to_image_casa(self, f_aperture, axes, dx):
        # uses +2pi convention so ifft is used
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f_aperture, axes=axes), axes=axes), axes=axes) * dx


@pytest.mark.parametrize('convention', ['fourier', 'casa'])
def test_fourier_conventions(convention):
    dx = 0.1
    n = 100
    x = np.arange(-n, n + 1) * dx
    nu = np.fft.fftshift(np.fft.fftfreq(2 * n + 1, dx))
    dnu = nu[1] - nu[0]
    f_aperture = np.exp(-x ** 2) + x

    am = ApertureTransform(convention=convention)

    f_image = am.to_image(f_aperture, axes=(0,), dx=dx)
    plt.plot(nu,
             np.abs(f_image))  # This shows the gaussian shifted with peak split up! I expected it to be in the middle
    plt.title(convention + ': image')
    plt.show()

    rec_f_aperture = am.to_aperture(f_image, axes=(0,), dnu=dnu)
    # These agree and the gaussian is at the centre of both plots.
    plt.plot(x, np.abs(rec_f_aperture))
    plt.plot(x, np.abs(f_aperture))
    plt.title(convention + ': aperture')
    plt.show()

    # This passes for both conventions
    np.testing.assert_allclose(f_aperture, rec_f_aperture, atol=1e-6)

    # If we run with 'casa' convention, the plots all have mode in centre

    f_image = np.exp(-nu ** 2) + nu

    f_aperture = am.to_aperture(f_image, axes=(0,), dnu=dnu)
    plt.plot(nu,
             np.abs(
                 f_aperture))  # This shows the gaussian shifted with peak split up! I expected it to be in the middle
    plt.title(convention + ': aperture')
    plt.show()

    rec_f_image = am.to_image(f_aperture, axes=(0,), dx=dx)
    # These agree and the gaussian is at the centre of both plots.
    plt.plot(x, np.abs(rec_f_image))
    plt.plot(x, np.abs(f_image))
    plt.title(convention + ': image')
    plt.show()

    # This passes for both conventions
    np.testing.assert_allclose(f_image, rec_f_image, atol=1e-6)


@dataclasses.dataclass(eq=False)
class DishEffectsGainModel(GainModel):
    """
    Uses nearest neighbour interpolation to compute the gain model.

    The antennas have attenuation models in frame of antenna, call this the X-Y frame (see below).
    X points up, Y points to the right, Z points towards the source (along bore).
    """

    beam_gain_model: BeamGainModel

    freqs: au.Quantity  # [num_freqs]
    times: at.Time  # [num_times]
    num_antenna: int

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
    dtype: np.dtype = np.complex64

    def __post_init__(self):
        # make sure all 1D
        if self.freqs.isscalar:
            self.freqs = self.freqs.reshape((1,))
        if self.times.isscalar:
            self.times = self.times.reshape((1,))

        self.num_time = len(self.times)
        self.num_freq = len(self.freqs)
        self.wavelengths = constants.c / self.freqs

        # Check shapes
        if len(self.freqs.shape) != 1:
            raise ValueError(f"Expected freqs to have 1 dimension but got {len(self.freqs.shape)}")
        if len(self.times.shape) != 1:
            raise ValueError(f"Expected times to have 1 dimension but got {len(self.times.shape)}")

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
        np.random.seed(self.seed)
        self.elevation_point_error = self.elevation_pointing_error_stddev * np.random.normal(
            size=(self.num_time, self.num_antenna, 1)
        )
        self.cross_elevation_point_error = self.cross_elevation_pointing_error_stddev * np.random.normal(
            size=(self.num_time, self.num_antenna, 1)
        )
        self.axial_focus_error = self.axial_focus_error_stddev * np.random.normal(
            size=(self.num_time, self.num_antenna, 1)
        )
        self.elevation_feed_offset = self.elevation_feed_offset_stddev * np.random.normal(
            size=(self.num_antenna, 1)
        )
        self.cross_elevation_feed_offset = self.cross_elevation_feed_offset_stddev * np.random.normal(
            size=(self.num_antenna, 1)
        )
        self.horizon_peak_astigmatism = self.horizon_peak_astigmatism_stddev * np.random.normal(
            size=(self.num_antenna, 1)
        )
        self.surface_error = self.surface_error_mean + self.surface_error_stddev * np.random.normal(
            size=(self.num_antenna, 1)
        )
        # Compute aperture amplitude
        self.sampling_interval = np.min(self.wavelengths)
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
        self.lmn_data = au.Quantity(np.stack([L, M, N], axis=-1))  # [2n+1, 2n+1, 3]
        self.evanescent_mask = np.isnan(N)
        self.aperture_amplitude = self.compute_aperture_amplitude()

    def compute_aperture_amplitude(self):
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
        amplitude = gain_amplitude[..., 0, :, 0, 0]  # [2n+1, 2n+1, num_freq]
        amplitude[self.evanescent_mask] = 0.  #
        am = ApertureTransform(convention=self.convention)
        aperture_amplitude = am.to_aperture(
            f_image=amplitude, axes=(0, 1), dnu=self.dl * self.dm / self.sampling_interval ** 2
        )  # [2n+1, 2n+1, num_ant, num_freq]
        return aperture_amplitude

    def compute_aperture_field_model(self, time: at.Time, elevation: au.Quantity) -> au.Quantity:
        """
        Computes the E-field at the aperture of the dish.

        Args:
            elevation: the elevation

        Returns:
            (shape) + path length distortion in meters
        """

        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(time.jd, self.times.jd)

        def _interp_time(x):
            output = x[i0] * alpha0 + x[i1] * alpha1
            return output

        X = self.X[:, :, None, None]  # [2n+1, 2n+1, 1, 1]
        Y = self.Y[:, :, None, None]  # [2n+1, 2n+1, 1, 1]

        pointing_error = _interp_time(self.elevation_point_error.to('rad').value) * X - _interp_time(
            self.cross_elevation_point_error.to('rad').value) * Y

        r = np.sqrt(X ** 2 + Y ** 2)
        focal_ratio = r / self.focal_length  # [2n+1, 2n+1, 1, 1]

        sin_theta_p = focal_ratio / (1. + 0.25 * focal_ratio ** 2)
        cos_theta_p = (1. - 0.25 * focal_ratio ** 2) / (1. + 0.25 * focal_ratio ** 2)

        cos_phi = np.where(r == 0., 1., X / r)
        sin_phi = np.where(r == 0., 0., Y / r)

        feed_shift_error = (
                _interp_time(self.axial_focus_error) * cos_theta_p
                - self.elevation_feed_offset * sin_theta_p * cos_phi
                - self.cross_elevation_feed_offset * sin_theta_p * sin_phi
        )
        cos_2phi = 2. * cos_phi ** 2 - 1.
        cos_elevation = np.cos(elevation.to('rad').value)
        peak_astigmatism = self.horizon_peak_astigmatism * cos_elevation
        astigmatism_error = peak_astigmatism * ((r / (0.5 * self.dish_diameter / np.sqrt(2.))) ** 2 - 1.) * cos_2phi

        total_path_length_error = pointing_error + feed_shift_error + astigmatism_error + self.surface_error  # [2n+1, 2n+1, num_ant, 1]

        if self.convention == 'casa':
            constant = np.asarray(2j * np.pi, self.dtype)  # [num_freqs]
        elif self.convention == 'fourier':
            constant = np.asarray(-2j * np.pi, self.dtype)  # [num_freqs]
        else:
            raise ValueError(f"Unknown convention {self.convention}")

        aperture_field = np.exp(constant * total_path_length_error / self.wavelengths)
        aperture_field *= self.aperture_amplitude[..., None, :]  # [2n+1, 2n+1, num_ant, num_freq]
        return aperture_field

    def compute_beam(self, sources: ac.ICRS, phase_tracking: ac.ICRS, array_location: ac.EarthLocation, time: at.Time,
                     mode: str = 'fft'):

        shape = sources.shape
        sources = sources.reshape((-1,))

        altaz_frame = ac.AltAz(location=array_location, obstime=time)
        elevation = phase_tracking.transform_to(altaz_frame).alt  #

        aperture_field = self.compute_aperture_field_model(
            time=time, elevation=elevation
        )  # [2n+1, 2n+1, num_antenna, num_freq]

        lmn_sources = icrs_to_lmn(
            sources=sources,
            array_location=array_location,
            time=time,
            phase_tracking=phase_tracking
        )  # [num_sources, 3]

        # Find the nearest neighbour of each source to that in data
        lmn_data = self.lmn_data.reshape((-1, 3))  # [num_dir, 3]

        if mode == 'fft':
            # Fourier to fair-field using FFT
            am = ApertureTransform(convention=self.convention)
            image_field = am.to_image(
                f_aperture=aperture_field, axes=(0, 1), dx=self.dx * self.dy
            )  # [2n+1, 2n+1, num_ant, num_freq]

            image_field = multilinear_interp_2d(
                x=lmn_sources[:, 1], y=lmn_sources[:, 0],
                xp=self.mvec, yp=self.lvec, z=image_field,
            )  # [num_sources, num_ant, num_freq]

        elif mode == 'dft':
            # Opposing constant
            if self.convention == 'casa':
                constant = np.asarray(-2j * np.pi, self.dtype)  # [num_freqs]
            elif self.convention == 'fourier':
                constant = np.asarray(2j * np.pi, self.dtype)  # [num_freqs]
            else:
                raise ValueError(f"Unknown convention {self.convention}")

            Y_lambda = self.Y[:, :, None, None, None] / self.wavelengths  # [2n+1, 2n+1, 1, 1, num_freqs]
            X_lambda = self.X[:, :, None, None, None] / self.wavelengths  # [2n+1, 2n+1, 1, 1, num_freqs]
            l_sources = lmn_sources[:, None, None, 0]  # [num_sources, 1, 1]
            m_sources = lmn_sources[:, None, None, 1]  # [num_sources, 1, 1]

            aperture_field = aperture_field[:, :, None, :, :]  # [2n+1, 2n+1, 1, num_ant, num_freq]

            # L = -Y axis, M = X axis
            unity_root = np.exp(
                constant * (-Y_lambda * l_sources + X_lambda * m_sources)
            )  # [2n+1, 2n+1, num_sources, 1, num_freqs]
            image_field = (unity_root * aperture_field)  # [2n+1, 2n+1, num_sources, num_ant, num_freq]
            image_field = np.sum(image_field, axis=(0, 1)) * (self.dx * self.dy)  # [num_sources, num_ant, num_freqs]
        else:
            raise ValueError(f"Unknown mode {mode}")

        image_field = image_field.reshape(shape + image_field.shape[1:])  # (source_shape) + [num_ant, num_freqs]
        # set diagonal
        gains = np.zeros(image_field.shape + (2, 2), self.dtype)
        gains[..., 0, 0] = image_field
        gains[..., 1, 1] = image_field

        return gains


@pytest.mark.parametrize('mode', ['fft', 'dft'])
def test_dish_effects_gain_model_real_data(mode):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    dish_model = array.antenna_beam().get_model()
    theta = dish_model.get_theta() * au.deg
    phi = dish_model.get_phi() * au.deg
    theta, phi = np.meshgrid(theta, phi, indexing='ij')
    theta = theta.reshape((-1,))
    phi = phi.reshape((-1,))
    num_antenna = len(array.get_antennas())

    freqs = dish_model.get_freqs() * au.Hz
    amplitude = dish_model.get_amplitude()  # [num_theta, num_phi, num_freqs]
    amplitude = amplitude.reshape((-1, len(freqs)))  # [num_dir, num_freqs]
    voltage_gain = dish_model.get_voltage_gain()
    amplitude = amplitude / voltage_gain

    beam_gain_model = BeamGainModel(
        freqs=freqs,
        theta=theta,
        phi=phi,
        amplitude=amplitude,
        num_antenna=num_antenna
    )

    dish_effects_gain_model = DishEffectsGainModel(
        beam_gain_model=beam_gain_model,
        freqs=freqs,
        times=at.Time(['2021-01-01T00:00:00', '2021-01-01T00:01:00'], scale='utc'),
        num_antenna=num_antenna
    )
    assert np.allclose(dish_effects_gain_model.dy / dish_effects_gain_model.dl, 2.5 * au.m, atol=0.1)
    assert np.all(np.isfinite(dish_effects_gain_model.aperture_amplitude))
    assert dish_effects_gain_model.dx.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.dy.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.dl.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.dm.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.lmn_data.unit.is_equivalent(au.dimensionless_unscaled)
    assert dish_effects_gain_model.X.unit.is_equivalent(au.m)
    assert dish_effects_gain_model.Y.unit.is_equivalent(au.m)

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
        np.abs(aperture_field[:, :, 0, 0]),
        origin='lower',
        extent=(dish_effects_gain_model.X.min().value, dish_effects_gain_model.X.max().value,
                dish_effects_gain_model.Y.min().value, dish_effects_gain_model.Y.max().value)
    )
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.colorbar()
    plt.show()

    plt.imshow(
        np.angle(aperture_field[:, :, 0, 0]),
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
            np.abs(gains[:, :, 0, 0, 0, 0]),
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
    assert gains.shape == sources.shape + (num_antenna, len(freqs), 2, 2)
