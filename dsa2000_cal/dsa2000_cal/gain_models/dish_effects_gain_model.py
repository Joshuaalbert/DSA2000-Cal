import dataclasses
from typing import Literal

import numpy as np
from astropy import units as au, coordinates as ac, time as at, constants

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.coord_utils import lmn_to_icrs
from dsa2000_cal.gain_models.beam_gain_model import BeamGainModel
from dsa2000_cal.gain_models.gain_model import GainModel


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
        if not self.freqs.unit.is_equivalent(au.Hz):
            raise ValueError(f"Expected freqs to be in Hz but got {self.freqs.unit}")

        # Generate the dish effects
        np.random.seed(self.seed)
        self.elevation_point_error = self.elevation_pointing_error_stddev * np.random.normal(
            size=(self.num_time, self.num_antenna)
        )
        self.cross_elevation_point_error = self.cross_elevation_pointing_error_stddev * np.random.normal(
            size=(self.num_time, self.num_antenna)
        )
        self.axial_focus_error = self.axial_focus_error_stddev * np.random.normal(
            size=(self.num_time, self.num_antenna)
        )
        self.elevation_feed_offset = self.elevation_feed_offset_stddev * np.random.normal(
            size=(1, self.num_antenna)
        )
        self.cross_elevation_feed_offset = self.cross_elevation_feed_offset_stddev * np.random.normal(
            size=(1, self.num_antenna)
        )
        self.horizon_peak_astigmatism = self.horizon_peak_astigmatism_stddev * np.random.normal(
            size=(1, self.num_antenna)
        )
        self.surface_error = self.surface_error_mean + self.surface_error_stddev * np.random.normal(
            size=(1, self.num_antenna)
        )
        n = int(np.max(self.dish_diameter / self.wavelengths))
        dx = dy = self.dish_diameter / (2. * n)
        self.y, self.x = np.meshgrid(np.arange(-n, n + 1) * dy, np.arange(-n, n + 1) * dx, indexing='ij')
        self.aperture_amplitude = self.compute_aperture_amplitude()

    def compute_aperture_amplitude(self):

        dl = dm = np.min(self.wavelengths / self.dish_diameter)
        n = int(np.max(self.dish_diameter / self.wavelengths))
        lvec = np.arange(-n, n + 1) * dl
        mvec = np.arange(-n, n + 1) * dm
        L, M = np.meshgrid(lvec, mvec, indexing='ij')
        N = np.sqrt(1 - L ** 2 - M ** 2)
        lmn = np.stack([L, M, N], axis=-1) * dl.unit  # [2n+1, 2n+1, 3]
        array_location = ac.EarthLocation(lat=0, lon=0, height=0)
        time = at.Time('2021-01-01T00:00:00', scale='utc')
        phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
        sources = lmn_to_icrs(lmn, array_location=array_location, time=time, phase_tracking=phase_tracking)
        gain_amplitude = self.beam_gain_model.compute_beam(
            sources=sources,
            phase_tracking=phase_tracking,
            array_location=array_location,
            time=time
        )  # [2n+1, 2n+1, num_ant, num_freq, 2, 2]
        amplitude = gain_amplitude[..., 0, 0]  # [2n+1, 2n+1, num_ant, num_freq]
        aperture_amplitude = np.fft.rfftn(amplitude, axes=(0, 1))
        return aperture_amplitude

    def path_length_distortion_model(self, elevation: au.Quantity) -> au.Quantity:
        """
        Gets the pathlength distortions in the dish aperature.

        Args:
            elevation: the elevation

        Returns:
            (shape) + path length distortion in meters
        """
        shape = np.shape(self.x)

        pointing_error = self.elevation_point_error * self.x - self.cross_elevation_point_error * self.y

        r = np.sqrt(self.x ** 2 + self.y ** 2)
        focal_ratio = r / self.focal_length

        sin_theta_p = focal_ratio / (1. + 0.25 * focal_ratio ** 2)
        cos_theta_p = (1. - 0.25 * focal_ratio ** 2) / (1. + 0.25 * focal_ratio ** 2)

        cos_phi = self.x / r
        sin_phi = self.y / r

        feed_shift_error = (
                self.axial_focus_error * cos_theta_p
                - self.elevation_feed_offset * sin_theta_p * cos_phi
                - self.cross_elevation_feed_offset * sin_theta_p * sin_phi
        )
        cos_2phi = 2. * cos_phi ** 2 - 1.
        cos_elevation = np.cos(elevation.to('rad').value)
        peak_astigmatism = self.horizon_peak_astigmatism * cos_elevation
        astigmatism_error = peak_astigmatism * r ** 2 * cos_2phi

        return pointing_error + feed_shift_error + astigmatism_error

    def compute_beam(self, sources: ac.ICRS, phase_tracking: ac.ICRS, array_location: ac.EarthLocation, time: at.Time):

        shape = sources.shape
        sources = sources.reshape((-1,))

        altaz_frame = ac.AltAz(location=array_location, obstime=time)
        elevation = phase_tracking.transform_to(altaz_frame).elevation

        aperture_phase = ...

        # lmn_sources = icrs_to_lmn(
        #     sources=sources,
        #     array_location=array_location,
        #     time=time,
        #     phase_tracking=phase_tracking
        # )  # [source_shape, 3]
        # # Find the nearest neighbour of each source to that in data
        #
        # dist_sq = np.sum(np.square(lmn_sources[:, None, :] - lmn_data[None, :, :]), axis=-1)  # [num_sources, num_dir]
        # closest = np.argmin(dist_sq, axis=-1)  # [num_sources]
        #
        # amplitude = self.amplitude[closest, :]  # [num_sources, num_freqs]
        # amplitude = np.repeat(amplitude[:, None, :], self.num_antenna, axis=1)  # [num_sources, num_ant, num_freqs]
        # # set diagonal
        # gains = np.zeros(amplitude.shape + (2, 2), self.dtype)
        # gains[..., 0, 0] = amplitude
        # gains[..., 1, 1] = amplitude
        #
        # return gains


def test_dish_effects_gain_model_real_data():
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
        times=at.Time('2021-01-01T00:00:00', scale='utc'),
        num_antenna=num_antenna
    )
    # print(dish_effects_gain_model)

    print(dish_effects_gain_model.aperture_amplitude.shape)

    # sources = ac.ICRS(ra=[0, 1] * au.deg, dec=[2, 3] * au.deg)
    # phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg, )
    # array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    # time = at.Time('2021-01-01T00:00:00', scale='utc')
    #
    # gains = beam_gain_model.compute_beam(
    #     sources=sources, phase_tracking=phase_tracking, array_location=array_location, time=time
    # )
    #
    # # print(gains)
    # assert gains.shape == (len(sources), num_antenna, len(freqs), 2, 2)
