import dataclasses

import numpy as np
from astropy import units as au, coordinates as ac, time as at

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.gain_models.gain_model import GainModel


def lmn_from_phi_theta(phi, theta):
    """
    Convert phi, theta to cartesian coordinates.
    Right-handed X-Y-Z(bore), with phi azimuthal measured from X, and theta from Z

    L-M frame is the same as (-Y)-X frame, i.e. Y is -L and X is M.

    Args:
        phi: in [0, 2pi]
        theta: in [0, pi]

    Returns:
        lmn: [3] array
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    l = -y
    m = x
    n = z
    return l, m, n


@dataclasses.dataclass(eq=False)
class BeamGainModel(GainModel):
    """
    Uses nearest neighbour interpolation to compute the gain model.

    The antennas have attenuation models in frame of antenna, call this the X-Y frame (see below).
    X points up, Y points to the right, Z points towards the source (along bore).
    """
    freqs: au.Quantity  # [num_freqs]
    theta: au.Quantity  # [num_dir] # Theta is in [0, 180] measured from bore-sight
    phi: au.Quantity  # [num_dir] # Phi is in [0, 360] measured from x-axis
    amplitude: au.Quantity  # [num_dir, num_freqs]
    num_antenna: int

    dtype: np.dtype = np.complex64

    def __post_init__(self):
        # make sure all 1D
        if self.freqs.isscalar:
            self.freqs = self.freqs.reshape((1,))
        if self.theta.isscalar:
            self.theta = self.theta.reshape((1,))
        if self.phi.isscalar:
            self.phi = self.phi.reshape((1,))

        # Check shapes
        if len(self.freqs.shape) != 1:
            raise ValueError(f"Expected freqs to have 1 dimension but got {len(self.freqs.shape)}")
        if len(self.theta.shape) != 1:
            raise ValueError(f"Expected theta to have 1 dimension but got {len(self.theta.shape)}")
        if len(self.phi.shape) != 1:
            raise ValueError(f"Expected phi to have 1 dimension but got {len(self.phi.shape)}")
        if len(self.amplitude.shape) != 2:
            raise ValueError(f"Expected amplitude to have 2 dimensions but got {len(self.amplitude.shape)}")
        if self.amplitude.shape != (self.theta.shape[0], self.freqs.shape[0]):
            raise ValueError(
                f"amplitude shape {self.amplitude.shape} does not match theta shape {self.theta.shape} "
                f"and freqs shape {self.freqs.shape}."
            )

        # Ensure phi,theta,freq units congrutent
        if not self.theta.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected theta to be in degrees but got {self.theta.unit}")
        if not self.phi.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected phi to be in degrees but got {self.phi.unit}")
        if not self.freqs.unit.is_equivalent(au.Hz):
            raise ValueError(f"Expected freqs to be in Hz but got {self.freqs.unit}")
        # Amplitude is probably in units, but not worried about it for now

    def compute_beam(self, sources: ac.ICRS, phase_tracking: ac.ICRS, array_location: ac.EarthLocation, time: at.Time):
        shape = sources.shape
        sources = sources.reshape((-1,))
        # Convert phi,theta to lmn coordinates, where Y-X frame matches L-M frame
        # First to cartesian
        y, x, z_bore = lmn_from_phi_theta(phi=self.phi.to('rad').value, theta=self.theta.to('rad').value)
        lmn_data = au.Quantity(np.stack([y, x, z_bore], axis=-1), unit=au.dimensionless_unscaled)  # [num_dir, 3]

        lmn_sources = icrs_to_lmn(
            sources=sources,
            array_location=array_location,
            time=time,
            phase_tracking=phase_tracking
        )  # [source_shape, 3]
        # Find the nearest neighbour of each source to that in data

        dist_sq = np.sum(np.square(lmn_sources[:, None, :] - lmn_data[None, :, :]), axis=-1)  # [num_sources, num_dir]
        closest = np.argmin(dist_sq, axis=-1)  # [num_sources]

        amplitude = self.amplitude[closest, :]  # [num_sources, num_freqs]
        amplitude = np.repeat(amplitude[:, None, :], self.num_antenna, axis=1)  # [num_sources, num_ant, num_freqs]
        # set diagonal
        gains = np.zeros(amplitude.shape + (2, 2), self.dtype)
        gains[..., 0, 0] = amplitude
        gains[..., 1, 1] = amplitude

        gains = gains.reshape(shape + gains.shape[1:])

        return gains


def test_beam_gain_model():
    freqs = au.Quantity([1000, 2000], unit=au.Hz)
    theta = au.Quantity([0, 90], unit=au.deg)
    phi = au.Quantity([0, 90], unit=au.deg)
    amplitude = au.Quantity([[1, 2], [3, 4]], unit=au.dimensionless_unscaled)
    num_antenna = 5

    beam_gain_model = BeamGainModel(
        freqs=freqs,
        theta=theta,
        phi=phi,
        amplitude=amplitude,
        num_antenna=num_antenna
    )

    sources = ac.ICRS(ra=[0, 1] * au.deg, dec=[2, 3] * au.deg).reshape((2, 1))
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg, )
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    gains = beam_gain_model.compute_beam(
        sources=sources,
        phase_tracking=phase_tracking,
        array_location=array_location,
        time=time
    )

    assert gains.shape == sources.shape + (num_antenna, len(freqs), 2, 2)


def test_beam_gain_model_real_data():
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
    # print(beam_gain_model)

    sources = ac.ICRS(ra=[0, 1] * au.deg, dec=[2, 3] * au.deg)
    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg, )
    array_location = ac.EarthLocation(lat=0, lon=0, height=0)
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    gains = beam_gain_model.compute_beam(
        sources=sources, phase_tracking=phase_tracking, array_location=array_location, time=time
    )

    # print(gains)
    assert gains.shape == (len(sources), num_antenna, len(freqs), 2, 2)
