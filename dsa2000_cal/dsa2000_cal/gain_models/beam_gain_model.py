import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as au, coordinates as ac, time as at
from tomographic_kernel.frames import ENU

from dsa2000_cal.assets.content_registry import fill_registries, NoMatchFound
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
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
    x = jnp.sin(theta) * jnp.cos(phi)  # M
    y = jnp.sin(theta) * jnp.sin(phi)  # -L
    bore_z = jnp.cos(theta)
    l = -y
    m = x
    n = bore_z
    return l, m, n


@dataclasses.dataclass(eq=False)
class BeamGainModel(GainModel):
    """
    Uses nearest neighbour interpolation to compute the gain model.

    The antennas have attenuation models in frame of antenna, call this the X-Y frame (see below).
    X points up, Y points to the right, Z points towards the source (along bore).
    """

    model_freqs: au.Quantity  # [num_freqs_in_model]
    model_theta: au.Quantity  # [num_dir] # Theta is in [0, 180] measured from bore-sight
    model_phi: au.Quantity  # [num_dir] # Phi is in [0, 360] measured from x-axis
    model_gains: au.Quantity  # [num_dir, num_freqs, 2, 2]

    dtype: jnp.dtype = jnp.complex64

    def __post_init__(self):
        # make sure all 1D
        if self.model_freqs.isscalar:
            raise ValueError("Expected model_freqs to be an array.")
        if self.model_theta.isscalar:
            raise ValueError("Expected theta to be an array.")
        if self.model_phi.isscalar:
            raise ValueError("Expected phi to be an array.")

        # Check shapes
        if len(self.model_freqs.shape) != 1:
            raise ValueError(f"Expected model_freqs to have 1 dimension but got {len(self.model_freqs.shape)}")
        if len(self.model_theta.shape) != 1:
            raise ValueError(f"Expected theta to have 1 dimension but got {len(self.model_theta.shape)}")
        if len(self.model_phi.shape) != 1:
            raise ValueError(f"Expected phi to have 1 dimension but got {len(self.model_phi.shape)}")
        if self.model_gains.shape != (len(self.model_theta), len(self.model_freqs), 2, 2):
            raise ValueError(
                f"gains shape {self.model_gains.shape} does not match theta shape (num_dir, num_freqs, 2, 2)."
            )

        # Ensure phi,theta,freq units congrutent
        if not self.model_theta.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected theta to be in degrees but got {self.model_theta.unit}")
        if not self.model_phi.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected phi to be in degrees but got {self.model_phi.unit}")
        if not self.model_freqs.unit.is_equivalent(au.Hz):
            raise ValueError(f"Expected model_freqs to be in Hz but got {self.model_freqs.unit}")
        if not self.model_gains.unit.is_equivalent(au.dimensionless_unscaled):
            raise ValueError(f"Expected model_gains to be dimensionless but got {self.model_gains.unit}")

        # Convert phi,theta to lmn coordinates, where Y-X frame matches L-M frame
        # First to cartesian
        l, m, n = lmn_from_phi_theta(
            phi=quantity_to_jnp(self.model_phi, 'rad'),
            theta=quantity_to_jnp(self.model_theta, 'rad')
        )
        self.lmn_data = au.Quantity(np.stack([l, m, n], axis=-1), unit=au.dimensionless_unscaled)  # [num_dir, 3]

    @partial(jax.jit, static_argnums=(0,))
    def _compute_gain_jax(self, freqs: jax.Array, lmn_sources: jax.Array):
        """
        Compute the beam gain at the given source coordinates.

        Args:
            freqs: (num_freqs) The frequencies at which to compute the beam gain.
            lmn_sources: (source_shape) + [num_ant, 3] The source coordinates in the L-M-N frame.

        Returns:
            (source_shape) + [num_ant, num_freq, 2, 2] The beam gain at the given source coordinates.
        """
        lmn_data = quantity_to_jnp(self.lmn_data)
        gains = jnp.asarray(quantity_to_jnp(self.model_gains), dtype=self.dtype)
        model_freqs = quantity_to_jnp(self.model_freqs)

        # Interpolate in freq
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(freqs, model_freqs)
        gains = gains[..., i0, :, :] * alpha0[:, None, None] + gains[..., i1, :, :] * alpha1[:, None,
                                                                                      None]  # [num_dir, num_freqs, 2, 2]

        shape = lmn_sources.shape[:-1]
        lmn_sources = lmn_sources.reshape((-1, 3))
        cos_dist = jnp.sum(lmn_sources[:, None, :] * lmn_data[None, :, :], axis=-1)  # [num_sources, num_dir]
        closest = jnp.nanargmax(cos_dist, axis=-1)  # [num_sources]

        gains = gains[closest, :, :, :]  # [num_sources, num_freqs, 2 ,2]
        evanescent_mask = jnp.isnan(lmn_sources[..., 2])  # [num_sources]
        gains = jnp.where(evanescent_mask[:, None, None, None], jnp.nan,
                          gains)  # [num_sources, num_freqs, 2, 2]

        gains = gains.reshape(shape + gains.shape[1:])  # (source_shape) + [num_ant, num_freq, 2, 2]
        if gains.shape[-4] != len(self.antennas):
            gains = jnp.repeat(gains, len(self.antennas), axis=-4)  # (source_shape) + [num_ant, num_freq, 2, 2]

        return gains

    def compute_gain(self, freqs: au.Quantity, sources: ac.ICRS | ENU, pointing: ac.ICRS | None,
                     array_location: ac.EarthLocation, time: at.Time, **kwargs):

        self.check_inputs(
            freqs=freqs,
            sources=sources,
            pointing=pointing,
            array_location=array_location,
            time=time,
            **kwargs
        )

        if pointing is None:
            # TODO: Use location=self.antennas but manage memory blowup
            pointing = zenith = ENU(east=0, north=0, up=1, location=array_location, obstime=time).transform_to(
                ac.ICRS())  # []

        pointing = pointing.reshape([1] * len(sources.shape) + [-1])  # [1,..., num_ant]
        if isinstance(sources, ENU):
            # relative to antenna positions
            enu_frame = ENU(location=array_location, obstime=time)
            sources = sources.transform_to(enu_frame)  # (source_shape) + [3]
            antennas = self.antennas.get_itrs(obstime=time, location=array_location).transform_to(
                enu_frame
            ).reshape([1] * len(sources.shape) + [-1])  # [1,..., num_ant]
            source_sep = sources.reshape(
                sources.shape + (1,)).cartesian.xyz - antennas.cartesian.xyz  # [3]+ (source_shape) + [num_ant]
            source_sep /= np.linalg.norm(source_sep, axis=0, keepdims=True)
            sources = ENU(east=source_sep[0], north=source_sep[1], up=source_sep[2],
                          location=array_location, obstime=time).transform_to(ac.ICRS())  # (source_shape) + [num_ant]
        elif isinstance(sources, ac.ICRS):
            sources = sources.reshape(sources.shape + (1,))  # (source_shape) + [1]
        else:
            raise ValueError(f"Expected sources to be ICRS or ENU but got {sources}")

        lmn_sources = icrs_to_lmn(sources=sources, time=time,
                                  phase_tracking=pointing)  # (source_shape) + [num_ant, 3]
        print(lmn_sources)
        gains = self._compute_gain_jax(
            freqs=quantity_to_jnp(freqs),
            lmn_sources=quantity_to_jnp(lmn_sources)
        )  # (source_shape) + [num_ant, num_freq, 2, 2]

        return gains


def beam_gain_model_factory(array_name: str, **kwargs) -> BeamGainModel:
    fill_registries()
    try:
        array = array_registry.get_instance(array_registry.get_match(array_name))
    except NoMatchFound as e:
        raise ValueError(f"Array {array_name} not found in registry. Add it to use the BeamGainModel factory.") from e

    dish_model = array.get_antenna_model()
    theta = dish_model.get_theta()
    phi = dish_model.get_phi()
    theta, phi = np.meshgrid(theta, phi, indexing='ij')
    theta = theta.reshape((-1,))
    phi = phi.reshape((-1,))

    antennas = array.get_antennas()
    model_freqs = dish_model.get_freqs()
    amplitude = dish_model.get_amplitude()  # [num_theta, num_phi, num_freqs, 2, 2]
    voltage_gain = dish_model.get_voltage_gain()  # [num_freqs]
    amplitude = au.Quantity(amplitude / voltage_gain[:, None, None])  # [num_theta, num_phi, num_freqs, 2, 2]
    print(amplitude.max(), amplitude.min())

    phase = dish_model.get_phase()  # [num_theta, num_phi, num_freqs, 2, 2]
    gains = amplitude * np.exp(1j * phase.to('rad').value)  # [num_theta, num_phi, num_freqs, 2, 2]
    gains = gains.reshape((-1, len(model_freqs), 2, 2)) * au.dimensionless_unscaled  # [num_dir, num_freqs, 2, 2]
    beam_gain_model = BeamGainModel(
        antennas=antennas,
        model_freqs=model_freqs,
        model_theta=theta,
        model_phi=phi,
        model_gains=gains,
        **kwargs
    )
    return beam_gain_model
