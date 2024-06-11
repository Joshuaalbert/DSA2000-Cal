import dataclasses
from functools import partial

import jax
import numpy as np
from astropy import units as au, coordinates as ac, time as at
from jax import numpy as jnp, lax
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights, apply_interp
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.gain_model import GainModel


@dataclasses.dataclass(eq=False)
class SphericalInterpolatorGainModel(GainModel):
    """
    Uses nearest neighbour interpolation to compute the gain model.

    The antennas have attenuation models in frame of antenna, call this the X-Y frame (see below).
    X points up, Y points to the right, Z points towards the source (along bore).

    Args:
        model_freqs: [num_model_freqs] The frequencies at which the model is defined.
        model_theta: [num_model_dir] The theta values in [0, 180] measured from bore-sight.
        model_phi: [num_model_dir] The phi values in [0, 360] measured from x-axis.
        model_times: [num_model_times] The times at which the model is defined.
        model_gains: [num_model_times, num_model_dir, [num_ant,] num_model_freqs, 2, 2] The gain model.
            If tile_antennas=False then the model gains much include antenna dimension, otherwise they are assume
            identical per antenna and tiled.
        tile_antennas: If True, the model gains are assumed to be identical for each antenna and are tiled.
        dtype: The dtype of the model gains.
    """

    model_freqs: au.Quantity  # [num_model_freqs]
    model_theta: au.Quantity  # [num_model_dir] # Theta is in [0, 180] measured from bore-sight
    model_phi: au.Quantity  # [num_model_dir] # Phi is in [0, 360] measured from x-axis
    model_times: at.Time  # [num_model_times] # Times at which the model is defined
    model_gains: au.Quantity  # [num_model_times, num_model_dir, [num_ant,] num_model_freqs, 2, 2]

    tile_antennas: bool = False
    use_scan: bool = False

    dtype: jnp.dtype = jnp.complex64

    def __post_init__(self):
        # make sure all 1D
        if self.model_freqs.isscalar:
            raise ValueError("Expected model_freqs to be an array.")
        if self.model_theta.isscalar:
            raise ValueError("Expected theta to be an array.")
        if self.model_phi.isscalar:
            raise ValueError("Expected phi to be an array.")
        if self.model_times.isscalar:
            raise ValueError("Expected model_times to be an array.")

        # Check shapes
        if len(self.model_freqs.shape) != 1:
            raise ValueError(f"Expected model_freqs to have 1 dimension but got {len(self.model_freqs.shape)}")
        if len(self.model_theta.shape) != 1:
            raise ValueError(f"Expected theta to have 1 dimension but got {len(self.model_theta.shape)}")
        if len(self.model_phi.shape) != 1:
            raise ValueError(f"Expected phi to have 1 dimension but got {len(self.model_phi.shape)}")
        if len(self.model_times.shape) != 1:
            raise ValueError(f"Expected model_times to have 1 dimension but got {len(self.model_times.shape)}")
        if self.tile_antennas:
            if self.model_gains.shape != (len(self.model_times), len(self.model_theta), len(self.model_freqs), 2, 2):
                raise ValueError(
                    f"gains shape {self.model_gains.shape} does not match shape "
                    f"(num_times, num_dir, num_freqs, 2, 2)."
                )
            self.model_gains = self.model_gains[:, :, None, :, :, :]  # [time, dir, 1, freq, 2, 2]

        else:
            if self.model_gains.shape != (
                    len(self.model_times), len(self.model_theta), len(self.antennas), len(self.model_freqs), 2, 2):
                raise ValueError(
                    f"gains shape {self.model_gains.shape} does not match shape "
                    f"(num_times, num_dir, num_ant, num_freqs, 2, 2)."
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
    def _compute_gain_jax(self, freqs: jax.Array, relative_time: jax.Array, lmn_sources: jax.Array):
        """
        Compute the beam gain at the given source coordinates.

        Args:
            freqs: (num_freqs) The frequencies at which to compute the beam gain.
            relative_time: Relative time in seconds from start.
            lmn_sources: (source_shape) + [num_ant/1, 3] The source coordinates in the L-M-N frame.

        Returns:
            (source_shape) + [num_ant, num_freq, 2, 2] The beam gain at the given source coordinates.
        """

        pointing_per_antenna = np.shape(lmn_sources)[-2] > 1
        if pointing_per_antenna:
            print("Assuming unique per-antennas pointings.")
        else:
            print("Assuming same pointing per antenna.")
        if self.tile_antennas:
            print("Assuming identical antenna beams.")
        else:
            print("Assuming unique per-antenna beams.")

        lmn_data = quantity_to_jnp(self.lmn_data)
        gains = jnp.asarray(
            quantity_to_jnp(self.model_gains),
            dtype=self.dtype
        )  # [num_model_time, num_model_dir,  A, num_model_freqs, 2, 2]
        model_freqs = quantity_to_jnp(self.model_freqs)
        relative_model_times = quantity_to_jnp((self.model_times - self.model_times[0]).sec * au.s)

        # Interpolate in time
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(relative_time, relative_model_times)
        gains = apply_interp(gains, i0, alpha0, i1, alpha1, axis=0)  # [num_model_dir,  A, num_model_freqs, 2, 2]

        # Interpolate in freq
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(freqs, model_freqs)
        gains = apply_interp(gains, i0, alpha0, i1, alpha1, axis=-3)  # [num_model_dir,  A, num_freqs, 2, 2]

        # Select closest direction
        shape = np.shape(lmn_sources)[:-2]
        lmn_sources = lmn_sources.reshape((-1,) + np.shape(lmn_sources)[-2:])  # [num_sources, num_ant/1, 3]

        def body_fn(carry, lmn_source):
            cos_dist = jnp.sum(lmn_source[:, None, :] * lmn_data, axis=-1)  # [num_ant/1, num_model_dir]
            closest = jnp.nanargmax(cos_dist, axis=-1)  # [num_ant/1]
            # cos_dist = jnp.where(jnp.isnan(cos_dist), -jnp.inf, cos_dist)
            # top_k_values, top_k_indices = jax.lax.top_k(cos_dist, k)
            return (), closest

        mem_usage_GB = (
                               lmn_sources.itemsize * np.shape(lmn_sources)[0] * np.shape(lmn_sources)[1] *
                               np.shape(lmn_data)[0]
                       ) >> 30

        # NB: 8GB is an arbitrary cutoff, but seems reasonable.
        if self.use_scan or mem_usage_GB > 8:
            def compute_closest(lmn_sources):
                _, closest = lax.scan(
                    body_fn,
                    (),
                    lmn_sources[:, None, :],
                    unroll=1
                )  # [num_sources, 1]
                return closest[:, 0]

            closest = jax.vmap(compute_closest, in_axes=1, out_axes=1)(lmn_sources)
        else:
            closest = jax.vmap(lambda lmn_source: body_fn(None, lmn_source)[1])(
                lmn_sources)  # [num_sources, num_ant/1]

        if pointing_per_antenna:
            # closest has shape # [num_sources, num_ant]
            if self.tile_antennas:
                # Gains have A=1
                gains = jax.vmap(lambda _closest: gains[_closest, 0, :, :, :], in_axes=1, out_axes=1)(
                    closest)  # [num_sources, num_ant, num_freqs, 2, 2]
            else:
                # Gains have A=num_ant
                gains = jax.vmap(lambda _closest, _gains: _gains[_closest], in_axes=1, out_axes=1)(
                    closest, gains)  # [num_sources, num_ant, num_freqs, 2, 2]
        else:
            # closest has shape # [num_sources, 1]
            if self.tile_antennas:
                # Gains have A=1
                gains = gains[closest[:, 0]]  # [num_sources, 1, num_freqs, 2, 2]
                gains = jnp.tile(gains, (1, len(self.antennas), 1, 1, 1))  # [num_sources, num_ant, num_freqs, 2, 2]
            else:
                # Gains have A=num_ant
                gains = gains[closest[:, 0]]  # [num_sources, num_ant, num_freqs, 2, 2]

        # Mask out evanescent sources
        evanescent_mask = jnp.isnan(lmn_sources[..., 2])  # [num_sources, num_ant/1]
        gains = jnp.where(
            evanescent_mask[:, :, None, None, None],
            jnp.nan,
            gains
        )  # [num_sources, num_ant, num_freqs, 2, 2]

        # Reshape to source shape
        gains = gains.reshape(shape + gains.shape[1:])  # (source_shape) + [num_ant, num_freq, 2, 2]
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
            pointing = zenith = ENU(east=0, north=0, up=1, location=self.antennas, obstime=time).transform_to(
                ac.ICRS())  # [num_ant]

        # Ensure pointing broadcasts with antennas
        pointing = pointing.reshape([1] * len(sources.shape) + [-1])  # [1,..., num_ant/1]

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
                                  phase_tracking=pointing)  # (source_shape) + [num_ant/1, 3]

        gains = self._compute_gain_jax(
            freqs=quantity_to_jnp(freqs),
            relative_time=quantity_to_jnp((time - self.model_times[0]).sec * au.s),
            lmn_sources=quantity_to_jnp(lmn_sources)
        )  # (source_shape) + [num_ant, num_freq, 2, 2]

        return gains


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
