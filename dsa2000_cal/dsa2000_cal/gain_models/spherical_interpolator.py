import dataclasses
from functools import partial

import jax
import numpy as np
from astropy import units as au, time as at
from jax import numpy as jnp

from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights, apply_interp, is_regular_grid
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np
from dsa2000_cal.gain_models.gain_model import GainModel


@dataclasses.dataclass(eq=False)
class SphericalInterpolatorGainModel(GainModel):
    """
    Uses nearest neighbour interpolation to compute the gain model.

    The antennas have attenuation models in frame of antenna, call this the X-Y frame (see below).
    X points up, Y points to the right, Z points towards the source (along bore).

    Same as standing facing South, and looking up at the sky.
    Theta measures the angle from the bore-sight, and phi measures West from South, so that phi=0 is South=M,
    phi=90 is West=-L, phi=-90 is East=L.

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
            if self.model_gains.shape[:3] != (len(self.model_times), len(self.model_theta), len(self.model_freqs)):
                raise ValueError(
                    f"gains shape {self.model_gains.shape} does not match shape "
                    f"(num_times, num_dir, num_freqs[, 2, 2])."
                )

        else:
            if self.model_gains.shape[:4] != (
                    len(self.model_times), len(self.model_theta), len(self.antennas), len(self.model_freqs)):
                raise ValueError(
                    f"gains shape {self.model_gains.shape} does not match shape "
                    f"(num_times, num_dir, num_ant, num_freqs[, 2, 2])."
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

        if self.tile_antennas:
            print("Assuming identical antenna beams.")
        else:
            print("Assuming unique per-antenna beams.")

    def is_full_stokes(self) -> bool:
        return self.model_gains.shape[-2:] == (2, 2)

    def _compute_gain_jax(self, freqs: jax.Array, times: jax.Array, lmn_geodesic: jax.Array):
        """
        Compute the beam gain at the given source coordinates.

        Args:
            freqs: (num_freqs) The frequencies at which to compute the beam gain.
            times: [num_times] Relative time in seconds from start, TT scale.
            lmn_geodesic: [num_sources, num_times, num_ant, 3] lmn coords in antenna frame.

        Returns:
            [num_sources, num_times, num_ant, num_freq[, 2, 2]] The beam gain at the given source coordinates.
        """

        if len(np.shape(lmn_geodesic)) == 4:
            geodesic_mapping = "[s,t,a,3]"
        elif len(np.shape(lmn_geodesic)) == 3:
            geodesic_mapping = "[s,t,3]"
        else:
            raise ValueError(f"Expected geodesic to have shape (num_sources, num_times, [num_ant,] num_freqs, 3) "
                             f"but got {np.shape(lmn_geodesic)}")

        lmn_data = quantity_to_jnp(self.lmn_data)  # [num_model_dir, 3]
        gains = jnp.asarray(
            quantity_to_jnp(self.model_gains),
            dtype=self.dtype
        )  # [num_model_time, num_model_dir,  [num_ants,] num_model_freqs[, 2, 2]]

        # Interpolate in time
        relative_model_times = quantity_to_jnp((self.model_times - self.model_times[0]).sec * au.s)
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(
            x=times,
            xp=relative_model_times,
            regular_grid=is_regular_grid(quantity_to_np((self.model_times - self.model_times[0]).sec * au.s))
        )
        gains = apply_interp(gains, i0, alpha0, i1, alpha1,
                             axis=0)  # [num_times, num_model_dir,  [num_ant,] num_model_freqs[, 2, 2]]

        # Interpolate in freq
        model_freqs = quantity_to_jnp(self.model_freqs)
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(
            x=freqs,
            xp=model_freqs,
            regular_grid=is_regular_grid(quantity_to_np(self.model_freqs))
        )
        if self.tile_antennas:
            gains = apply_interp(gains, i0, alpha0, i1, alpha1,
                                 axis=2)  # [num_times, num_model_dir,  num_freqs[, 2, 2]]
            if self.is_full_stokes():
                gains_mapping = "[t,S,f,2,2]"
            else:
                gains_mapping = "[t,S,f]"
        else:
            gains = apply_interp(gains, i0, alpha0, i1, alpha1,
                                 axis=3)  # [num_times, num_model_dir, num_ant, num_freqs[, 2, 2]]
            if self.is_full_stokes():
                gains_mapping = "[t,S,a,f,2,2]"
            else:
                gains_mapping = "[t,S,a,f]"

        # Select closest direction
        mem_usage_GB = (
                               lmn_geodesic.itemsize * np.shape(lmn_geodesic)[0] * np.shape(lmn_geodesic)[1] *
                               np.shape(lmn_data)[0]
                       ) >> 30
        use_scan = mem_usage_GB > 8

        @partial(
            multi_vmap,
            in_mapping=f"[s,t,a,3],{gains_mapping}",
            out_mapping="[s,t,a,f,...]",
            scan_dims={"s"} if use_scan else None
        )
        def interp_source(lmn_geodesic, gains):
            """
            Compute the closest direction to the given lmn coordinates.

            Args:
                lmn_geodesic: [3]
                gains: [num_model_dir[, 2, 2]]

            Returns:
                [[2, 2]] The gain for the closest direction.
            """
            cos_dist = jnp.sum(lmn_geodesic * lmn_data, axis=-1)  # [num_model_dir]
            closest = jnp.nanargmax(cos_dist, axis=-1)  # []
            # cos_dist = jnp.where(jnp.isnan(cos_dist), -jnp.inf, cos_dist)
            # top_k_values, top_k_indices = jax.lax.top_k(cos_dist, k)
            evanescent_mask = jnp.isnan(lmn_geodesic[2])
            return jnp.where(evanescent_mask, jnp.nan, gains[closest])  # [[2, 2]]

        gains = interp_source(lmn_geodesic, gains)  # [num_sources, num_times, num_ant, num_freq[, 2, 2]]

        return gains

    def compute_gain(self, freqs: jax.Array, times: jax.Array, geodesics: jax.Array) -> jax.Array:
        gains = self._compute_gain_jax(
            freqs=freqs,
            times=times,
            lmn_geodesic=geodesics
        )  # [num_sources, num_times, num_ant, num_freq[, 2, 2]]
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


def phi_theta_from_lmn(l, m, n):
    """
    Convert cartesian coordinates to phi, theta.
    Right-handed X-Y-Z(bore), with phi azimuthal measured from X, and theta from Z

    L-M frame is the same as (-Y)-X frame, i.e. Y is -L and X is M.

    Args:
        l: L component
        m: M component
        n: N component

    Returns:
        phi: azimuthal angle in [0, 2pi]
        theta: polar angle in [0, pi]
    """
    phi = jnp.arctan2(-l, m)
    theta = jnp.arccos(n)

    def wrap(angle):
        return (angle + 2 * np.pi) % (2 * np.pi)

    return wrap(phi), theta
