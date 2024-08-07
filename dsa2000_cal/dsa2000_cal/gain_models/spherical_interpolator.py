import dataclasses
from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as au, time as at
from jax import numpy as jnp, lax

from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights, apply_interp, is_regular_grid
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np
from dsa2000_cal.gain_models.gain_model import GainModel


@partial(jax.jit, static_argnames=['resolution'])
def regrid_to_regular_grid(model_lmn: jax.Array, model_gains: jax.Array, resolution: int):
    """
    Regrid the input to a regular grid.

    Args:
        model_lmn: [num_model_dir, 3] The lmn coordinates.
        model_gains: [num_model_times, num_model_dir, [num_ant,] num_model_freqs[, 2, 2]] The gain model.
        resolution: The resolution of the regridded model.

    Returns:
        [num_model_times, resolution, resolution, [num_ant,] num_model_freqs[, 2, 2]] The regridded gains.
    """
    lvec = jnp.linspace(-1., 1., resolution)
    mvec = jnp.linspace(-1., 1., resolution)

    # Get the closest gains at each Theta, Phi
    if len(np.shape(model_gains)) == 6:
        gain_mapping = "[T,D,A,F,2,2]"
        out_mapping = "[T,lres,mres,A,F,2,2]"
    elif len(np.shape(model_gains)) == 5:
        gain_mapping = "[T,D,F,2,2]"
        out_mapping = "[T,lres,mres,F,2,2]"
    elif len(np.shape(model_gains)) == 4:
        gain_mapping = "[T,D,A,F]"
        out_mapping = "[T,lres,mres,A,F]"
    elif len(np.shape(model_gains)) == 3:
        gain_mapping = "[T,D,F]"
        out_mapping = "[T,lres,mres,F]"
    else:
        raise ValueError(f"Unsupported shape {np.shape(model_gains)}")

    @partial(
        multi_vmap,
        in_mapping=f"[lres],[mres],{gain_mapping}",
        out_mapping=out_mapping,
        scan_dims={'T'},
        verbose=True
    )
    def regrid_model_gains(l, m, model_gains):
        # Assume symmetric in n
        lm2 = (jnp.square(l) + jnp.square(m))
        n = jnp.sqrt(jnp.abs(1. - lm2))
        lmn = jnp.stack(
            [l, m, n]
        )  # [3]
        dist = 1. - jnp.sum(lmn * model_lmn, axis=-1)  # [num_model_dir]
        neg_dist_k, idx_k = lax.top_k(-dist, k=4)
        weights = 1. / (-neg_dist_k + 1e-3)
        value = jnp.sum(weights * model_gains[idx_k]) / jnp.sum(weights)
        # value = model_gains[jnp.argmin(dist, axis=-1)]
        horizon_decay = jnp.exp(-10. * n)
        return jnp.where(lm2 > 1., horizon_decay * value, value)

    gains = regrid_model_gains(lvec, mvec,
                               model_gains)  # [num_model_times, lres, mres, [num_ant,] num_model_freqs, [2,2]]

    return lvec, mvec, gains


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

        self.lmn_data = jnp.stack(
            lmn_from_phi_theta(
                phi=quantity_to_jnp(self.model_phi, 'rad'),
                theta=quantity_to_jnp(self.model_theta, 'rad')
            ),
            axis=-1
        )

        self.lvec_jax, self.mvec_jax, self.model_gains_jax = regrid_to_regular_grid(
            model_lmn=self.lmn_data,
            model_gains=quantity_to_jnp(self.model_gains),
            resolution=128
        )  # [num_model_times, lres, mres, [num_ant,] num_model_freqs, [2,2]]

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

        relative_model_times = quantity_to_jnp((self.model_times.tt - self.model_times[0].tt).sec * au.s)

        if self.tile_antennas:
            if self.is_full_stokes():
                gain_mapping = "[T,lres,mres,F,2,2]"
            else:
                gain_mapping = "[T,lres,mres,F]"
        else:
            if self.is_full_stokes():
                gain_mapping = "[T,lres,mres,a,F,2,2]"
            else:
                gain_mapping = "[T,lres,mres,a,F]"

        @partial(
            multi_vmap,
            in_mapping=f"[t],[s,t,a],[s,t,a],[f],{gain_mapping}",
            out_mapping="[s,t,a,f,...]",
            verbose=True
        )
        def interp_model_gains(time, l, m, freq, gains):
            """
            Compute the gain for the given time, theta, phi, freq.

            Args:
                time: the time in seconds since start of obs.
                l: the geodesic l component.
                m: the godesic m component.
                freq: the frequency in Hz.
                gains: [T,lres,mres,F[,2,2]] the model gains.

            Returns:
                [[2, 2]] The gain for the given time, theta, phi, freq.
            """
            # Get time
            (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(
                x=time,
                xp=relative_model_times,
                regular_grid=is_regular_grid(quantity_to_np((self.model_times - self.model_times[0]).sec * au.s))
            )
            # jax.debug.print("time: {i0} {alpha0} {i1} {alpha1}", i0=i0, alpha0=alpha0, i1=i1, alpha1=alpha1)
            gains = apply_interp(gains, i0, alpha0, i1, alpha1,
                                 axis=0)  # [lres, mres, F[, 2, 2]]
            # get l
            (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(
                x=l,
                xp=self.lvec_jax,
                regular_grid=True
            )
            # jax.debug.print("l: {i0} {alpha0} {i1} {alpha1}", i0=i0, alpha0=alpha0, i1=i1, alpha1=alpha1)
            gains = apply_interp(gains, i0, alpha0, i1, alpha1,
                                 axis=0)  # [mres, F[, 2, 2]]
            # get m
            (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(
                x=m,
                xp=self.mvec_jax,
                regular_grid=True
            )
            # jax.debug.print("m: {i0} {alpha0} {i1} {alpha1}", i0=i0, alpha0=alpha0, i1=i1, alpha1=alpha1)
            gains = apply_interp(gains, i0, alpha0, i1, alpha1,
                                 axis=0)  # [F[, 2, 2]]
            # get freq
            (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(
                x=freq,
                xp=quantity_to_jnp(self.model_freqs),
                regular_grid=is_regular_grid(quantity_to_np(self.model_freqs))
            )
            # jax.debug.print("freq: {i0} {alpha0} {i1} {alpha1}", i0=i0, alpha0=alpha0, i1=i1, alpha1=alpha1)
            gains = apply_interp(gains, i0, alpha0, i1, alpha1,
                                 axis=0)  # [[2, 2]]
            return gains

        gains = interp_model_gains(times, lmn_geodesic[..., 0],
                                   lmn_geodesic[..., 1], freqs,
                                   self.model_gains_jax)  # [num_sources, num_times, num_ant, num_freq[, 2, 2]]

        return gains

    def compute_gain(self, freqs: jax.Array, times: jax.Array, geodesics: jax.Array) -> jax.Array:
        gains = self._compute_gain_jax(
            freqs=freqs,
            times=times,
            lmn_geodesic=geodesics
        )  # [num_sources, num_times, num_ant, num_freq[, 2, 2]]
        return gains

    def plot_beam(self, save_fig: str | None = None, time_idx: int = 0, ant_idx: int = 0, freq_idx: int = 0,
                  p_idx: int = 0, q_idx: int = 0):
        """
        Plot the beam gain model screen over plane of sky wrt antenna pointing.

        Args:
            save_fig: the path to save the figure to.
            time_idx: the time index to plot.
            ant_idx: the antenna index to plot.
            freq_idx: the frequency index to plot.
            p_idx: the p index to plot.
            q_idx: the q index to plot.
        """
        if self.tile_antennas:
            if self.is_full_stokes():
                gain_screen = self.model_gains_jax[time_idx, :, :, freq_idx, p_idx, q_idx]  # [nl,nm]
            else:
                gain_screen = self.model_gains_jax[time_idx, :, :, freq_idx]  # [nl,nm]
        else:
            if self.is_full_stokes():
                gain_screen = self.model_gains_jax[time_idx, :, :, ant_idx, freq_idx, p_idx, q_idx]  # [nl,nm]
            else:
                gain_screen = self.model_gains_jax[time_idx, :, :, ant_idx, freq_idx]  # [nl,nm]
        l_screen, m_screen = np.meshgrid(self.lvec_jax, self.mvec_jax, indexing='ij')
        fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True, sharey=True, squeeze=False)
        # Plot log10(amp)
        sc = axs[0, 0].imshow(
            np.log10(np.abs(gain_screen.T)),
            extent=(self.lvec_jax[0], self.lvec_jax[-1], self.mvec_jax[0], self.mvec_jax[-1]),
            origin='lower',
            cmap='PuOr',
            interpolation='none'
        )
        fig.colorbar(sc, ax=axs[0, 0])
        axs[0, 0].set_ylabel('m (proj. rad.)')
        axs[0, 0].set_title('log10(Amplitude)')
        # Plot phase
        sc = axs[1, 0].imshow(
            np.angle(gain_screen.T),
            extent=(self.lvec_jax[0], self.lvec_jax[-1], self.mvec_jax[0], self.mvec_jax[-1]),
            origin='lower',
            cmap='hsv',
            vmin=-np.pi,
            vmax=np.pi,
            interpolation='none'
        )
        fig.colorbar(sc, ax=axs[1, 0])
        axs[1, 0].set_xlabel('l (proj. rad.)')
        axs[1, 0].set_ylabel('m (proj. rad.)')
        axs[1, 0].set_title('Phase')

        fig.tight_layout()

        if save_fig is not None:
            plt.savefig(save_fig)

        plt.show()


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
