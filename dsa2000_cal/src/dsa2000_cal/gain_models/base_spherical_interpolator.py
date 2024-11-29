import dataclasses
import pickle
import warnings
from functools import partial
from typing import Tuple, List, Any

import jax
import numpy as np
from astropy import coordinates as ac, units as au, time as at
from jax import numpy as jnp
from matplotlib import pyplot as plt

from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights, apply_interp
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.nearest_neighbours import kd_tree_nn
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.gain_model import GainModel


@dataclasses.dataclass(eq=False)
class BaseSphericalInterpolatorGainModel(GainModel):
    """
    A base class for spherical interpolator gain models.

    Args:
        model_freqs: [num_model_freqs] The frequencies at which the model is defined. Must be regular grid.
        model_times: [num_model_times] The times at which the model is defined. Must be regular grid.
        lvec: [lres] The l values of the model. Must be regular grid.
        mvec: [mres] The m values of the model. Must be regular grid.
        model_gains: [num_model_times, lres, mres, [num_ant,] num_model_freqs[, 2, 2]] The model gains.
        tile_antennas: Whether to tile the antennas.
        full_stokes: Whether to use full stokes.
    """
    model_freqs: jax.Array  # [num_model_freqs]
    model_times: jax.Array  # [num_model_times]
    lvec: jax.Array  # [lres]
    mvec: jax.Array  # [mres]
    model_gains: jax.Array  # [num_model_times, lres, mres, [num_ant,] num_model_freqs[, 2, 2]]
    tile_antennas: bool
    full_stokes: bool
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return
        num_model_freqs = np.shape(self.model_freqs)[0]
        num_model_times = np.shape(self.model_times)[0]
        lres = np.shape(self.lvec)[0]
        mres = np.shape(self.mvec)[0]
        if lres != mres:
            raise ValueError("Expected lvec and mvec to have the same length. We like squares.")
        if self.tile_antennas:
            if self.full_stokes:
                assert np.shape(self.model_gains) == (num_model_times, lres, mres, num_model_freqs, 2, 2)
            else:
                assert np.shape(self.model_gains) == (num_model_times, lres, mres, num_model_freqs)
        else:
            num_ant = np.shape(self.model_gains)[3]
            if self.full_stokes:
                assert np.shape(self.model_gains) == (num_model_times, lres, mres, num_ant, num_model_freqs, 2, 2)
            else:
                assert np.shape(self.model_gains) == (num_model_times, lres, mres, num_ant, num_model_freqs)
        self.model_freqs = mp_policy.cast_to_freq(self.model_freqs)
        self.model_times = mp_policy.cast_to_time(self.model_times)
        self.lvec = mp_policy.cast_to_angle(self.lvec)
        self.mvec = mp_policy.cast_to_angle(self.mvec)
        self.model_gains = mp_policy.cast_to_gain(self.model_gains)

    def is_full_stokes(self) -> bool:
        return self.full_stokes

    def compute_gain(self, freqs: jax.Array, times: jax.Array, lmn_geodesic: jax.Array):

        if self.tile_antennas:
            if self.full_stokes:
                gain_mapping = "[T,lres,mres,F,2,2]"
            else:
                gain_mapping = "[T,lres,mres,F]"
        else:
            if self.full_stokes:
                gain_mapping = "[T,lres,mres,a,F,2,2]"
            else:
                gain_mapping = "[T,lres,mres,a,F]"

        @partial(
            multi_vmap,
            in_mapping=f"[t],[t,a,s],[t,a,s],[f],{gain_mapping}",
            out_mapping="[t,a,f,s,...]",
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

            # interpolating along last is always better, for memory contiguity
            # ==> order: F, mres, lres, T

            # freq
            (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(
                x=freq,
                xp=self.model_freqs,
                regular_grid=True
            )
            # jax.debug.print("freq: {i0} {alpha0} {i1} {alpha1}", i0=i0, alpha0=alpha0, i1=i1, alpha1=alpha1)
            gains = apply_interp(gains, i0, alpha0, i1, alpha1, axis=3)  # [T, lres, mres, [2, 2]]

            # mres
            (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(
                x=m,
                xp=self.mvec,
                regular_grid=True
            )
            # jax.debug.print("m: {i0} {alpha0} {i1} {alpha1}", i0=i0, alpha0=alpha0, i1=i1, alpha1=alpha1)
            gains = apply_interp(gains, i0, alpha0, i1, alpha1, axis=2)  # [T, lres, [2, 2]]

            # lres
            (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(
                x=l,
                xp=self.lvec,
                regular_grid=True
            )
            # jax.debug.print("l: {i0} {alpha0} {i1} {alpha1}", i0=i0, alpha0=alpha0, i1=i1, alpha1=alpha1)
            gains = apply_interp(gains, i0, alpha0, i1, alpha1, axis=1)  # [T, [, 2, 2]]

            # time
            if len(self.model_times) == 1:
                gains = gains[0]  # [[, 2, 2]]
            else:
                # Get time
                (i0, alpha0, i1, alpha1) = get_interp_indices_and_weights(
                    x=time,
                    xp=self.model_times,
                    regular_grid=True
                )
                # jax.debug.print("time: {i0} {alpha0} {i1} {alpha1}", i0=i0, alpha0=alpha0, i1=i1, alpha1=alpha1)
                gains = apply_interp(gains, i0, alpha0, i1, alpha1, axis=0)  # [[, 2, 2]]

            return mp_policy.cast_to_gain(gains)
        print(self)
        print(times,
            lmn_geodesic[..., 0],
            lmn_geodesic[..., 1],
            freqs,
            self.model_gains)

        gains = interp_model_gains(
            times,
            lmn_geodesic[..., 0],
            lmn_geodesic[..., 1],
            freqs,
            self.model_gains
        )  # [num_times, num_ant, num_freq, num_sources, [, 2, 2]]

        return gains

    def plot_regridded_beam(self, save_fig: str | None = None, time_idx: int = 0, ant_idx: int = 0, freq_idx: int = 0,
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
                gain_screen = self.model_gains[time_idx, :, :, freq_idx, p_idx, q_idx]  # [nl,nm]
            else:
                gain_screen = self.model_gains[time_idx, :, :, freq_idx]  # [nl,nm]
        else:
            if self.is_full_stokes():
                gain_screen = self.model_gains[time_idx, :, :, ant_idx, freq_idx, p_idx, q_idx]  # [nl,nm]
            else:
                gain_screen = self.model_gains[time_idx, :, :, ant_idx, freq_idx]  # [nl,nm]
        fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True, sharey=True, squeeze=False)
        # Plot log10(amp)
        sc = axs[0, 0].imshow(
            np.log10(np.abs(gain_screen.T)),
            extent=(self.lvec[0], self.lvec[-1],
                    self.mvec[0], self.mvec[-1]),
            origin='lower',
            cmap='PuOr',
            interpolation='none'
        )
        fig.colorbar(sc, ax=axs[0, 0])
        axs[0, 0].set_ylabel('m (proj. rad.)')
        axs[0, 0].set_title(f'Gridded log10(Amplitude)[T={time_idx},A={ant_idx},F={freq_idx},P={p_idx},Q={q_idx}]')
        # Plot phase
        sc = axs[1, 0].imshow(
            np.angle(gain_screen.T),
            extent=(self.lvec[0], self.lvec[-1],
                    self.mvec[0], self.mvec[-1]),
            origin='lower',
            cmap='hsv',
            vmin=-np.pi,
            vmax=np.pi,
            interpolation='none'
        )
        fig.colorbar(sc, ax=axs[1, 0])
        axs[1, 0].set_xlabel('l (proj. rad.)')
        axs[1, 0].set_ylabel('m (proj. rad.)')
        axs[1, 0].set_title(f'Gridded Phase[T={time_idx},A={ant_idx},F={freq_idx},P={p_idx},Q={q_idx}]')

        fig.tight_layout()

        if save_fig is not None:
            plt.savefig(save_fig)
            plt.close(fig)
        else:
            plt.show()

    def save(self, filename: str):
        """
        Serialise the model to file.

        Args:
            filename: the filename
        """
        if not filename.endswith('.pkl'):
            warnings.warn(f"Filename {filename} does not end with .pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        """
        Load the model from file.

        Args:
            filename: the filename

        Returns:
            the model
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        children, aux_data = self.flatten(self)
        children_np = jax.tree.map(np.asarray, children)
        serialised = (aux_data, children_np)
        return (self._deserialise, (serialised,))

    @classmethod
    def _deserialise(cls, serialised):
        # Create a new instance, bypassing __init__ and setting the actor directly
        (aux_data, children_np) = serialised
        children_jax = jax.tree.map(jnp.asarray, children_np)
        return cls.unflatten(aux_data, children_jax)

    @classmethod
    def register_pytree(cls):
        jax.tree_util.register_pytree_node(cls, cls.flatten, cls.unflatten)

    # an abstract classmethod
    @classmethod
    def flatten(cls, this: "BaseSphericalInterpolatorGainModel") -> Tuple[List[Any], Tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        return (
            [
                this.model_freqs,
                this.model_times,
                this.lvec,
                this.mvec,
                this.model_gains
            ], (
                this.tile_antennas,
                this.full_stokes
            )
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "BaseSphericalInterpolatorGainModel":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        model_freqs, model_times, lvec, mvec, model_gains = children
        tile_antennas, full_stokes = aux_data
        return BaseSphericalInterpolatorGainModel(
            model_freqs, model_times, lvec, mvec, model_gains, tile_antennas, full_stokes,
            skip_post_init=True
        )

BaseSphericalInterpolatorGainModel.register_pytree()


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
    if resolution % 2 == 0:
        raise ValueError("Resolution must be odd so that central pixel falls on l=0,m=0.")

    lvec = jnp.linspace(-1., 1., resolution)
    mvec = jnp.linspace(-1., 1., resolution)
    L, M = jnp.meshgrid(lvec, mvec, indexing='ij')
    LM2 = L ** 2 - M ** 2
    N = jnp.sqrt(jnp.abs(1. - L ** 2 - M ** 2))
    N = jnp.where(LM2 > 1., -N, N)
    lmn_grid = jnp.stack([L.flatten(), M.flatten(), N.flatten()], axis=-1)

    dist, idx = kd_tree_nn(model_lmn, lmn_grid, k=6)

    _, idx0 = kd_tree_nn(model_lmn, jnp.asarray([[0., 0., 1.]]), k=1)  # [1,1]
    idx0 = idx0[0, 0]

    # Get the closest gains at each Theta, Phi
    if len(np.shape(model_gains)) == 6:
        gain_mapping = "[T,D,A,F,p,q]"
        out_mapping = "[T,R,A,F,p,q]"
    elif len(np.shape(model_gains)) == 5:
        gain_mapping = "[T,D,F,p,q]"
        out_mapping = "[T,R,F,p,q]"
    elif len(np.shape(model_gains)) == 4:
        gain_mapping = "[T,D,A,F]"
        out_mapping = "[T,R,A,F]"
    elif len(np.shape(model_gains)) == 3:
        gain_mapping = "[T,D,F]"
        out_mapping = "[T,R,F]"
    else:
        raise ValueError(f"Unsupported shape {np.shape(model_gains)}")

    @partial(
        multi_vmap,
        in_mapping=f"[R],[R],[R,k],[R,k],{gain_mapping}",
        out_mapping=out_mapping,
        scan_dims={'T'},
        verbose=True
    )
    def regrid_model_gains(l_eval, m_eval, idx_k, dist_k, model_gains):
        weights = 1. / (dist_k + 1e-6) ** 0.5
        # At peak the interpolation smooths out peak too much
        near_peak = (l_eval == 0.) & (m_eval == 0.)
        value = jnp.sum(weights * model_gains[idx_k]) / jnp.sum(weights)
        value = jnp.where(near_peak, model_gains[idx0], value)
        neg_n = l_eval ** 2 + m_eval ** 2 > 1.
        value = jnp.where(neg_n, jnp.zeros_like(value), value)
        return value

    gains = regrid_model_gains(
        lmn_grid[:, 0], lmn_grid[:, 1],
        idx, dist, model_gains
    )  # [num_model_times, lres*mres, [num_ant,] num_model_freqs, [2,2]]
    gains = jnp.reshape(gains, (model_gains.shape[0], resolution, resolution, *model_gains.shape[2:]))

    return mp_policy.cast_to_angle(lvec), mp_policy.cast_to_angle(mvec), mp_policy.cast_to_gain(gains)


def build_spherical_interpolator(
        antennas: ac.EarthLocation,  # [num_ant]
        model_freqs: au.Quantity,  # [num_model_freqs]
        model_theta: au.Quantity,  # [num_model_dir] # Theta is in [0, 180] measured from bore-sight
        model_phi: au.Quantity,  # [num_model_dir] # Phi is in [0, 360] measured from x-axis
        model_times: at.Time,  # [num_model_times] # Times at which the model is defined
        model_gains: au.Quantity,  # [num_model_times, num_model_dir, [num_ant,] num_model_freqs, 2, 2]
        ref_time: at.Time,
        tile_antennas: bool,
        resolution: int = 257
):
    """
    Uses nearest neighbour interpolation to construct the gain model.

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
    """

    # make sure all 1D
    if model_freqs.isscalar:
        raise ValueError("Expected model_freqs to be an array.")
    if model_theta.isscalar:
        raise ValueError("Expected theta to be an array.")
    if model_phi.isscalar:
        raise ValueError("Expected phi to be an array.")
    if model_times.isscalar:
        raise ValueError("Expected model_times to be an array.")

    # Check shapes
    if len(model_freqs.shape) != 1:
        raise ValueError(f"Expected model_freqs to have 1 dimension but got {len(model_freqs.shape)}")
    if len(model_theta.shape) != 1:
        raise ValueError(f"Expected theta to have 1 dimension but got {len(model_theta.shape)}")
    if len(model_phi.shape) != 1:
        raise ValueError(f"Expected phi to have 1 dimension but got {len(model_phi.shape)}")
    if len(model_times.shape) != 1:
        raise ValueError(f"Expected model_times to have 1 dimension but got {len(model_times.shape)}")
    if tile_antennas:
        if model_gains.shape[:3] != (len(model_times), len(model_theta), len(model_freqs)):
            raise ValueError(
                f"gains shape {model_gains.shape} does not match shape "
                f"(num_times, num_dir, num_freqs[, 2, 2])."
            )

    else:
        if model_gains.shape[:4] != (
                len(model_times), len(model_theta), len(antennas), len(model_freqs)):
            raise ValueError(
                f"gains shape {model_gains.shape} does not match shape "
                f"(num_times, num_dir, num_ant, num_freqs[, 2, 2])."
            )

    # Ensure phi,theta,freq units congrutent
    if not model_theta.unit.is_equivalent(au.deg):
        raise ValueError(f"Expected theta to be in degrees but got {model_theta.unit}")
    if not model_phi.unit.is_equivalent(au.deg):
        raise ValueError(f"Expected phi to be in degrees but got {model_phi.unit}")
    if not model_freqs.unit.is_equivalent(au.Hz):
        raise ValueError(f"Expected model_freqs to be in Hz but got {model_freqs.unit}")
    if not model_gains.unit.is_equivalent(au.dimensionless_unscaled):
        raise ValueError(f"Expected model_gains to be dimensionless but got {model_gains.unit}")

    lmn_data = mp_policy.cast_to_angle(
        jnp.stack(
            lmn_from_phi_theta(
                phi=quantity_to_jnp(model_phi, 'rad'),
                theta=quantity_to_jnp(model_theta, 'rad')
            ),
            axis=-1
        )
    )  # [num_model_dir, 3]

    lvec_jax, mvec_jax, model_gains_jax = regrid_to_regular_grid(
        model_lmn=lmn_data,
        model_gains=quantity_to_jnp(model_gains),
        resolution=resolution
    )  # [num_model_times, lres, mres, [num_ant,] num_model_freqs, [2,2]]

    if tile_antennas:
        print("Assuming identical antenna beams.")
    else:
        print("Assuming unique per-antenna beams.")

    base_spherical_interpolator = BaseSphericalInterpolatorGainModel(
        model_freqs=quantity_to_jnp(model_freqs),
        model_times=quantity_to_jnp((model_times.tt - ref_time.tt).sec * au.s),
        lvec=lvec_jax,
        mvec=mvec_jax,
        model_gains=model_gains_jax,
        tile_antennas=tile_antennas,
        full_stokes=model_gains.shape[-2:] == (2, 2)
    )
    return base_spherical_interpolator


def plot_beam(
        model_theta: au.Quantity,  # [num_model_dir] # Theta is in [0, 180] measured from bore-sight
        model_phi: au.Quantity,  # [num_model_dir] # Phi is in [0, 360] measured from x-axis
        model_gains: au.Quantity,  # [num_model_times, num_model_dir, [num_ant,] num_model_freqs, 2, 2]
        tile_antennas: bool,
        full_stokes: bool,
        save_fig: str | None = None,
        time_idx: int = 0,
        ant_idx: int = 0,
        freq_idx: int = 0,
        p_idx: int = 0,
        q_idx: int = 0):
    """
    Plot the beam gain model screen over plane of sky wrt antenna pointing.

    Args:
        model_theta: [num_model_dir] Theta is in [0, 180] measured from bore-sight
        model_phi: [num_model_dir] Phi is in [0, 360] measured from x-axis
        model_gains: [num_model_times, num_model_dir, [num_ant,] num_model_freqs, 2, 2]
        tile_antennas: Whether the model gains are tiled per antenna.
        full_stokes: Whether the model gains are full stokes.
        save_fig: if not None, the path to save the figure to.
        time_idx: the time index to plot.
        ant_idx: the antenna index to plot.
        freq_idx: the frequency index to plot.
        p_idx: the p index to plot.
        q_idx: the q index to plot.
    """
    # Like plot_regridded_beam, but plots with scatter using original data
    # self.model_gains are [num_model_times, num_model_dir, [num_ant,] num_model_freqs, 2, 2]
    if tile_antennas:
        if full_stokes:
            gains_data = model_gains[time_idx, :, freq_idx, p_idx, q_idx]  # [num_model_dir]
        else:
            gains_data = model_gains[time_idx, :, freq_idx]
    else:
        if full_stokes:
            gains_data = model_gains[time_idx, :, ant_idx, freq_idx, p_idx, q_idx]
        else:
            gains_data = model_gains[time_idx, :, ant_idx, freq_idx]
    l, m, n = lmn_from_phi_theta(model_phi.to('rad').value, model_theta.to('rad').value)
    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True, sharey=True, squeeze=False)
    # Plot log10(amp)
    sc = axs[0, 0].scatter(
        l, m, c=np.log10(np.abs(gains_data)),
        cmap='PuOr', s=1
    )
    fig.colorbar(sc, ax=axs[0, 0])
    axs[0, 0].set_xlabel('l (proj. rad.)')
    axs[0, 0].set_ylabel('m (proj. rad.)')
    axs[0, 0].set_title(f'log10(Amplitude)[T={time_idx},A={ant_idx},F={freq_idx},P={p_idx},Q={q_idx}]')
    # Plot phase
    sc = axs[1, 0].scatter(
        l, m, c=np.angle(gains_data),
        cmap='hsv', vmin=-np.pi, vmax=np.pi, s=1
    )
    fig.colorbar(sc, ax=axs[1, 0])
    axs[1, 0].set_xlabel('l (proj. rad.)')
    axs[1, 0].set_ylabel('m (proj. rad.)')
    axs[1, 0].set_title(f'Phase[T={time_idx},A={ant_idx},F={freq_idx},P={p_idx},Q={q_idx}]')

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
    return mp_policy.cast_to_angle((l, m, n))


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

    return mp_policy.cast_to_angle((wrap(phi), theta))
