import dataclasses
import itertools
import warnings
from functools import partial

import astropy.constants as const
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights, apply_interp
from dsa2000_cal.common.quantity_utils import quantity_to_jnp


# from astropy.coordinates import solar_system_ephemeris
# solar_system_ephemeris.set('jpl')

@dataclasses.dataclass(eq=False)
class InterpolatedArray:
    times: jax.Array  # [N]
    values: jax.Array  # [..., N, ...] `axis` has N elements

    axis: int = 0
    regular_grid: bool = False

    def __post_init__(self):

        if len(np.shape(self.times)) != 1:
            raise ValueError(f"Times must be 1D, got {np.shape(self.times)}.")

        def _assert_shape(x):
            if np.shape(x)[self.axis] != np.size(self.times):
                raise ValueError(f"Input values must have time length on `axis` dimension, got {np.shape(x)}.")

        jax.tree.map(_assert_shape, self.values)

        self.times, self.values = jax.tree.map(jnp.asarray, (self.times, self.values))

    @property
    def shape(self):
        return jax.tree.map(lambda x: np.shape(x)[:self.axis] + np.shape(x)[self.axis + 1:], self.values)

    def __call__(self, time: jax.Array) -> jax.Array:
        """
        Interpolate at time based on input times.

        Args:
            time: time to evaluate at.

        Returns:
            value at given time
        """
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(time, self.times, regular_grid=self.regular_grid)
        return jax.tree.map(lambda x: apply_interp(x, i0, alpha0, i1, alpha1, axis=self.axis), self.values)


def norm(x, axis=-1, keepdims: bool = False):
    return jnp.sqrt(norm2(x, axis, keepdims))


def norm2(x, axis=-1, keepdims: bool = False):
    return jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims)


def perley_icrs_from_lmn(l, m, n, ra0, dec0):
    dec = jnp.arcsin(m * jnp.cos(dec0) + n * jnp.sin(dec0))
    ra = ra0 + jnp.arctan2(l, n * jnp.cos(dec0) - m * jnp.sin(dec0))
    return ra, dec


def perley_lmn_from_icrs(alpha, dec, alpha0, dec0):
    dra = alpha - alpha0

    l = jnp.cos(dec) * jnp.sin(dra)
    m = jnp.sin(dec) * jnp.cos(dec0) - jnp.cos(dec) * jnp.sin(dec0) * jnp.cos(dra)
    n = jnp.sin(dec) * jnp.sin(dec0) + jnp.cos(dec) * jnp.cos(dec0) * jnp.cos(dra)
    return l, m, n


def celestial_to_cartesian(ra, dec):
    x = jnp.cos(ra) * jnp.cos(dec)
    y = jnp.sin(ra) * jnp.cos(dec)
    z = jnp.sin(dec)
    return jnp.stack([x, y, z], axis=-1)


def compute_uvw(antennas: ac.EarthLocation, times: at.Time, phase_center: ac.ICRS,
                resolution: au.Quantity | None = None, with_autocorr: bool = True, verbose: bool = False):
    """
    Compute the UVW coordinates for a given phase center, using VLBI delay model.

    Args:
        antennas: [num_ant] EarthLocation of antennas.
        times: [num_time] Time of observation.
        phase_center: the phase center of the observation.
        resolution: temporal resolution for interpolation, default of 120 seconds.
        verbose: whether to print information out.

    Returns:
        uvw: [num_time, num_baselines, 3] UVW coordinates.
    """
    if not config.jax_enable_x64:
        warnings.warn("jax_enable_x64 is not set, UVW computations may be inaccurate.")

    bodies_except_earth = (
        'sun', 'moon', 'mercury', 'venus',
        'mars', 'jupiter', 'saturn', 'uranus',
        'neptune'
    )
    GM_bodies = {
        'sun': const.GM_sun,
        'moon': 0.0123 * const.GM_earth,
        'mercury': 0.0553 * const.GM_earth,
        'venus': 0.815 * const.GM_earth,
        'mars': 0.107 * const.GM_earth,
        'jupiter': const.GM_jup,
        'saturn': 95.16 * const.GM_earth,
        'uranus': 14.54 * const.GM_earth,
        'neptune': 17.15 * const.GM_earth
    }
    ra0 = jnp.asarray(phase_center.ra.rad)
    dec0 = jnp.asarray(phase_center.dec.rad)

    if resolution is None:
        resolution = 120 * au.s

    if not resolution.unit.is_equivalent(au.s):
        raise ValueError(f"resolution must be in seconds got {resolution.unit}")

    if len(antennas.shape) != 1:
        raise ValueError(f"antennas must be 1D got {antennas.shape}")

    if antennas.shape[0] < 2:
        raise ValueError(f"Need at least 2 antennas to form a baseline.")

    if len(times.shape) != 1:
        raise ValueError(f"times must be 1D got {times.shape}")

    times = times.tt
    start_time = times.min()

    num_grid_times = int(np.ceil((times.max() - times.min()) / resolution)) + 1
    num_ants = len(antennas)

    # Define the interpolation grid
    interp_times = start_time.tt + np.arange(num_grid_times) * resolution  # [T]

    # Define the antennas
    antennas_gcrs = antennas.reshape((1, num_ants)).get_gcrs(
        obstime=interp_times.reshape((num_grid_times, 1))
    )  # [T, num_ants]
    antennas_position_gcrs = antennas_gcrs.cartesian.xyz
    antennas_velocity_gcrs = antennas_gcrs.velocity.d_xyz

    (earth_position_bcrs, earth_velocity_bcrs) = ac.get_body_barycentric_posvel(
        body='earth',
        time=interp_times
    )  # [T]
    earth_position_bcrs = earth_position_bcrs.xyz
    earth_velocity_bcrs = earth_velocity_bcrs.xyz

    sun_position_bcrs = ac.get_body_barycentric(
        body='sun',
        time=interp_times
    )  # [T]
    sun_position_bcrs = sun_position_bcrs.xyz
    R_earth_bcrs = earth_position_bcrs - sun_position_bcrs  # [T]

    system_positions_bcrs = []
    for body in bodies_except_earth:
        body_position_bcrs = ac.get_body_barycentric(
            body=body,
            time=interp_times
        )  # [T, N]
        body_position_bcrs = np.transpose(body_position_bcrs.xyz, (1, 0))  # [T, 3]
        system_positions_bcrs.append(body_position_bcrs)
    system_positions_bcrs = np.stack(system_positions_bcrs, axis=1)  # [T, N, 3]

    GM_system = au.Quantity([GM_bodies[body] for body in bodies_except_earth])

    if verbose:
        print(f"Computing UVW for phase center: {phase_center}")
        print(f"Number of antennas: {len(antennas)}")
        print(f"Number of times: {len(times)}")
        print(f"Interpolation resolution: {resolution}")
        print(f"Number interpolation points: {num_grid_times}")
        print(f"Gravitational effects included from:")
        for body in sorted(bodies_except_earth + ('earth',)):
            print(f"\t{body.title()}")

    # Convert to JAX

    x_antennas_gcrs = quantity_to_jnp(
        np.transpose(antennas_position_gcrs, (1, 2, 0))
    )  # [T, num_ants, 3]

    w_antennas_gcrs = quantity_to_jnp(
        np.transpose(antennas_velocity_gcrs, (1, 2, 0))
    )  # [T, num_ants, 3]

    X_earth_bcrs = quantity_to_jnp(
        np.transpose(earth_position_bcrs, (1, 0))
    )  # [T, 3]
    V_earth_bcrs = quantity_to_jnp(
        np.transpose(earth_velocity_bcrs, (1, 0))
    )  # [T, 3]

    R_earth_bcrs = quantity_to_jnp(
        np.transpose(R_earth_bcrs, (1, 0))
    )  # [T, 3]

    system_positions_bcrs = quantity_to_jnp(
        system_positions_bcrs
    )  # [T, N_J, 3]

    GM_J = quantity_to_jnp(GM_system)  # [N_J]

    ref_time = interp_times[0]

    interp_times_jax = jnp.asarray((interp_times - ref_time).sec)  # [T]

    times_jax = jnp.asarray((times - ref_time).sec)  # [num_time]
    if with_autocorr:
        antenna_1, antenna_2 = jnp.asarray(list(itertools.combinations_with_replacement(range(num_ants), 2))).T
    else:
        antenna_1, antenna_2 = jnp.asarray(list(itertools.combinations(range(num_ants), 2))).T

    # Create interpolation objects
    X_earth_bcrs = InterpolatedArray(
        times=interp_times_jax,
        values=X_earth_bcrs,
        axis=0
    )
    V_earth_bcrs = InterpolatedArray(
        times=interp_times_jax,
        values=V_earth_bcrs,
        axis=0
    )

    R_earth_bcrs = InterpolatedArray(
        times=interp_times_jax,
        values=R_earth_bcrs,
        axis=0
    )

    X_J_bcrs = InterpolatedArray(
        times=interp_times_jax,
        values=system_positions_bcrs,
        axis=0
    )

    def _delay_from_lm(l: jax.Array, m: jax.Array,
                       t1: jax.Array, i1: jax.Array, i2: jax.Array) -> jax.Array:
        n = jnp.sqrt(1. - jnp.square(l) - jnp.square(m))
        ra, dec = perley_icrs_from_lmn(l=l, m=m, n=n, ra0=ra0, dec0=dec0)
        K_bcrs = celestial_to_cartesian(ra, dec)

        x_1_gcrs = InterpolatedArray(
            times=interp_times_jax,
            values=x_antennas_gcrs[:, i1, :],
            axis=0
        )

        x_2_gcrs = InterpolatedArray(
            times=interp_times_jax,
            values=x_antennas_gcrs[:, i2, :],
            axis=0
        )

        w_1_gcrs = InterpolatedArray(
            times=interp_times_jax,
            values=w_antennas_gcrs[:, i1, :],
            axis=0
        )

        w_2_gcrs = InterpolatedArray(
            times=interp_times_jax,
            values=w_antennas_gcrs[:, i2, :],
            axis=0
        )

        delta_t = delay(
            K_bcrs=K_bcrs,
            t1=t1,
            x_1_gcrs=x_1_gcrs,
            x_2_gcrs=x_2_gcrs,
            w_1_gcrs=w_1_gcrs,
            w_2_gcrs=w_2_gcrs,
            X_earth_bcrs=X_earth_bcrs,
            V_earth_bcrs=V_earth_bcrs,
            R_earth_bcrs=R_earth_bcrs,
            X_J_bcrs=X_J_bcrs,
            GM_J=GM_J
        )  # s
        # Unsure why the negative sign needs to be introduced to match,
        # since delta_t=t2-t1 is time for signal to travel from 1 to 2.
        return -delta_t * quantity_to_jnp(const.c)

    @partial(jax.vmap, in_axes=(0, None, None))
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def _compute_uvw(t1, i1, i2):
        l = m = jnp.asarray(0.)
        # tau = (-?) c * delay = u l + v m + w sqrt(1 - l^2 - m^2) ==> w = tau(l=0, m=0)
        # d/dl tau = u + w l / sqrt(1 - l^2 - m^2) ==> u = d/dl tau(l=0, m=0)
        # d/dm tau = v + w m / sqrt(1 - l^2 - m^2) ==> v = d/dm tau(l=0, m=0)
        w, (u, v) = jax.value_and_grad(_delay_from_lm, argnums=(0, 1))(l, m, t1, i1, i2)
        return jnp.stack([u, v, w], axis=-1)

    return np.asarray(_compute_uvw(times_jax, antenna_1, antenna_2)) * au.m


def delay(
        K_bcrs: jax.Array,
        t1: jax.Array,
        x_1_gcrs: InterpolatedArray,
        x_2_gcrs: InterpolatedArray,
        w_1_gcrs: InterpolatedArray,
        w_2_gcrs: InterpolatedArray,
        X_earth_bcrs: InterpolatedArray,
        V_earth_bcrs: InterpolatedArray,
        R_earth_bcrs: InterpolatedArray,
        X_J_bcrs: InterpolatedArray,
        GM_J: jax.Array,
        include_atmosphere: bool = False
):
    """
    The VLBI delay model of [1]. Should not be used for sources inside the solar system.

    Args:
        K_bcrs: Unit vector to source in absence of aberation.
        t1: time at first antenna (which serves as reference).
        x_1_gcrs: Interpolator for station 1 position.
        x_2_gcrs: Interpolator for station 2 position.
        w_1_gcrs: Interpolator for station 1 velocity.
        w_2_gcrs: Interpolator for station 2 velocity.
        X_earth_bcrs: Interpolator for geocenter position.
        V_earth_bcrs: Interpolator for geocenter velocity.
        R_earth_bcrs: Interpolator for vector from Sun to geocenter.
        X_J_bcrs: [num_J] Interpolator for position of J-th body.
        GM_J: [num_J] GM of J-th body.
        include_atmosphere: if True then add atmosphere delay model.

    Returns:
        The delay in seconds at time t1, for baseline b=x2-x1.

    References:
        [1] IERS Technical Note No. 36, IERS Conventions (2010)
            https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36.pdf
    """
    c = quantity_to_jnp(const.c)  # m / s
    L_G = jnp.asarray(6.969290134e-10)  # 1 - d(TT) / d(TCG)
    GM_earth = quantity_to_jnp(const.GM_earth)  # m^3 / s^2

    b_gcrs = x_2_gcrs(t1) - x_1_gcrs(t1)

    # Eq 11.6, accurate for use in 11.3 and 11.5
    X_1_bcrs = X_earth_bcrs(t1) + x_1_gcrs(t1)  # [3]
    X_2_bcrs = X_earth_bcrs(t1) + x_2_gcrs(t1)  # [3]

    # Eq 11.3
    t_1J = jnp.minimum(t1, t1 - ((X_J_bcrs(t1) - X_1_bcrs) @ K_bcrs) / c)  # [num_J]
    # Eq 11.4
    R_1J = X_1_bcrs - X_J_bcrs(t_1J)  # [num_J, 3]
    # Eq 11.5
    R_2J = X_2_bcrs - X_J_bcrs(t_1J) - V_earth_bcrs(t1) * (K_bcrs @ b_gcrs) / c  # [num_J, 3]

    # Eq 11.1
    delta_T_grav_J = 2. * (GM_J) / c ** 3 * jnp.log(
        (norm(R_1J) + R_1J @ K_bcrs) / (norm(R_2J) + R_2J @ K_bcrs)
    )  # [num_J]

    # Eq 11.2
    delta_T_grav_earth = 2. * GM_earth / c ** 3 * jnp.log(
        (norm(x_1_gcrs(t1)) + K_bcrs @ x_1_gcrs(t1)) / (norm(x_2_gcrs(t1)) + K_bcrs @ x_2_gcrs(t1))
    )  # []

    # Eq 11.7
    delta_T_grav = jnp.sum(delta_T_grav_J) + delta_T_grav_earth  # []

    # Eq 11.9: (delta_T_grav - K.b/c [1 - A / c^2] - V.b/c^2 [1 + B / c]) / (1 + C / c)
    U = GM_earth / jnp.linalg.norm(R_earth_bcrs(t1))
    A = 2. * U + 0.5 * norm2(V_earth_bcrs(t1)) + V_earth_bcrs(t1) @ w_2_gcrs(t1)
    B = 0.5 * (K_bcrs @ V_earth_bcrs(t1))
    C = K_bcrs @ (V_earth_bcrs(t1) + w_2_gcrs(t1))
    vacuum_delay = (delta_T_grav - (K_bcrs @ b_gcrs) / c * (1. - A / c ** 2) - (V_earth_bcrs(t1) @ b_gcrs) / c ** 2 * (
            1 + B / c)) / (1 + C / c)

    if include_atmosphere:
        # aberated source vectors for geodesics (x_1_gcrs, k_1_gcrs), (x_2_gcrs, k_2_gcrs)
        # k_1_gcrs = K_bcrs + (V_earth_bcrs(t1) + w_1_gcrs(t1) - K_bcrs * (K_bcrs @ (V_earth_bcrs(t1) + w_1_gcrs(t1)))) / c
        # delay_atm_1 = ...
        # k_2_gcrs = K_bcrs + (V_earth_bcrs(t1) + w_2_gcrs(t1) - K_bcrs * (K_bcrs @ (V_earth_bcrs(t1) + w_2_gcrs(t1)))) / c
        # delay_atm_2 = ...
        # total_delay = vacuum_delay + (delay_atm_2 - delay_atm_1) + delay_atm_1 * (K_bcrs @ (w_2_gcrs(t1) - w_1_gcrs(t1))) / c
        raise NotImplementedError(f"Atmosphere model is not implemented yet.")
    else:
        total_delay = vacuum_delay

    # delay produced by a correlator may be considered to be, within the uncertainty aimed at, equal
    # to the TT coordinate time interval. This is because station clocks are synchronised and syntonised,
    # i.e. have same rate as TT. However, the analysis above give Geocentric Coordinate Time (TCG). The delays must be
    # made TT-compatible, via dTT = (1 - L_G) dTCG

    total_delay *= (1 - L_G)

    return total_delay
