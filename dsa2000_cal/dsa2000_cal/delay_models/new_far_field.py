import dataclasses
import itertools
import time as time_mod
import warnings
from functools import partial
from typing import Tuple, NamedTuple

import jax
import numpy as np
from astropy import coordinates as ac, time as at, units as au, constants as const
from jax import config, numpy as jnp, lax

from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.delay_models.uvw_utils import perley_icrs_from_lmn, celestial_to_cartesian, norm, norm2


class VisibilityCoords(NamedTuple):
    """
    Coordinates for a single visibility.
    """
    uvw: jax.Array | np.ndarray  # [rows, 3] the uvw coordinates
    time_obs: jax.Array | np.ndarray  # [rows] the time relative to the reference time (observation start)
    antenna_1: jax.Array | np.ndarray  # [rows] the first antenna
    antenna_2: jax.Array | np.ndarray  # [rows] the second antenna
    time_idx: jax.Array | np.ndarray  # [rows] the time index


@dataclasses.dataclass(eq=False)
class FarFieldDelayEngine:
    """
    Engine to compute the delay for far field sources, outside the solar system. This includes the effects of
    gravitational bodies in the solar system. Which contributes to delay on the order of 0.2 mm * (|baseline|/1km).

    UVW coordinates are computed using the delay model, via the standard approximation:

    delay(l,m) ~ u l + v m + w sqrt(1 - l^2 - m^2)

    from which it follows:

    w = delay(l=0, m=0)
    u = d/dl delay(l=0, m=0)
    v = d/dm delay(l=0, m=0)

    The delay error based on this approximation is then:

    error(l,m) = delay(l,m) - (u l + v m + w sqrt(1 - l^2 - m^2))

    The delay model is based on the IERS conventions [1] and the general relativistic model of VLBI delay observations [2].

    References:
        [1] IERS Technical Note No. 36, IERS Conventions (2010)
            https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36.pdf
        [2] Klioner, S. A. (1991). General relativistic model of VLBI delay observations.
            https://www.researchgate.net/publication/253171626
    """
    antennas: ac.EarthLocation
    start_time: at.Time
    end_time: at.Time
    phase_center: ac.ICRS

    resolution: au.Quantity | None = None
    verbose: bool = False

    def __post_init__(self):
        if not config.jax_enable_x64:
            warnings.warn("jax_enable_x64 is not set, UVW computations may be inaccurate.")

        if self.resolution is None:
            # compute max baseline
            antenna_1, antenna_2 = np.asarray(list(itertools.combinations(range(len(self.antennas)), 2))).T
            antennas_itrs = self.antennas.get_itrs().cartesian.xyz.T
            max_baseline = np.max(np.linalg.norm(antennas_itrs[antenna_2] - antennas_itrs[antenna_1], axis=-1))
            # Select resolution to keep interpolation error below 1 mm
            if max_baseline <= 10 * au.km:
                self.resolution = 10 * au.s
            elif max_baseline <= 100 * au.km:
                self.resolution = 4 * au.s
            elif max_baseline <= 1000 * au.km:
                self.resolution = 1 * au.s
            else:
                warnings.warn(
                    f"Max baseline is {max_baseline} > 1000 km, setting resolution to 0.1 s, "
                    f"may lead to slow ephemeris calculations."
                )
                self.resolution = 0.1 * au.s

        if not self.resolution.unit.is_equivalent(au.s):
            raise ValueError(f"resolution must be in seconds got {self.resolution.unit}")

        if len(self.antennas.shape) != 1:
            raise ValueError(f"antennas must be 1D got {self.antennas.shape}")

        if self.antennas.shape[0] < 2:
            raise ValueError(f"Need at least 2 antennas to form a baseline.")

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
        self.J2_coefficients = {
            'sun': 2.2e-7,
            'moon': 2.034e-4,
            'mercury': 6.0e-6,
            'venus': 4.458e-6,
            'earth': 1.08263e-3,
            'mars': 1.960e-3,
            'jupiter': 1.4736e-2,
            'saturn': 1.6298e-2,
            'uranus': 3.34343e-3,
            'neptune': 3.411e-3
        }
        self.radii = {
            'sun': const.R_sun,
            'moon': 0.2727 * const.R_earth,
            'mercury': 0.3829 * const.R_earth,
            'venus': 0.9499 * const.R_earth,
            'earth': const.R_earth,
            'mars': 0.5320 * const.R_earth,
            'jupiter': const.R_jup,
            'saturn': 9.45 * const.R_earth,
            'uranus': 4.01 * const.R_earth,
            'neptune': 3.88 * const.R_earth
        }

        self.ra0 = jnp.asarray(self.phase_center.ra.rad)
        self.dec0 = jnp.asarray(self.phase_center.dec.rad)
        if not self.start_time.isscalar or not self.end_time.isscalar:
            raise ValueError(f"start_time and end_time must be scalar got {self.start_time} and {self.end_time}")

        self.ref_time = start_time = self.start_time.tt
        end_time = self.end_time.tt

        earth_light_cross_time = 2. * const.R_earth / const.c

        start_grid_time = start_time - earth_light_cross_time
        end_grid_time = end_time + earth_light_cross_time

        num_grid_times = int(np.ceil(float((end_grid_time - start_grid_time) / self.resolution))) + 1
        num_ants = len(self.antennas)

        # Define the interpolation grid
        interp_times = start_grid_time + np.arange(num_grid_times) * self.resolution  # [T]

        if self.verbose:
            print(f"Computing UVW for phase center: {self.phase_center}")
            print(f"Number of antennas: {len(self.antennas)}")
            print(f"Between {start_time} and {end_time} ({(end_time - start_time).sec} s)")
            print(f"Interpolation resolution: {self.resolution}")
            print(f"Number interpolation points: {num_grid_times}")
            print(f"Gravitational effects included from:")
            for body in sorted(bodies_except_earth + ('earth',)):
                print(f"\t{body.title()}")

        # Compute ephemeris'
        ephem_compute_t0 = time_mod.time()

        # Define the antennas
        antennas_gcrs = self.antennas.reshape((1, num_ants)).get_gcrs(
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
        system_velocity_bcrs = []
        for body in bodies_except_earth:
            body_position_bcrs, body_velocity_bcrs = ac.get_body_barycentric_posvel(
                body=body,
                time=interp_times
            )  # [T, N]
            body_position_bcrs = np.transpose(body_position_bcrs.xyz, (1, 0))  # [T, 3]
            body_velocity_bcrs = np.transpose(body_velocity_bcrs.xyz, (1, 0))  # [T, 3]
            system_positions_bcrs.append(body_position_bcrs)
            system_velocity_bcrs.append(body_velocity_bcrs)
        system_positions_bcrs = np.stack(system_positions_bcrs, axis=1)  # [T, N, 3]
        system_velocity_bcrs = np.stack(system_velocity_bcrs, axis=1)  # [T, N, 3]

        GM_system = au.Quantity([GM_bodies[body] for body in bodies_except_earth])

        ephem_compute_time = time_mod.time() - ephem_compute_t0

        if self.verbose:
            print(f"Time to compute ephemeris: {ephem_compute_time:.2f} s")

        # Convert to JAX

        self.x_antennas_gcrs = quantity_to_jnp(
            np.transpose(antennas_position_gcrs, (1, 2, 0))
        )  # [T, num_ants, 3]

        self.w_antennas_gcrs = quantity_to_jnp(
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
        system_velocities_bcrs = quantity_to_jnp(
            system_velocity_bcrs
        )  # [T, N_J, 3]

        self.GM_J = quantity_to_jnp(GM_system)  # [N_J]

        self.interp_times_jax = interp_times_jax = jnp.asarray((interp_times - self.ref_time).sec)  # [T]

        # Create interpolation objects
        self.X_earth_bcrs = InterpolatedArray(
            x=interp_times_jax,
            values=X_earth_bcrs,
            axis=0,
            regular_grid=True
        )
        self.V_earth_bcrs = InterpolatedArray(
            x=interp_times_jax,
            values=V_earth_bcrs,
            axis=0,
            regular_grid=True
        )

        self.R_earth_bcrs = InterpolatedArray(
            x=interp_times_jax,
            values=R_earth_bcrs,
            axis=0,
            regular_grid=True
        )

        self.X_J_bcrs = InterpolatedArray(
            x=interp_times_jax,
            values=system_positions_bcrs,
            axis=0,
            regular_grid=True
        )

        self.V_J_bcrs = InterpolatedArray(
            x=interp_times_jax,
            values=system_velocities_bcrs,
            axis=0,
            regular_grid=True
        )

    def compute_delay_from_lm_jax(self,
                                  l: jax.Array, m: jax.Array,
                                  t1: jax.Array, i1: jax.Array,
                                  i2: jax.Array) -> jax.Array:
        """
        Compute the delay for a given phase center, using VLBI delay model.

        Args:
            l: the l coordinate.
            m: the m coordinate.
            t1: the time of observation, in tt scale in seconds, relative to the first time.
            i1: the index of the first antenna.
            i2: the index of the second antenna.

        Returns:
            delay: the delay in meters, i.e. light travel distance, from i2 to antenna i1.
        """

        if np.shape(l) != () or np.shape(m) != ():
            raise ValueError(f"l, m must be scalars got {np.shape(l)}, {np.shape(m)}")

        if np.shape(t1) != () or np.shape(i1) != () or np.shape(i2) != ():
            raise ValueError(f"t1, i1, i2 must be scalars got {np.shape(t1)}, {np.shape(i1)}, {np.shape(i2)}")

        n = jnp.sqrt(1. - (jnp.square(l) + jnp.square(m)))
        ra, dec = perley_icrs_from_lmn(l=l, m=m, n=n, ra0=self.ra0, dec0=self.dec0)
        K_bcrs = celestial_to_cartesian(ra, dec)

        x_1_gcrs = InterpolatedArray(
            x=self.interp_times_jax,
            values=self.x_antennas_gcrs[:, i1, :],
            axis=0,
            regular_grid=True
        )

        x_2_gcrs = InterpolatedArray(
            x=self.interp_times_jax,
            values=self.x_antennas_gcrs[:, i2, :],
            axis=0,
            regular_grid=True
        )

        w_1_gcrs = InterpolatedArray(
            x=self.interp_times_jax,
            values=self.w_antennas_gcrs[:, i1, :],
            axis=0,
            regular_grid=True
        )

        w_2_gcrs = InterpolatedArray(
            x=self.interp_times_jax,
            values=self.w_antennas_gcrs[:, i2, :],
            axis=0,
            regular_grid=True
        )

        delta_t = far_field_delay(
            K_bcrs=K_bcrs,
            t1=t1,
            x_1_gcrs=x_1_gcrs,
            x_2_gcrs=x_2_gcrs,
            w_1_gcrs=w_1_gcrs,
            w_2_gcrs=w_2_gcrs,
            X_earth_bcrs=self.X_earth_bcrs,
            V_earth_bcrs=self.V_earth_bcrs,
            R_earth_bcrs=self.R_earth_bcrs,
            X_J_bcrs=self.X_J_bcrs,
            V_J_bcrs=self.V_J_bcrs,
            GM_J=self.GM_J
        )  # s
        # Unsure why the negative sign needs to be introduced to match,
        # since delta_t=t2-t1 is time for signal to travel from 1 to 2.

        # I *think* it is because we've flipped direction of photon by using K_bcrs for photon travel.
        # Then essentially, we're computing the delay for the signal to travel from 2 to 1, but there should be an error
        # from using t1 for reference point.
        return -delta_t

    def _single_compute_uvw(self, t1: jax.Array, i1: jax.Array, i2: jax.Array) -> jax.Array:
        """
        Compute the UVW coordinates for a given phase center, using VLBI delay model.

        Args:
            t1: time of observation, in tt scale in seconds, relative to the first time.
            i1: index of the first antenna.
            i2: index of the second antenna.

        Returns:
            uvw: [3] UVW coordinates in meters.
        """
        l = m = jnp.asarray(0.)
        # tau = (-?) c * delay = u l + v m + w sqrt(1 - l^2 - m^2) ==> w = tau(l=0, m=0)
        # d/dl tau = u + w l / sqrt(1 - l^2 - m^2) ==> u = d/dl tau(l=0, m=0)
        # d/dm tau = v + w m / sqrt(1 - l^2 - m^2) ==> v = d/dm tau(l=0, m=0)
        w, (u, v) = jax.value_and_grad(self.compute_delay_from_lm_jax, argnums=(0, 1))(l, m, t1, i1, i2)
        return jnp.stack([u, v, w], axis=-1)  # [3]

    def compute_uvw_jax(self, times: jax.Array, antenna_1: jax.Array, antenna_2: jax.Array) -> jax.Array:
        """
        Compute the UVW coordinates for a given phase center, using VLBI delay model.

        Args:
            times: [N] Time of observation, in tt scale in seconds, relative to the first time.
            antenna_1: [N] Index of the first antenna.
            antenna_2: [N] Index of the second antenna.

        Returns:
            uvw: [N, 3] UVW coordinates in meters.
        """
        return jax.vmap(self._single_compute_uvw)(times, antenna_1, antenna_2)

    def time_to_jnp(self, times: at.Time) -> jax.Array:
        """
        Make the times relative to the first time, in seconds in tt scale.

        Args:
            times: [...] Time of observation.

        Returns:
            times_jax: [...] Time of observation, in tt scale in seconds, relative to the first time.
        """
        return jnp.asarray((times.tt - self.ref_time.tt).sec)  # [N]

    def compute_visibility_coords(self, times: jax.Array, with_autocorr: bool = True,
                                  convention: str = 'physical') -> VisibilityCoords:
        """
        Compute the UVW coordinates for a given phase center, using VLBI delay model in batched mode.

        Args:
            times: [T] Time of observation, in tt scale in seconds, relative to the first time.
            with_autocorr: bool, whether to include autocorrelations.
            convention: str, the convention to use for the UVW coordinates.

        Returns:
            visibility_coords: [T*B] stacked time-wise
        """
        if with_autocorr:
            antenna_1, antenna_2 = jnp.asarray(
                list(itertools.combinations_with_replacement(range(len(self.antennas)), 2))).T
        else:
            antenna_1, antenna_2 = jnp.asarray(list(itertools.combinations(range(len(self.antennas)), 2))).T

        if convention == 'physical':
            antenna_1, antenna_2 = antenna_1, antenna_2
        elif convention == 'casa':
            antenna_1, antenna_2 = antenna_2, antenna_1
        else:
            raise ValueError(f"Unknown convention {convention}")

        @partial(multi_vmap, in_mapping="[T],[T],[B],[B]", out_mapping="[T,B,...],[T,B],[T,B],[T,B],[T,B]",
                 verbose=True)
        def _compute_uvw_batched(time_idx: jax.Array, t1: jax.Array, i1: jax.Array, i2: jax.Array
                                 ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            return self._single_compute_uvw(t1, i1, i2), time_idx, t1, i1, i2

        num_baselines = len(antenna_2)
        num_times = len(times)
        num_rows = num_baselines * num_times
        uvw, time_idx, time_obs, antenna_1, antenna_2 = _compute_uvw_batched(
            jnp.arange(num_times), times, antenna_1, antenna_2)
        return VisibilityCoords(
            uvw=lax.reshape(uvw, (num_rows, 3)),
            time_idx=lax.reshape(time_idx, (num_rows,)),
            time_obs=lax.reshape(time_obs, (num_rows,)),
            antenna_1=lax.reshape(antenna_1, (num_rows,)),
            antenna_2=lax.reshape(antenna_2, (num_rows,))
        )


def far_field_delay(
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
        V_J_bcrs: InterpolatedArray,
        GM_J: jax.Array,
        include_atmosphere: bool = False
):
    """
    The VLBI delay model of [1] built on [2]. Should not be used for sources inside the solar system.

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
        V_J_bcrs: [num_J] Interpolator for velocity of J-th body.
        GM_J: [num_J] GM of J-th body.
        include_atmosphere: if True then add atmosphere delay model.

    Returns:
        The delay in metres at time t1, for baseline b=x2-x1.

    References:
        [1] IERS Technical Note No. 36, IERS Conventions (2010)
            https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36.pdf
        [2] Klioner, S. A. (1991). General relativistic model of VLBI delay observations.
            https://www.researchgate.net/publication/253171626
    """
    c = quantity_to_jnp(const.c)  # m / s
    L_G = jnp.asarray(6.969290134e-10)  # 1 - d(TT) / d(TCG)
    GM_earth = quantity_to_jnp(const.GM_earth)  # m^3 / s^2

    b_gcrs = x_2_gcrs(t1) - x_1_gcrs(t1)

    # Eq 11.6, accurate for use in 11.3 and 11.5
    X_1_bcrs = X_earth_bcrs(t1) + x_1_gcrs(t1)  # [3]
    X_2_bcrs = X_earth_bcrs(t1) + x_2_gcrs(t1)  # [3]

    # Eq 11.3 -- Time of closest approach of signal to J-th body
    t_1J = jnp.minimum(t1, t1 - ((X_J_bcrs(t1) - X_1_bcrs) @ K_bcrs) / c)  # [num_J]
    # Eq 11.4
    # X_J_bcrs(t_1J) -- Don't use interpolation, since it would mak interpolation axis too large
    X_J_bcrs_t1J = X_J_bcrs(t1) + V_J_bcrs(t1) * (t_1J - t1)[:, None]  # [num_J, 3]

    R_1J = X_1_bcrs - X_J_bcrs_t1J  # [num_J, 3]
    # Eq 11.5
    R_2J = X_2_bcrs - X_J_bcrs_t1J - V_earth_bcrs(t1) * (K_bcrs @ b_gcrs) / c  # [num_J, 3]

    # Eq 11.1
    delta_T_grav_J = 2. * (GM_J) / c ** 2 * (1. + (V_J_bcrs(t_1J) @ K_bcrs) / c) * jnp.log(
        (norm(R_1J) + R_1J @ K_bcrs) / (norm(R_2J) + R_2J @ K_bcrs)
    )  # [num_J]

    # Eq 11.2 =7.383900660090742e-11 - 7.383279239381223e-11 =
    delta_T_grav_earth = 2. * GM_earth / c ** 2 * (1. + (K_bcrs @ V_earth_bcrs(t1)) / c) * jnp.log(
        (norm(x_1_gcrs(t1)) + K_bcrs @ x_1_gcrs(t1)) / (norm(x_2_gcrs(t1)) + K_bcrs @ x_2_gcrs(t1))
    )  # []
    # (K @ V)/c term is around 1e-4 for Earth term (around 1e-15m delay)

    # Eq 11.7
    delta_T_grav = jnp.sum(delta_T_grav_J) + delta_T_grav_earth  # []
    # Around delta_T_grav=-0.00016 m * (|baseline|/1km)

    # Since we perform analysis in BCRS kinematically non-rotating dynamic frame we need to convert to GCRS TT-compatible
    # Eq 11.9: (delta_T_grav - K.b/c [1 - A / c^2] - V.b/c^2 [1 + B / c]) / (1 + C / c)
    U = GM_earth / jnp.linalg.norm(R_earth_bcrs(t1))
    A = 2. * U + 0.5 * norm2(V_earth_bcrs(t1)) + V_earth_bcrs(t1) @ w_2_gcrs(t1)
    B = 0.5 * (K_bcrs @ V_earth_bcrs(t1))
    C = K_bcrs @ (V_earth_bcrs(t1) + w_2_gcrs(t1))
    coordinate_delay_tcg = (
            (
                    delta_T_grav
                    - (K_bcrs @ b_gcrs) * (1. - A / c ** 2)
                    - (V_earth_bcrs(t1) @ b_gcrs) / c * (1 + B / c)
            ) / (
                    1 + C / c
            )
    )

    if include_atmosphere:
        # aberated source vectors for geodesics (x_1_gcrs, k_1_gcrs), (x_2_gcrs, k_2_gcrs)
        # k_1_gcrs = K_bcrs + (V_earth_bcrs(t1) + w_1_gcrs(t1) - K_bcrs * (K_bcrs @ (V_earth_bcrs(t1) + w_1_gcrs(t1)))) / c
        # delay_atm_1 = ...
        # k_2_gcrs = K_bcrs + (V_earth_bcrs(t1) + w_2_gcrs(t1) - K_bcrs * (K_bcrs @ (V_earth_bcrs(t1) + w_2_gcrs(t1)))) / c
        # delay_atm_2 = ...
        # coordinate_delay_tcg = coordinate_delay_tcg + (delay_atm_2 - delay_atm_1) + delay_atm_1 * (K_bcrs @ (w_2_gcrs(t1) - w_1_gcrs(t1))) / c
        raise NotImplementedError(f"Atmosphere model is not implemented.")

    # TT is defined with a rate that coincides with mean proper rate on the geoid,
    # so to first order proper and TT are the linearly related for observers on the geoid.

    proper_delay = (1 - L_G) * coordinate_delay_tcg

    return proper_delay
