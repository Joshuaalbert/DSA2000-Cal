import dataclasses
import itertools
import time as time_mod
import warnings
from typing import Tuple

import jax
import numpy as np
from astropy import coordinates as ac, time as at, units as au, constants as const
from jax import config, numpy as jnp
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.delay_models.uvw_utils import norm


@dataclasses.dataclass(eq=False)
class NearFieldDelayEngine:
    """
    Compute the delay for baselines for stationary sources on geoid, and observers also on geoid.
    """
    antennas: ac.EarthLocation
    start_time: at.Time
    end_time: at.Time

    resolution: au.Quantity | None = None
    ref_location: ac.EarthLocation | None = None
    verbose: bool = False

    def __post_init__(self):
        if not config.jax_enable_x64:
            warnings.warn("jax_enable_x64 is not set, UVW computations may be inaccurate.")

        if self.ref_location is None:
            self.ref_location = self.antennas[0]

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
            if self.verbose:
                print(f"Setting resolution to {self.resolution} suitable for max baseline of {max_baseline}.")

        if not self.resolution.unit.is_equivalent(au.s):
            raise ValueError(f"resolution must be in seconds got {self.resolution.unit}")

        if len(self.antennas.shape) != 1:
            raise ValueError(f"antennas must be 1D got {self.antennas.shape}")

        if self.antennas.shape[0] < 2:
            raise ValueError(f"Need at least 2 antennas to form a baseline.")

        bodies_except_earth = ()

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
        self.interp_times = interp_times = start_grid_time + np.arange(num_grid_times) * self.resolution  # [T]

        if self.verbose:
            print(f"Computing near field delay")
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
        antennas_position_gcrs = antennas_gcrs.cartesian.xyz  # [3, T, num_ants]

        enu_origin = self.ref_location.get_gcrs(obstime=interp_times)
        enu_origin_gcrs = enu_origin.cartesian.xyz  # [3, T]

        enu_gcrs = ENU(
            east=np.reshape([1, 0, 0], (1, 3)),
            north=np.reshape([0, 1, 0], (1, 3)),
            up=np.reshape([0, 0, 1], (1, 3)),
            location=self.ref_location,
            obstime=interp_times.reshape((num_grid_times, 1))  # [T, 3]
        ).transform_to(ac.GCRS(obstime=interp_times.reshape((num_grid_times, 1))))
        enu_coords_gcrs = enu_gcrs.cartesian.xyz  # [3, T, 3]

        ephem_compute_time = time_mod.time() - ephem_compute_t0

        if self.verbose:
            print(f"Time to compute ephemeris: {ephem_compute_time:.2f} s")

        # Convert to JAX

        self.x_antennas_gcrs = quantity_to_jnp(
            np.transpose(antennas_position_gcrs, (1, 2, 0))
        )  # [T, num_ants, 3]

        self.enu_origin_gcrs = quantity_to_jnp(
            np.transpose(enu_origin_gcrs, (1, 0))
        )  # [T, 3]

        self.enu_coords_gcrs = quantity_to_jnp(
            np.transpose(enu_coords_gcrs, (1, 2, 0))
        )  # [T, 3, 3]

        self.interp_times_jax = jnp.asarray((interp_times.tt - self.ref_time.tt).sec)  # [T]

    def construct_x_0_gcrs(self, emitter: ac.EarthLocation) -> InterpolatedArray:
        """
        Construct the emitter location as a linear combination of radius vectors from first antenna to antennas 1,2,3.

        Args:
            emitter: [E] the location of the emitter.

        Returns:
            interpolator for emitter
        """
        obstimes = self.interp_times.reshape((-1, 1))
        emitter_gcrs = emitter.reshape((1, -1)).get_itrs(
            obstime=obstimes
        ).transform_to(
            ac.GCRS(
                obstime=obstimes
            )
        )  # [T, E]
        emitter_position_gcrs = emitter_gcrs.cartesian.xyz
        x_emitter_gcrs = quantity_to_jnp(
            np.transpose(emitter_position_gcrs, (1, 2, 0))
        )  # [T, E, 3]

        return InterpolatedArray(
            x=self.interp_times_jax,
            values=x_emitter_gcrs,
            axis=0,
            regular_grid=True
        )  # [T, E, 3]

    def _construct_x_0_gcrs_from_projection(self, a_east: jax.Array, a_north: jax.Array,
                                            a_up: jax.Array) -> InterpolatedArray:
        """
        Construct the emitter location as a linear combination of radius vectors from first antenna to antennas 1,2,3.

        Args:
            a_east: [E] coefficient for east direction from antenna[0]
            a_north: [E] coefficient for north direction
            a_up: [E] coefficient for up direction

        Returns:
            interpolator for emitter
        """
        a_east = jnp.reshape(a_east, (-1,))
        a_north = jnp.reshape(a_north, (-1,))
        a_up = jnp.reshape(a_up, (-1,))
        d_origin = self.enu_origin_gcrs[:, None, :]  # [T, 1, 3]
        d_east = self.enu_coords_gcrs[:, 0:1, :]
        d_north = self.enu_coords_gcrs[:, 1:2, :]
        d_up = self.enu_coords_gcrs[:, 2:3, :]

        values = a_east[:, None] * d_east + a_north[:, None] * d_north + a_up[:, None] * d_up + d_origin  # [T, E, 3]

        return InterpolatedArray(
            x=self.interp_times_jax,
            values=values,
            axis=0,
            regular_grid=True
        )  # [T, E, 3]

    def compute_delay_from_projection_jax(self,
                                          a_east: jax.Array,
                                          a_north: jax.Array,
                                          a_up: jax.Array,
                                          t1: jax.Array,
                                          i1: jax.Array,
                                          i2: jax.Array
                                          ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Compute the delay for a given phase center, using VLBI delay model.

        Args:
            a_east: [E] coefficient for east direction from antenna[0]
            a_north: [E] coefficient for (x2 - x0) direction
            a_up: [E] coefficient for (x3 - x0) direction
            t1: the time of observation, in tt scale in seconds, relative to the first time.
            i1: the index of the first antenna.
            i2: the index of the second antenna.

        Returns:
            delay: [E] the delay in meters, i.e. light travel distance, for each emitter.
            dist2: the distance in meters, for baseline b=x2-x0.
            dist1: the distance in meters, for baseline b=x1-x0.
        """
        if np.shape(a_east) != np.shape(a_north) or np.shape(a_east) != np.shape(a_up):
            raise ValueError(
                f"a_east, a_north, a_up must have the same shape "
                f"got {np.shape(a_east)}, {np.shape(a_north)}, {np.shape(a_up)}"
            )
        delay, dist2, dist1 = self._compute_delay_jax(
            self._construct_x_0_gcrs_from_projection(a_east, a_north, a_up),
            t1, i1, i2
        )
        return delay.reshape(np.shape(a_east)), dist2.reshape(np.shape(a_east)), dist1.reshape(np.shape(a_east))

    def compute_delay_from_emitter_jax(self,
                                       emitter: ac.EarthLocation,
                                       t1: jax.Array,
                                       i1: jax.Array,
                                       i2: jax.Array
                                       ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Compute the delay for a given phase center, using VLBI delay model.

        Args:
            emitter: [E] the location of the emitter.
            t1: the time of observation, in tt scale in seconds, relative to the first time.
            i1: the index of the first antenna.
            i2: the index of the second antenna.

        Returns:
            delay: [E] the delay in meters, i.e. light travel distance, for each emitter.
            dist2: the distance in meters, for baseline b=x2-x0.
            dist1: the distance in meters, for baseline b=x1-x0.
        """
        delay, dist2, dist1 = self._compute_delay_jax(
            self.construct_x_0_gcrs(emitter=emitter),
            t1, i1, i2
        )
        return delay.reshape(emitter.shape), dist2.reshape(emitter.shape), dist1.reshape(emitter.shape)

    def _compute_delay_jax(self,
                           x_0_gcrs: InterpolatedArray,
                           t1: jax.Array,
                           i1: jax.Array,
                           i2: jax.Array
                           ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Compute the delay for a given phase center, using VLBI delay model.

        Args:
            t1: the time of observation, in tt scale in seconds, relative to the first time.
            i1: the index of the first antenna.
            i2: the index of the second antenna.

        Returns:
            delay: the delay in meters, i.e. light travel distance.
            dist2: the distance in meters, for baseline b=x2-x0.
            dist1: the distance in meters, for baseline b=x1-x0.
        """

        if np.shape(t1) != () or np.shape(i1) != () or np.shape(i2) != ():
            raise ValueError(f"t1, i1, i2 must be scalars got {np.shape(t1)}, {np.shape(i1)}, {np.shape(i2)}")

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

        delta_t, dist2, dist1 = near_field_delay(
            t1=t1,
            x_0_gcrs=x_0_gcrs,
            x_1_gcrs=x_1_gcrs,
            x_2_gcrs=x_2_gcrs
        )  # s
        # Unsure why the negative sign needs to be introduced to match,
        # since delta_t=t2-t1 is time for signal to travel from 1 to 2.
        return delta_t, dist2, dist1

    def time_to_jnp(self, times: at.Time) -> jax.Array:
        """
        Make the times relative to the first time, in seconds in tt scale.

        Args:
            times: [...] Time of observation.

        Returns:
            times_jax: [...] Time of observation, in tt scale in seconds, relative to the first time.
        """
        return jnp.asarray((times.tt - self.ref_time.tt).sec)  # [N]


def near_field_delay(
        t1: jax.Array,
        x_0_gcrs: InterpolatedArray,
        x_1_gcrs: InterpolatedArray,
        x_2_gcrs: InterpolatedArray
):
    """
    The VLBI delay model of [1] built on [2]. Only for stationary Earth-based near field sources and observers.

    Args:
        t1: time at first antenna (which serves as reference).
        x_0_gcrs: [E] Interpolator for emitter position.
        x_1_gcrs: Interpolator for station 1 position.
        x_2_gcrs: Interpolator for station 2 position.

    Returns:
        [E] The delay in metres at time t1, for baseline b=x2-x1.
        [E] The distance in metres at time t1, for baseline b=x2-x0.
        [E] The distance in metres at time t1, for baseline b=x1-x0.


    References:
        [1] IERS Technical Note No. 36, IERS Conventions (2010)
            https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36.pdf
        [2] Klioner, S. A. (1991). General relativistic model of VLBI delay observations.
            https://www.researchgate.net/publication/253171626
    """

    # When we take GCRS we are able to work nicely in the Schwarzschild metric for the Earth.
    # If we want to include planetary effects, we would perturb the metric 11.17 of [1].

    c = quantity_to_jnp(const.c)  # m / s
    L_G = jnp.asarray(6.969290134e-10)  # 1 - d(TT) / d(TCG), i.e. the mean rate of proper time on Earth's surface.
    GM_earth = quantity_to_jnp(const.GM_earth)  # m^3 / s^2

    def _delay(x_i_gcrs: InterpolatedArray) -> Tuple[jax.Array, jax.Array]:
        # Eq 3.14 in [2] -- i.e. only Earth's potential
        b_gcrs = x_i_gcrs(t1) - x_0_gcrs(t1)  # [E, 3]
        # jax.debug.print("b_gcrs={b_gcrs}",b_gcrs=b_gcrs)
        # jax.debug.print("norm(x_i_gcrs(t1))={x}",x=norm(x_i_gcrs(t1)))
        # jax.debug.print("norm(x_0_gcrs(t1), axis=-1)={x}",x=norm(x_0_gcrs(t1), axis=-1))
        delta_T_grav_earth = 2. * GM_earth / c ** 2 * jnp.log(
            (
                    norm(x_i_gcrs(t1)) + norm(x_0_gcrs(t1), axis=-1) + norm(b_gcrs, axis=-1)
            ) / (
                    norm(x_i_gcrs(t1)) + norm(x_0_gcrs(t1), axis=-1) - norm(b_gcrs, axis=-1)
            )
        )  # [E]
        # jax.debug.print("delta_T_grav_earth={delta_T_grav_earth}",delta_T_grav_earth=delta_T_grav_earth)
        # Effect of Earth's potential on the delay ~ 1e-5 m
        coordinate_delay = norm(b_gcrs, axis=-1) + delta_T_grav_earth  # [E]

        # atomic clocks tick at the rate of proper time, thus we need to covert to proper time.
        # 4.19 in [2] -- for Earth based observers the rate of proper time is the same as TT (by construction).
        proper_delay = (1. - L_G) * coordinate_delay
        return proper_delay, norm(b_gcrs, axis=-1)

    proper_time1, dist1 = _delay(x_1_gcrs)
    proper_time2, dist2 = _delay(x_2_gcrs)
    return proper_time1 - proper_time2, dist2, dist1
