import dataclasses
import itertools
import pickle
import time as time_mod
import warnings
from typing import Tuple, Any, List

import jax
import numpy as np
from astropy import coordinates as ac, time as at, units as au, constants as const
from jax import config, numpy as jnp
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_cal.delay_models.uvw_utils import norm


@dataclasses.dataclass(eq=False)
class BaseNearFieldDelayEngine:
    """
    Compute the delay for baselines for stationary sources on geoid, and observers also on geoid.
    """
    x_antennas_gcrs: InterpolatedArray  # (t) -> [A, 3]
    enu_origin_gcrs: InterpolatedArray  # (t) -> [3]
    enu_coords_gcrs: InterpolatedArray  # (t) -> [3, 3]

    @staticmethod
    def construct_x_0_gcrs(interp_times: at.Time, ref_time: at.Time, emitter: ac.EarthLocation,
                           regular_grid: bool = False) -> InterpolatedArray:
        """
        Construct the emitter location as a linear combination of radius vectors from first antenna to antennas 1,2,3.

        Args:
            interp_times: [T] the times to interpolate the emitter location at.
            ref_time: the reference time
            emitter: [E] the location of the emitter.

        Returns:
            interpolator for emitter
        """
        obstimes = interp_times.reshape((-1, 1))
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

        interp_times_jax = time_to_jnp(interp_times, ref_time)  # [T]

        return InterpolatedArray(
            x=interp_times_jax,
            values=x_emitter_gcrs,
            axis=0,
            regular_grid=regular_grid
        )  # (t) -> [E, 3]

    def construct_x_0_gcrs_from_projection(self,
                                           a_east: jax.Array, a_north: jax.Array, a_up: jax.Array
                                           ) -> InterpolatedArray:
        """
        Construct the emitter location as a linear combination of radius vectors from first antenna to antennas 1,2,3.

        Args:
            a_east: [E] coefficient for east direction from reference_location
            a_north: [E] coefficient for north direction from reference_location
            a_up: [E] coefficient for up direction from reference_location

        Returns:
            interpolator for emitter
        """
        a_east = jnp.reshape(a_east, (-1,))
        a_north = jnp.reshape(a_north, (-1,))
        a_up = jnp.reshape(a_up, (-1,))

        return (
                self.enu_coords_gcrs[0:1, :] * a_east[:, None]
                + self.enu_coords_gcrs[1:2, :] * a_north[:, None]
                + self.enu_coords_gcrs[2:3, :] * a_up[:, None]
                + self.enu_origin_gcrs[None, :]
        )  # [E, 3]

    def compute_delay(self,
                      x_0_gcrs: InterpolatedArray,
                      t1: jax.Array,
                      i1: jax.Array,
                      i2: jax.Array
                      ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Compute the delay for a given phase center, using VLBI delay model.

        Args:
            x_0_gcrs: [E, 3] Interpolator for emitter position in GCRS.
            t1: the time of observation, in tt scale in seconds, relative to the ref time.
            i1: the index of the first antenna.
            i2: the index of the second antenna.

        Returns:
            delay: the delay in meters, i.e. light travel distance.
            dist2: the distance in meters, for baseline b=x2-x0.
            dist1: the distance in meters, for baseline b=x1-x0.
        """

        if np.shape(t1) != () or np.shape(i1) != () or np.shape(i2) != ():
            raise ValueError(f"t1, i1, i2 must be scalars got {np.shape(t1)}, {np.shape(i1)}, {np.shape(i2)}")

        if len(x_0_gcrs.shape) != 2:
            raise ValueError(f"x_0_gcrs must be [E, 3] got {x_0_gcrs.shape}")

        x_1_gcrs = self.x_antennas_gcrs[i1, :]

        x_2_gcrs = self.x_antennas_gcrs[i2, :]

        delta_t, dist2, dist1 = near_field_delay(
            t1=t1,
            x_0_gcrs=x_0_gcrs,
            x_1_gcrs=x_1_gcrs,
            x_2_gcrs=x_2_gcrs
        )  # s
        # Unsure why the negative sign needs to be introduced to match,
        # since delta_t=t2-t1 is time for signal to travel from 1 to 2.
        return delta_t, dist2, dist1

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
    def flatten(cls, this: "BaseNearFieldDelayEngine") -> Tuple[List[Any], Tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        return (
            this.x_antennas_gcrs, this.enu_origin_gcrs,
            this.enu_coords_gcrs
        ), ()

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "BaseNearFieldDelayEngine":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        x_antennas_gcrs, enu_origin_gcrs, enu_coords_gcrs = children
        return BaseNearFieldDelayEngine(
            x_antennas_gcrs=x_antennas_gcrs,
            enu_origin_gcrs=enu_origin_gcrs,
            enu_coords_gcrs=enu_coords_gcrs
        )


BaseNearFieldDelayEngine.register_pytree()


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
        return mp_policy.cast_to_length(proper_delay), mp_policy.cast_to_length(norm(b_gcrs, axis=-1))

    proper_time1, dist1 = _delay(x_1_gcrs)
    proper_time2, dist2 = _delay(x_2_gcrs)
    return proper_time1 - proper_time2, dist2, dist1


def build_near_field_delay_engine(
        antennas: ac.EarthLocation,
        start_time: at.Time,
        end_time: at.Time,
        ref_time: at.Time,
        resolution: au.Quantity | None = None,
        ref_location: ac.EarthLocation | None = None,
        verbose: bool = False,
) -> BaseNearFieldDelayEngine:
    """
    Build the near field delay engine for a given set of antennas.

    Args:
        antennas: [A] the antennas to compute the delay for.
        start_time: the start time of the delay computation.
        end_time: the end time of the delay computation.
        ref_time: the reference time for the delay computation.
        resolution: the resolution of the delay computation.
        ref_location: the reference location for the delay computation.
        verbose: whether to print verbose output.

    Returns:
        the near field delay engine.
    """
    if not config.jax_enable_x64:
        warnings.warn("jax_enable_x64 is not set, UVW computations may be inaccurate.")

    if ref_location is None:
        ref_location = antennas[0]

    if resolution is None:
        # compute max baseline
        antenna1, antenna2 = np.asarray(list(itertools.combinations(range(len(antennas)), 2))).T
        antennas_itrs = antennas.get_itrs().cartesian.xyz.T
        max_baseline = np.max(np.linalg.norm(antennas_itrs[antenna2] - antennas_itrs[antenna1], axis=-1))
        # Select resolution to keep interpolation error below 1 mm
        if max_baseline <= 10 * au.km:
            resolution = 10 * au.s
        elif max_baseline <= 100 * au.km:
            resolution = 4 * au.s
        elif max_baseline <= 1000 * au.km:
            resolution = 1 * au.s
        else:
            warnings.warn(
                f"Max baseline is {max_baseline} > 1000 km, setting resolution to 0.1 s, "
                f"may lead to slow ephemeris calculations."
            )
            resolution = 0.1 * au.s
        if verbose:
            print(f"Setting resolution to {resolution} suitable for max baseline of {max_baseline}.")

    if not resolution.unit.is_equivalent(au.s):
        raise ValueError(f"resolution must be in seconds got {resolution.unit}")

    if len(antennas.shape) != 1:
        raise ValueError(f"antennas must be 1D got {antennas.shape}")

    if antennas.shape[0] < 2:
        raise ValueError(f"Need at least 2 antennas to form a baseline.")

    bodies_except_earth = ()

    if not start_time.isscalar or not end_time.isscalar:
        raise ValueError(f"start_time and end_time must be scalar got {start_time} and {end_time}")

    ref_time = ref_time.tt
    start_time = start_time.tt
    end_time = end_time.tt

    earth_light_cross_time = 2. * const.R_earth / const.c

    start_grid_time = start_time - earth_light_cross_time
    end_grid_time = end_time + earth_light_cross_time

    num_grid_times = int(np.ceil(float((end_grid_time - start_grid_time) / resolution))) + 1
    num_ants = len(antennas)

    # Define the interpolation grid
    interp_times = start_grid_time + np.arange(num_grid_times) * resolution  # [T]

    if verbose:
        print(f"Computing near field delay")
        print(f"Number of antennas: {len(antennas)}")
        print(f"Between {start_time} and {end_time} ({(end_time - start_time).sec} s)")
        print(f"Interpolation resolution: {resolution}")
        print(f"Number interpolation points: {num_grid_times}")
        print(f"Gravitational effects included from:")
        for body in sorted(bodies_except_earth + ('earth',)):
            print(f"\t{body.title()}")

    # Compute ephemeris'
    ephem_compute_t0 = time_mod.time()

    # Define the antennas
    antennas_gcrs = antennas.reshape((1, num_ants)).get_gcrs(
        obstime=interp_times.reshape((num_grid_times, 1))
    )  # [T, num_ants]
    antennas_position_gcrs = antennas_gcrs.cartesian.xyz  # [3, T, num_ants]

    enu_origin = ref_location.get_gcrs(obstime=interp_times)
    enu_origin_gcrs = enu_origin.cartesian.xyz  # [3, T]

    enu_gcrs = ENU(
        east=np.reshape([1, 0, 0], (1, 3)),
        north=np.reshape([0, 1, 0], (1, 3)),
        up=np.reshape([0, 0, 1], (1, 3)),
        location=ref_location,
        obstime=interp_times.reshape((num_grid_times, 1))  # [T, 3]
    ).transform_to(ac.GCRS(obstime=interp_times.reshape((num_grid_times, 1))))
    enu_coords_gcrs = enu_gcrs.cartesian.xyz  # [3, T, 3]

    ephem_compute_time = time_mod.time() - ephem_compute_t0

    if verbose:
        print(f"Time to compute ephemeris: {ephem_compute_time:.2f} s")

    # Convert to interpolators
    interp_times_jax = jnp.asarray((interp_times.tt - ref_time.tt).sec)  # [T]

    x_antennas_gcrs = quantity_to_jnp(
        np.transpose(antennas_position_gcrs, (1, 2, 0))
    )  # [T, num_ants, 3]
    x_antennas_gcrs = InterpolatedArray(
        x=interp_times_jax,
        values=x_antennas_gcrs,
        axis=0,
        regular_grid=True
    )

    enu_origin_gcrs = quantity_to_jnp(
        np.transpose(enu_origin_gcrs, (1, 0))
    )  # [T, 3]
    enu_origin_gcrs = InterpolatedArray(
        x=interp_times_jax,
        values=enu_origin_gcrs,
        axis=0,
        regular_grid=True
    )

    enu_coords_gcrs = quantity_to_jnp(
        np.transpose(enu_coords_gcrs, (1, 2, 0))
    )  # [T, 3, 3]
    enu_coords_gcrs = InterpolatedArray(
        x=interp_times_jax,
        values=enu_coords_gcrs,
        axis=0,
        regular_grid=True
    )
    return BaseNearFieldDelayEngine(
        x_antennas_gcrs=x_antennas_gcrs,
        enu_origin_gcrs=enu_origin_gcrs,
        enu_coords_gcrs=enu_coords_gcrs
    )
