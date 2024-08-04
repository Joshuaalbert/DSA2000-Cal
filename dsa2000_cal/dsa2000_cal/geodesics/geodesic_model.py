import dataclasses
from functools import partial

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
from jax import numpy as jnp
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.interp_utils import InterpolatedArray, is_regular_grid
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np
from dsa2000_cal.delay_models.uvw_utils import perley_icrs_from_lmn, perley_lmn_from_icrs


@dataclasses.dataclass(eq=False)
class GeodesicModel:
    antennas: ac.EarthLocation
    array_location: ac.EarthLocation
    phase_center: ac.ICRS
    obstimes: at.Time  # [num_model_times] over which to compute the zenith
    ref_time: at.Time
    pointings: ac.ICRS | None  # [[num_ant]] or None which means Zenith

    def __post_init__(self):
        enu_frame = ENU(location=self.array_location, obstime=self.ref_time)
        self.ra0, self.dec0 = quantity_to_jnp(self.phase_center.ra), quantity_to_jnp(self.phase_center.dec)
        self.antennas_enu = quantity_to_jnp(
            self.antennas.get_itrs(
                obstime=self.ref_time,
                location=self.array_location
            ).transform_to(enu_frame).cartesian.xyz.T
        )  # [num_ant, 3]

        zenith_lmn = quantity_to_jnp(
            icrs_to_lmn(
                sources=ENU(east=0, north=0, up=1, location=self.array_location, obstime=self.obstimes).transform_to(
                    ac.ICRS()),
                phase_tracking=self.phase_center
            )
        )  # [num_model_times, 3]
        regular_grid = is_regular_grid(quantity_to_np((self.obstimes.tt - self.ref_time.tt).sec * au.s))
        obstimes = quantity_to_jnp((self.obstimes.tt - self.ref_time.tt).sec * au.s)
        self.lmn_zenith = InterpolatedArray(
            x=obstimes,
            values=zenith_lmn,
            axis=0,
            regular_grid=regular_grid
        )  # [num_model_times, 3]
        if self.pointings is None:
            self.lmn_pointings = self.lmn_zenith
        else:
            pointing_lmn = quantity_to_jnp(
                icrs_to_lmn(
                    self.pointings,
                    self.phase_center
                )
            )  # [[num_ant,] 3]
            self.lmn_pointings = InterpolatedArray(
                x=jnp.stack([obstimes[0], obstimes[-1]]),
                values=jnp.stack([pointing_lmn, pointing_lmn], axis=0),
                axis=0,
                regular_grid=regular_grid
            )

        self.tile_antennas = len(self.lmn_pointings.shape) == 1

    def time_to_jnp(self, times: at.Time) -> jax.Array:
        """
        Convert the times to jax.Array, in TT scale since ref time.

        Args:
            times: [num_time] the times

        Returns:
            times: [num_time]
        """
        return quantity_to_jnp((times.tt - self.ref_time.tt).sec * au.s)

    def compute_far_field_geodesic(self, times: jax.Array, lmn_sources: jax.Array,
                                   antenna_indices: jax.Array | None = None) -> jax.Array:
        """
        Compute the far field geodesic for the given time, as LMN coords relative to pointing center, potentially
        shifting from phase center if different.

        Args:
            times: [num_time] the time in TT, since start of observation.
            lmn_sources: [num_sources, 3], the LMN of sources
            antenna_indices: [num_ant] the indices of antennas to compute for, or None for all.

        Returns:
            geodesic: [num_sources, num_time, num_ant, 3]
        """
        if len(np.shape(times)) != 1:
            raise ValueError(f"times must have shape [num_time], got {np.shape(times)}")
        if len(np.shape(lmn_sources)) != 2:
            raise ValueError(f"lmn_sources must have shape [num_sources, 3], got {np.shape(lmn_sources)}")
        if antenna_indices is not None:
            if len(np.shape(antenna_indices)) != 1:
                raise ValueError(f"antenna_indices must have shape [num_ant], got {np.shape(antenna_indices)}")

        lmn_pointing = self.lmn_pointings(times)  # [num_time, [num_ant',] 3]
        if self.tile_antennas:
            pointing_mapping = "[t,3]"
        else:
            pointing_mapping = "[t,a,3]"
            if antenna_indices is not None:
                lmn_pointing = lmn_pointing[:, antenna_indices, :]  # [num_time, num_ant, 3]

        lmn_pointing /= jnp.linalg.norm(lmn_pointing, axis=-1, keepdims=True)  # normalise for correctness

        antennas_enu = self.antennas_enu  # [num_ant', 3]
        if antenna_indices is not None:
            antennas_enu = antennas_enu[antenna_indices]  # [num_ant, 3]

        @partial(
            multi_vmap,
            in_mapping=f"[a,3],{pointing_mapping},[s,3]",
            out_mapping="[s,t,a,...]",
            verbose=True
        )
        def create_geodesics(antennas_enu, lmn_pointing, lmn_source):
            # Note: antennas_enu arg is necessary for multi_vmap to work correctly, even though it's not used.
            # Get RA/DEC of pointings wrt phase centre
            ra0, dec0 = self.ra0, self.dec0
            ra_pointing, dec_pointing = perley_icrs_from_lmn(
                l=lmn_pointing[0],
                m=lmn_pointing[1],
                n=lmn_pointing[2],
                ra0=ra0,
                dec0=dec0
            )  # [[num_ant]]
            ra, dec = perley_icrs_from_lmn(
                l=lmn_source[0],
                m=lmn_source[1],
                n=lmn_source[2],
                ra0=ra0,
                dec0=dec0
            )  # []
            l, m, n = perley_lmn_from_icrs(ra, dec, ra_pointing, dec_pointing)  # []
            return jnp.stack([l, m, n], axis=-1)  # [3]

        return create_geodesics(antennas_enu, lmn_pointing, lmn_sources)  # [num_sources, num_time, num_ant, 3]

    def compute_near_field_geodesics(self, times: jax.Array, source_positions_enu: jax.Array,
                                     antenna_indices: jax.Array | None = None) -> jax.Array:
        """
        Compute the near field geodesic for the given time, as LMN coords relative to pointing center, potentially
        shifting from phase center if different.

        Args:
            times: [num_time] the time in TT, since start of observation.
            source_positions_enu: [num_sources, 3], the ENU positions of the sources.
            antenna_indices: [num_ant] the indices of antennas to compute for, or None for all.

        Returns:
            geodesic: [num_sources, num_time, num_ant, 3]
        """
        if len(np.shape(times)) != 1:
            raise ValueError(f"times must have shape [num_time], got {np.shape(times)}")
        if len(np.shape(source_positions_enu)) != 2:
            raise ValueError(
                f"source_positions_enu must have shape [num_sources, 3], got {np.shape(source_positions_enu)}"
            )
        if antenna_indices is not None:
            if len(np.shape(antenna_indices)) != 1:
                raise ValueError(f"antenna_indices must have shape [num_ant], got {np.shape(antenna_indices)}")

        lmn_pointing = self.lmn_pointings(times)  # [num_tims, [num_ant',] 3]
        if self.tile_antennas:
            pointing_mapping = "[t,3]"
        else:
            pointing_mapping = "[t,a,3]"
            if antenna_indices is not None:
                lmn_pointing = lmn_pointing[:, antenna_indices, :]  # [num_time, num_ant, 3]

        lmn_pointing /= jnp.linalg.norm(lmn_pointing, axis=-1, keepdims=True)  # normalise for correctness

        antennas_enu = self.antennas_enu  # [num_ant', 3]
        if antenna_indices is not None:
            antennas_enu = antennas_enu[antenna_indices]  # [num_ant, 3]

        @partial(
            multi_vmap,
            in_mapping=f"[t],{pointing_mapping},[a,3],[s,3]",
            out_mapping="[s,t,a,...]",
            verbose=True
        )
        def create_geodesics(time: jax.Array, lmn_pointing: jax.Array, antennas_enu: jax.Array,
                             source_position_enu: jax.Array) -> jax.Array:
            # Get the directions of sources from each antenna to each source
            direction_enu = source_position_enu - antennas_enu  # [3]
            direction_enu /= jnp.linalg.norm(direction_enu, axis=-1, keepdims=True)  # normalise for correctness

            ra0, dec0 = self.ra0, self.dec0
            lmn_zenith = self.lmn_zenith(time)  # [3]
            lmn_zenith /= jnp.linalg.norm(lmn_zenith, axis=-1, keepdims=True)
            ra_zenith, dec_zenith = perley_icrs_from_lmn(
                l=lmn_zenith[0],
                m=lmn_zenith[1],
                n=lmn_zenith[2],
                ra0=ra0,
                dec0=dec0
            )  # []

            ra, dec = perley_icrs_from_lmn(
                l=direction_enu[0],
                m=direction_enu[1],
                n=direction_enu[2],
                ra0=ra_zenith,
                dec0=dec_zenith
            )

            ra_pointing, dec_pointing = perley_icrs_from_lmn(
                l=lmn_pointing[0],
                m=lmn_pointing[1],
                n=lmn_pointing[2],
                ra0=ra0,
                dec0=dec0
            )  # []

            l, m, n = perley_lmn_from_icrs(ra, dec, ra_pointing, dec_pointing)  # []

            return jnp.stack([l, m, n], axis=-1)

        return create_geodesics(times, lmn_pointing, antennas_enu,
                                source_positions_enu)  # [num_sources, num_time, num_ant, 3]
