import dataclasses
from functools import partial
from typing import Tuple, Union

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
from jax import numpy as jnp
from tomographic_kernel.frames import ENU

from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.array_types import FloatArray, IntArray
from dsa2000_cal.delay_models.uvw_utils import perley_icrs_from_lmn, perley_lmn_from_icrs


@dataclasses.dataclass(eq=False)
class BaseGeodesicModel:
    """
    Computes geodesics for each antenna of an array in both near and far field.

    That is, given source location information, e.g. lmn (wrt phase center), or source location in ENU
    (wrt array location), return the lmn coordinates of the sources for each antenna. This can handle zenith
    pointing antennas by setting pointing to None.

    Args:
        ra0: the RA of the phase center in radians.
        dec0: the DEC of the phase center in radians.
        antennas_enu: the ENU positions of the antennas in meters relative to some reference location.
        lmn_zenith: the LMN of the zenith direction at each time.
        lmn_pointings: the LMN of the pointing direction at each time.
        tile_antennas: if True, tile the antennas for each time, else repeat the same antennas for each time.
    """
    ra0: FloatArray  # []
    dec0: FloatArray  # []
    antennas_enu: FloatArray  # [num_ant, 3]
    lmn_zenith: InterpolatedArray  # (t) -> [3]
    lmn_pointings: InterpolatedArray  # (t) -> [[num_ant,] 3]
    tile_antennas: bool
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return
        if len(np.shape(self.ra0)) != 0:
            raise ValueError(f"ra0 must have shape [], got {np.shape(self.ra0)}")
        if len(np.shape(self.dec0)) != 0:
            raise ValueError(f"dec0 must have shape [], got {np.shape(self.dec0)}")
        if len(np.shape(self.antennas_enu)) != 2:
            raise ValueError(f"antennas_enu must have shape [num_ant, 3], got {np.shape(self.antennas_enu)}")
        if self.lmn_zenith.shape != (3,):
            raise ValueError(f"lmn_zenith must have shape [3], got {self.lmn_zenith.shape}")
        if self.tile_antennas:
            if self.lmn_pointings.shape != (3,):
                raise ValueError(f"lmn_pointings must have shape [3], got {self.lmn_pointings.shape}")
        else:
            num_ant = np.shape(self.antennas_enu)[0]
            if self.lmn_pointings.shape != (num_ant, 3):
                raise ValueError(f"lmn_pointings must have shape [num_ant, 3], got {self.lmn_pointings.shape}")

    @property
    def num_antennas(self) -> int:
        return np.shape(self.antennas_enu)[0]

    def compute_elevation_from_lmn(self, lmn: FloatArray, time: FloatArray) -> FloatArray:
        """
        Compute the elevation of the given LMN sources wrt the antenna.

        Args:
            lmn: [..., 3] the LMN of the sources relative to the antenna.
            time: the time in TT, since start of observation.

        Returns:
            elevation: [...] the elevation of the sources wrt the antenna in radians
        """
        # Compute ra/dec of sources
        ra, dec = perley_icrs_from_lmn(lmn[..., 0], lmn[..., 1], lmn[..., 2], self.ra0, self.dec0)
        # compute the zenith ra/dec
        lmn_zenith = self.lmn_zenith(time)  # [3]
        lmn_zenith /= jnp.linalg.norm(lmn_zenith, axis=-1, keepdims=True)
        ra_zenith, dec_zenith = perley_icrs_from_lmn(
            l=lmn_zenith[0],
            m=lmn_zenith[1],
            n=lmn_zenith[2],
            ra0=self.ra0,
            dec0=self.dec0
        )  # []
        _, _, n = perley_lmn_from_icrs(ra, dec, ra_zenith, dec_zenith)
        elevation = mp_policy.cast_to_angle(jnp.arcsin(n))
        return elevation

    def compute_far_field_lmn(self, ra: FloatArray, dec: FloatArray, time: FloatArray | None = None,
                              return_elevation: bool = False
                              ) -> Union[FloatArray, Tuple[FloatArray, FloatArray]]:
        """
        Compute the LMN of the given RA/DEC sources relative to the phase center.
        This is not necessarily relative to pointing centre.

        Args:
            ra: [...] the RA of the sources in radians.
            dec: [...] the DEC of the sources in radians.
            time: the time in TT, since start of observation.
            return_elevation: if True, return the elevation of the source wrt the antenna.

        Returns:
            lmn: [..., 3] the LMN of the sources relative to the phase center.
            if return_elevation, also return elevation: [...]
        """
        if np.shape(ra) != np.shape(dec):
            raise ValueError(f"ra and dec must have the same shape, got {np.shape(ra)} and {np.shape(dec)}")

        l, m, n = perley_lmn_from_icrs(ra, dec, self.ra0, self.dec0)
        lmn = mp_policy.cast_to_angle(jnp.stack([l, m, n], axis=-1))
        if return_elevation:
            if time is None:
                raise ValueError("time must be provided if return_elevation is True")
            lmn_zenith = self.lmn_zenith(time)  # [3]
            lmn_zenith /= jnp.linalg.norm(lmn_zenith, axis=-1, keepdims=True)
            ra_zenith, dec_zenith = perley_icrs_from_lmn(
                l=lmn_zenith[0],
                m=lmn_zenith[1],
                n=lmn_zenith[2],
                ra0=self.ra0,
                dec0=self.dec0
            )  # []
            _, _, n = perley_lmn_from_icrs(ra, dec, ra_zenith, dec_zenith)
            elevation = mp_policy.cast_to_angle(jnp.arcsin(n))
            return lmn, elevation
        return lmn

    def compute_far_field_geodesic(self, times: FloatArray, lmn_sources: FloatArray,
                                   antenna_indices: IntArray | None = None,
                                   return_elevation: bool = False) -> Union[FloatArray, Tuple[FloatArray, IntArray]]:
        """
        Compute the far field geodesic for the given time, as LMN coords relative to pointing center, potentially
        shifting from phase center if different.

        Args:
            times: [num_time] the time in TT, since start of observation.
            lmn_sources: [num_sources, 3], the LMN of sources
            antenna_indices: [num_ant] the indices of antennas to compute for, or None for all.
            return_elevation: if True, return the elevation of the source wrt the antenna.

        Returns:
            geodesic: [num_time, num_ant, num_sources, 3]
            if return_elevation, also return elevation: [num_time, num_ant, num_sources]
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

        if return_elevation:
            out_mapping = "[t,a,s,...],[t,a,s,...]"
        else:
            out_mapping = "[t,a,s,...]"

        @partial(
            multi_vmap,
            in_mapping=f"[t],[a,3],{pointing_mapping},[s,3]",
            out_mapping=out_mapping,
            verbose=True
        )
        def compute_far_field_geodesics(time: jax.Array, antennas_enu, lmn_pointing, lmn_source) -> Union[
            jax.Array, Tuple[jax.Array, jax.Array]]:
            # Note: antennas_enu arg is necessary for multi_vmap to work correctly, even though it's not used.
            # Get RA/DEC of pointings wrt phase centre
            ra0, dec0 = self.ra0, self.dec0
            # jax.debug.print("ra0={ra0}, dec0={dec0}",ra0=ra0, dec0=dec0)
            # jax.debug.print("lmn_pointing={lmn_pointing}",lmn_pointing=lmn_pointing)
            ra_pointing, dec_pointing = perley_icrs_from_lmn(
                l=lmn_pointing[0],
                m=lmn_pointing[1],
                n=lmn_pointing[2],
                ra0=ra0,
                dec0=dec0
            )  # [[num_ant]]
            # jax.debug.print("ra_pointing={ra_pointing}, dec_pointing={dec_pointing}",ra_pointing=ra_pointing, dec_pointing=dec_pointing)
            ra, dec = perley_icrs_from_lmn(
                l=lmn_source[0],
                m=lmn_source[1],
                n=lmn_source[2],
                ra0=ra0,
                dec0=dec0
            )  # []
            # jax.debug.print("ra={ra}, dec={dec}",ra=ra, dec=dec)
            l, m, n = perley_lmn_from_icrs(ra, dec, ra_pointing, dec_pointing)  # []
            # jax.debug.print("l={l}, m={m}, n={n}",l=l, m=m, n=n)
            lmn = mp_policy.cast_to_angle(jnp.stack([l, m, n], axis=-1))  # [3]
            if return_elevation:
                lmn_zenith = self.lmn_zenith(time)  # [3]
                lmn_zenith /= jnp.linalg.norm(lmn_zenith, axis=-1, keepdims=True)
                ra_zenith, dec_zenith = perley_icrs_from_lmn(
                    l=lmn_zenith[0],
                    m=lmn_zenith[1],
                    n=lmn_zenith[2],
                    ra0=ra0,
                    dec0=dec0
                )  # []
                l, m, n = perley_lmn_from_icrs(ra, dec, ra_zenith, dec_zenith)
                elevation = mp_policy.cast_to_angle(jnp.arcsin(n))
                return lmn, elevation
            return lmn

        return compute_far_field_geodesics(times, antennas_enu, lmn_pointing,
                                           lmn_sources)  # [num_time, num_ant, num_sources, 3]

    def compute_near_field_geodesics(self, times: FloatArray, source_positions_enu: FloatArray,
                                     antenna_indices: IntArray | None = None,
                                     return_elevation: bool = False) -> Union[FloatArray, Tuple[FloatArray, IntArray]]:
        """
        Compute the near field geodesic for the given time, as LMN coords relative to pointing center, potentially
        shifting from phase center if different.

        Args:
            times: [num_time] the time in TT, since start of observation.
            source_positions_enu: [num_sources, 3], the ENU positions of the sources.
            antenna_indices: [num_ant] the indices of antennas to compute for, or None for all.
            return_elevation: if True, return the elevation of the source wrt the antenna.

        Returns:
            geodesic: [num_time, num_ant, num_sources, 3]
            if return_elevation, also return elevation: [num_time, num_ant, num_sources]
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

        if return_elevation:
            out_mapping = "[t,a,s,...],[t,a,s,...]"
        else:
            out_mapping = "[t,a,s,...]"

        @partial(
            multi_vmap,
            in_mapping=f"[t],{pointing_mapping},[a,3],[s,3]",
            out_mapping=out_mapping,
            verbose=True
        )
        def compute_near_field_geodesics(time: jax.Array, lmn_pointing: jax.Array, antennas_enu: jax.Array,
                                         source_position_enu: jax.Array) -> Union[
            jax.Array, Tuple[jax.Array, jax.Array]]:
            # Get the directions of sources from each antenna to each source
            direction_enu = source_position_enu - antennas_enu  # [3]
            norm = jnp.linalg.norm(direction_enu, axis=-1, keepdims=True)
            direction_enu = jnp.where(
                norm == 0.,
                jnp.asarray([0., 0., 1.], direction_enu.dtype),
                direction_enu / norm
            )  # normalise for correctness, give ontop source arbitrary direction up-hat

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
            lmn = mp_policy.cast_to_angle(jnp.stack([l, m, n], axis=-1))

            if return_elevation:
                elevation = mp_policy.cast_to_angle(jnp.arcsin(direction_enu[2]))  # [3]
                return lmn, elevation

            return lmn

        return compute_near_field_geodesics(
            times, lmn_pointing, antennas_enu, source_positions_enu
        )  # [num_time, num_ant, num_sources, 3]


def base_geodesic_model_flatten(base_geodesic_model: BaseGeodesicModel):
    return (
        [
            base_geodesic_model.ra0,
            base_geodesic_model.dec0,
            base_geodesic_model.antennas_enu,
            base_geodesic_model.lmn_zenith,
            base_geodesic_model.lmn_pointings
        ], (
            base_geodesic_model.tile_antennas,
        )
    )


def base_geodesic_model_unflatten(aux_data, children):
    ra0, dec0, antennas_enu, lmn_zenith, lmn_pointings = children
    (tile_antennas,) = aux_data
    return BaseGeodesicModel(ra0, dec0, antennas_enu, lmn_zenith, lmn_pointings, tile_antennas=tile_antennas,
                             skip_post_init=True)


jax.tree_util.register_pytree_node(
    BaseGeodesicModel,
    base_geodesic_model_flatten,
    base_geodesic_model_unflatten
)


def build_geodesic_model(
        antennas: ac.EarthLocation,
        array_location: ac.EarthLocation,
        phase_center: ac.ICRS,
        obstimes: at.Time,  # [num_model_times] over which to compute the zenith
        ref_time: at.Time,
        pointings: ac.ICRS | None,  # [[num_ant]] or None which means Zenith
) -> BaseGeodesicModel:
    """
    Build a base geodesic model for the given antennas, array location, phase center, and times.

    Args:
        antennas: [num_ant] the antennas
        array_location: the location of the array
        phase_center: the phase center
        obstimes: [num_model_times] the times over which to compute the zenith
        ref_time: the reference time
        pointings: [[num_ant]] the pointings of the antennas, or None for zenith

    Returns:
        The base geodesic model.
    """
    enu_frame = ENU(location=array_location, obstime=ref_time)
    ra0, dec0 = quantity_to_jnp(phase_center.ra), quantity_to_jnp(phase_center.dec)
    antennas_enu = quantity_to_jnp(
        antennas.get_itrs(
            obstime=ref_time,
            location=array_location
        ).transform_to(enu_frame).cartesian.xyz.T
    )  # [num_ant, 3]

    zenith_lmn = quantity_to_jnp(
        icrs_to_lmn(
            sources=ENU(east=0, north=0, up=1, location=array_location,
                        obstime=obstimes).transform_to(
                ac.ICRS()),
            phase_tracking=phase_center
        )
    )  # [num_model_times, 3]
    regular_grid = True  # is_regular_grid(quantity_to_np((obstimes.tt - ref_time.tt).sec * au.s))
    obstimes = quantity_to_jnp((obstimes.tt - ref_time.tt).sec * au.s)
    lmn_zenith = InterpolatedArray(
        x=obstimes,
        values=zenith_lmn,
        axis=0,
        regular_grid=regular_grid
    )  # [num_model_times, 3]
    if pointings is None:
        lmn_pointings = lmn_zenith
    else:
        pointing_lmn = quantity_to_jnp(
            icrs_to_lmn(
                pointings,
                phase_center
            )
        )  # [[num_ant,] 3]
        lmn_pointings = InterpolatedArray(
            x=jnp.stack([obstimes[0], obstimes[-1] + 1e-6]),  # Add 1e-6 to avoid interpolation errors
            values=jnp.stack([pointing_lmn, pointing_lmn], axis=0),
            axis=0,
            regular_grid=regular_grid
        )

    tile_antennas = len(lmn_pointings.shape) == 1

    return BaseGeodesicModel(
        ra0=ra0,
        dec0=dec0,
        antennas_enu=antennas_enu,
        lmn_zenith=lmn_zenith,
        lmn_pointings=lmn_pointings,
        tile_antennas=tile_antennas
    )
