import dataclasses
import itertools
import os.path
import warnings
from functools import cached_property, partial
from typing import Literal, List, Union, Annotated, NamedTuple, Generator, Tuple, Optional

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import tables as tb
from pydantic import Field

from dsa2000_cal.common.coord_utils import earth_location_to_uvw
from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel


class MeasurementSetMetaV0(SerialisableBaseModel):
    """
    A class to store metadata about a measurement set, based on the meta of CASA's MS v2.
    We use astropy quantities to store the values of the meta, as they are more robust than floats with assumed units.
    We will reference array data stored in hdf5 format using pytable's.

    We will assume a single pointing for the entire measurement set.
    We will assume single scan for the entire measurement set.
    We will assume single spectral window for the entire measurement set.
    """
    version: int = Field(
        default=0,
        description="Version of the meta file."
    )

    array_name: str = Field(
        description="Name of the array."
    )

    array_location: ac.EarthLocation = Field(
        description="Location of the array, from which UVW frame is defined."
    )
    phase_tracking: ac.ICRS = Field(
        description="Phase tracking direction, against which LMN coordinates are defined."
    )
    channel_width: au.Quantity = Field(
        description="Channel width."
    )
    integration_time: au.Quantity = Field(
        description="Integration time."
    )
    coherencies: Literal['stokes', 'linear', 'circular'] = Field(
        description="Coherency type."
    )

    pointings: ac.ICRS = Field(
        description="Pointing direction of each of the antennas."
    )  # [num_antenna]
    times: at.Time = Field(
        description="Centre times of data windows."
    )  # [num_times]
    freqs: au.Quantity = Field(
        description="Centre frequencies of data windows."
    )  # [num_freqs]
    antennas: ac.EarthLocation = Field(
        description="Antenna positions."
    )  # [num_antenna]
    antenna_names: List[str] = Field(
        description="Antenna names."
    )  # [num_antenna]
    antenna_diameters: au.Quantity = Field(
        description="Antenna diameters."
    )  # [num_antenna]

    with_autocorr: bool = Field(
        default=True,
        description="Whether to include autocorrelations."
    )


def _check_measurement_set_meta_v0(meta: MeasurementSetMetaV0):
    num_antennas = len(meta.antennas)
    num_times = len(meta.times)
    num_freqs = len(meta.freqs)

    if not meta.array_location.isscalar:
        raise ValueError(f"Expected a scalar EarthLocation, got {meta.array_location}")
    if not meta.phase_tracking.isscalar:
        raise ValueError(f"Expected a scalar ICRS, got {meta.phase_tracking}")
    if not meta.channel_width.isscalar:
        raise ValueError(f"Expected a scalar Quantity, got {meta.channel_width}")
    if not meta.integration_time.isscalar:
        raise ValueError(f"Expected a scalar Quantity, got {meta.integration_time}")
    if meta.pointings.isscalar:
        warnings.warn(f"Expected a vector ICRS, got {meta.pointings.shape}, assuming same pointing for all antennas.")
        meta.pointings = ac.ICRS(
            np.repeat(meta.pointings.ra.deg, num_antennas) * au.deg,
            np.repeat(meta.pointings.dec.deg, num_antennas) * au.deg
        )
    if meta.times.isscalar:
        raise ValueError(f"Expected a vector Time, got {meta.times}")
    if meta.freqs.isscalar:
        raise ValueError(f"Expected a vector Quantity, got {meta.freqs}")
    if meta.antennas.isscalar:
        raise ValueError(f"Expected a vector EarthLocation, got {meta.antennas}")
    if meta.antenna_diameters.isscalar:
        warnings.warn(f"Expected antenna_diameters to be a vector, got {meta.antenna_diameters}")
        meta.antenna_diameters = np.repeat(meta.antenna_diameters.value, num_antennas) * meta.antenna_diameters.unit

    if not meta.channel_width.unit.is_equivalent(au.Hz):
        raise ValueError(f"Expected channel width in Hz, got {meta.channel_width}")
    if not meta.integration_time.unit.is_equivalent(au.s):
        raise ValueError(f"Expected integration time in seconds, got {meta.integration_time}")
    if not meta.freqs.unit.is_equivalent(au.Hz):
        raise ValueError(f"Expected frequencies in Hz, got {meta.freqs}")
    if not meta.antenna_diameters.unit.is_equivalent(au.m):
        raise ValueError(f"Expected antenna diameters in meters, got {meta.antenna_diameters}")

    if meta.pointings.shape != (num_antennas,):
        raise ValueError(f"Expected pointings to have shape ({num_antennas},), got {meta.pointings.shape}")
    if len(meta.antenna_names) != num_antennas:
        raise ValueError(f"Expected antenna_names to have length {num_antennas}, got {len(meta.antenna_names)}")
    if meta.antenna_diameters.shape != (num_antennas,):
        raise ValueError(
            f"Expected antenna_diameters to have shape ({num_antennas},), got {meta.antenna_diameters.shape}")


MeasurementSetMeta = Annotated[Union[MeasurementSetMetaV0], Field(discriminator='version')]


def _check_measurement_set_meta(meta: MeasurementSetMeta):
    if meta.version == 0:
        _check_measurement_set_meta_v0(meta)
    else:
        raise ValueError(f"Unknown version {meta.version}.")


def _combination_with_replacement_index(i, j, n):
    # Starting index for the ith row
    k_i = i * (2 * n - i + 1) // 2
    # Index for the pair is the starting index of the row plus the offset in the row
    index = k_i + (j - i)
    return index


def _combination_index(i, j, n):
    return (i * (2 * n - i - 1)) // 2 + j - i - 1


class NotContiguous(Exception):
    pass


def _get_slice(indices):
    steps = np.diff(indices)
    if not np.all(steps == steps[0]):
        raise NotContiguous("Indices must be contiguous.")
    step = steps[0]
    return slice(indices[0], indices[-1] + 1, step)


def _try_get_slice(indices):
    try:
        return _get_slice(indices)
    except NotContiguous:
        return indices


class VisibilityCoords(NamedTuple):
    """
    Coordinates for a single visibility.
    """
    uvw: jax.Array | np.ndarray  # [rows, 3] the uvw coordinates
    time_obs: jax.Array | np.ndarray  # [rows] the time relative to the reference time (observation start)
    antenna_1: jax.Array | np.ndarray  # [rows] the first antenna
    antenna_2: jax.Array | np.ndarray  # [rows] the second antenna
    time_idx: jax.Array | np.ndarray  # [rows] the time index


class VisibilityData(NamedTuple):
    """
    Data for a single visibility.
    """
    vis: Optional[jax.Array | np.ndarray] = None  # [rows, num_freqs, 4] the visibility data
    weights: Optional[jax.Array | np.ndarray] = None  # [rows, num_freqs, 4] the weights
    flags: Optional[jax.Array | np.ndarray] = None  # [rows, num_freqs, 4] the flags


@dataclasses.dataclass(eq=False)
class MeasurementSet:
    ms_folder: str

    def __post_init__(self):
        self.meta_file = os.path.join(self.ms_folder, "meta.json")
        self.data_file = os.path.join(self.ms_folder, "data.h5")

        if not os.path.exists(self.meta_file):
            raise ValueError(f"Meta file {self.meta_file} does not exist.")

        if not os.path.exists(self.data_file):
            raise ValueError(f"Data file {self.data_file} does not exist.")

        self.meta = MeasurementSetMeta.parse_file(self.meta_file)
        _check_measurement_set_meta(self.meta)

    def get_rows(self, antenna_1: np.ndarray | int, antenna_2: np.ndarray | int,
                 time_idx: np.ndarray | int) -> np.ndarray:
        """
        Get the row index for the given antenna pair and time index.
        """
        if self.meta.with_autocorr:
            get_antenna_index = partial(_combination_with_replacement_index, n=len(self.meta.antennas))
        else:
            get_antenna_index = partial(_combination_index, n=len(self.meta.antennas))

        time_offset = time_idx * self.block_size

        return get_antenna_index(antenna_1, antenna_2) + time_offset

    @cached_property
    def num_rows(self) -> int:
        """
        Get the number of rows in the measurement set.
        """
        with tb.open_file(self.data_file) as f:
            return f.root.uvw.shape[0]

    @cached_property
    def block_size(self) -> int:
        """
        Get the number of rows in the measurement set.
        """
        num_antennas = len(self.meta.antennas)
        if self.meta.with_autocorr:
            return (num_antennas * (num_antennas + 1)) // 2
        return (num_antennas * (num_antennas - 1)) // 2

    @cached_property
    def ref_time(self) -> at.Time:
        """
        Get the reference time of the measurement set.
        """
        return self.meta.times[0]

    @staticmethod
    def create_measurement_set(ms_folder: str, meta: MeasurementSetMeta) -> 'MeasurementSet':
        """
        Create a measurement set from the given meta.
        """
        _check_measurement_set_meta(meta)

        os.makedirs(ms_folder, exist_ok=True)

        meta_file = os.path.join(ms_folder, "meta.json")
        with open(meta_file, "w") as f:
            f.write(meta.json(indent=2))

        data_file = os.path.join(ms_folder, "data.h5")

        # Data are:
        # uvw: [num_rows, 3]
        # antenna_1: [num_rows]
        # antenna_2: [num_rows]
        # time_idx: [num_rows]
        # vis: [num_rows, num_freqs, 4]
        # weight: [num_rows, num_freqs, 4]
        # flags: [num_rows, num_freqs, 4]

        num_antennas = len(meta.antennas)
        num_times = len(meta.times)
        num_freqs = len(meta.freqs)
        if meta.with_autocorr:
            block_size = (num_antennas * (num_antennas + 1)) // 2
        else:
            block_size = (num_antennas * (num_antennas - 1)) // 2
        num_rows = block_size * num_times
        with tb.open_file(data_file, "w") as f:
            f.create_array("/", "uvw", atom=tb.Float32Atom(), shape=(num_rows, 3))
            f.create_array("/", "antenna_1", atom=tb.Int16Atom(), shape=(num_rows,))
            f.create_array("/", "antenna_2", atom=tb.Int16Atom(), shape=(num_rows,))
            f.create_array("/", "time_idx", atom=tb.Int16Atom(), shape=(num_rows,))
            f.create_array("/", "vis", atom=tb.ComplexAtom(itemsize=8),
                           shape=(num_rows, num_freqs, 4))  # single precision complex
            f.create_array("/", "weights", atom=tb.Float16Atom(), shape=(num_rows, num_freqs, 4))
            f.create_array("/", "flags", atom=tb.BoolAtom(), shape=(num_rows, num_freqs, 4))

        start_row = 0
        for time_idx in range(num_times):
            # UVW are position(antenna_2) - position(antenna_1)
            # antenna_1, antenna_2 are all possible baselines
            time = meta.times[time_idx]

            if meta.with_autocorr:
                baseline_pairs = np.asarray(list(itertools.combinations_with_replacement(range(num_antennas), 2)),
                                            dtype=np.int32)
            else:
                baseline_pairs = np.asarray(list(itertools.combinations(range(num_antennas), 2)),
                                            dtype=np.int32)
            antenna_1 = baseline_pairs[:, 0]
            antenna_2 = baseline_pairs[:, 1]

            time_idx = np.full(antenna_1.shape, time_idx, dtype=np.int32)

            antennas_uvw = earth_location_to_uvw(
                antennas=meta.antennas,
                array_location=meta.array_location,
                time=time,
                phase_tracking=meta.phase_tracking
            )

            uvw = antennas_uvw[antenna_2] - antennas_uvw[antenna_1]

            end_row = start_row + uvw.shape[0]

            with tb.open_file(data_file, "r+") as f:
                f.root.uvw[start_row:end_row] = uvw
                f.root.antenna_1[start_row:end_row] = antenna_1
                f.root.antenna_2[start_row:end_row] = antenna_2
                f.root.time_idx[start_row:end_row] = time_idx
                f.root.vis[start_row:end_row] = 0.
                f.root.weights[start_row:end_row] = 1.
                f.root.flags[start_row:end_row] = False
            start_row += uvw.shape[0]

        return MeasurementSet(ms_folder=ms_folder)

    def match(self, antenna_1: np.ndarray | int, antenna_2: np.ndarray | int, times: at.Time,
              freqs: au.Quantity | None = None) -> VisibilityData:
        """
        Get the visibility data for the given antenna pair, times and frequencies. The shapes of inputs must broadcast.
        I.e. scalars will broadcast.

        Args:
            antenna_1: the first antenna
            antenna_2: the second antenna
            times: the times
            freqs: the frequencies, if None, all frequencies are returned

        Returns:
            the visibility data matching the given antenna pair, times and frequencies.
        """

        (i0_time, alpha0_time), (i1_time, alpha1_time) = get_interp_indices_and_weights(
            x=(times - self.ref_time).sec, xp=(self.meta.times - self.ref_time).sec
        )
        rows0 = self.get_rows(antenna_1=antenna_1, antenna_2=antenna_2, time_idx=i0_time)
        rows1 = self.get_rows(antenna_1=antenna_1, antenna_2=antenna_2, time_idx=i1_time)
        # For accessing HDF5 slices are faster
        rows0 = _try_get_slice(rows0)
        rows1 = _try_get_slice(rows1)

        def _access_non_unique(h5_array, rows):
            if isinstance(rows, slice):
                return h5_array[rows]
            unique_rows, inverse_map = np.unique(rows, return_inverse=True)
            unique_get = h5_array[unique_rows, ...]
            if len(unique_rows) == len(rows):
                return unique_get
            # Send back to original shape
            return unique_get[inverse_map]

        with tb.open_file(self.data_file, 'r') as f:
            vis = (
                    _access_non_unique(f.root.vis, rows0) * alpha0_time[:, None, None]
                    + _access_non_unique(f.root.vis, rows1) * alpha1_time[:, None, None]
            )
            weights = (
                    _access_non_unique(f.root.weights, rows0) * alpha0_time[:, None, None]
                    + _access_non_unique(f.root.weights, rows1) * alpha1_time[:, None, None]
            )
            flags = (
                    _access_non_unique(f.root.flags, rows0) * alpha0_time[:, None, None]
                    + _access_non_unique(f.root.flags, rows1) * alpha1_time[:, None, None]
            )

        if freqs is not None:
            (i0_freq, alpha0_freq), (i1_freq, alpha1_freq) = get_interp_indices_and_weights(
                x=freqs.value, xp=self.meta.freqs.value
            )
            i0_freq = _try_get_slice(i0_freq)
            i1_freq = _try_get_slice(i1_freq)
            vis = vis[:, i0_freq] * alpha0_freq + vis[:, i1_freq] * alpha1_freq
            weights = weights[:, i0_freq] * alpha0_freq + weights[:, i1_freq] * alpha1_freq
            flags = flags[:, i0_freq] * alpha0_freq + flags[:, i1_freq] * alpha1_freq

        # Cast flags to bool, will effective to OR operation
        flags = flags.astype(np.bool_)

        return VisibilityData(
            vis=vis,
            weights=weights,
            flags=flags
        )

    def create_block_generator(self, start_time_idx: int = 0, end_time_idx: int | None = None,
                               vis: bool = True, weights: bool = True, flags: bool = True) -> Generator[
        Tuple[at.Time, VisibilityCoords, VisibilityData], VisibilityData | None, None
    ]:
        if self.meta.with_autocorr:
            start_antenna_1 = 0
            start_antenna_2 = 0
        else:
            start_antenna_1 = 0
            start_antenna_2 = 1

        start_row = self.get_rows(start_antenna_1, start_antenna_2, start_time_idx)

        if end_time_idx is None:
            end_row = self.num_rows
        else:
            end_row = self.get_rows(start_antenna_1, start_antenna_2, end_time_idx) + self.block_size

        time_obs = jnp.asarray((self.meta.times - self.ref_time).sec, dtype=jnp.float32)

        with tb.open_file(self.data_file, 'r+') as f:
            time_idx = start_time_idx
            for row in range(start_row, end_row, self.block_size):
                time = self.meta.times[time_idx]
                time_idx += 1
                coords = VisibilityCoords(
                    uvw=f.root.uvw[row:row + self.block_size],
                    time_obs=time_obs[f.root.time_idx[row:row + self.block_size]],
                    antenna_1=f.root.antenna_1[row:row + self.block_size],
                    antenna_2=f.root.antenna_2[row:row + self.block_size],
                    time_idx=f.root.time_idx[row:row + self.block_size]
                )
                data = VisibilityData(
                    vis=f.root.vis[row:row + self.block_size] if vis else None,
                    weights=f.root.weights[row:row + self.block_size] if weights else None,
                    flags=f.root.flags[row:row + self.block_size] if flags else None
                )
                response = yield (time, coords, data)
                if response is not None and isinstance(response, VisibilityData):
                    if response.vis is not None:
                        f.root.vis[row:row + self.block_size] = response.vis
                    if response.weights is not None:
                        f.root.weights[row:row + self.block_size] = response.weights
                    if response.flags is not None:
                        f.root.flags[row:row + self.block_size] = response.flags
