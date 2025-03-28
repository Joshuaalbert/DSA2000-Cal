import dataclasses
import itertools
import os.path
import shutil
import warnings
from functools import cached_property, partial
from typing import Literal, List, NamedTuple, Generator, Tuple, Optional

import astropy.coordinates as ac
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import tables as tb
from astropy import time as at
from pydantic import Field

from dsa2000_common.common.corr_utils import broadcast_translate_corrs
from dsa2000_common.common.array_types import ComplexArray, FloatArray, BoolArray
from dsa2000_common.common.interp_utils import get_interp_indices_and_weights, get_centred_insert_index
from dsa2000_common.common.mixed_precision_utils import mp_policy
from dsa2000_common.common.quantity_utils import time_to_jnp, quantity_to_jnp
from dsa2000_common.common.serialise_utils import SerialisableBaseModel
from dsa2000_common.common.types import VisibilityCoords
from dsa2000_common.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine

from dsa2000_common.delay_models.base_far_field_delay_engine import build_far_field_delay_engine


from dsa2000_common.delay_models.base_near_field_delay_engine import build_near_field_delay_engine


from dsa2000_common.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel


from dsa2000_common.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_common.gain_models.gain_model import GainModel

from dsa2000_common.geodesics.base_geodesic_model import build_geodesic_model, BaseGeodesicModel


class MeasurementSetMeta(SerialisableBaseModel):
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
        description="Name of the array. Assumes all antennas are from the same array."
    )

    array_location: ac.EarthLocation = Field(
        description="Location of the array, from which UVW frame is defined."
    )
    phase_center: ac.ICRS = Field(
        description="Phase tracking direction, against which UVW coordinates are defined."
    )
    channel_width: au.Quantity = Field(
        description="Channel width."
    )
    integration_time: au.Quantity = Field(
        description="Integration time."
    )
    coherencies: Tuple[str, ...] = Field(
        description="Coherency type."
    )

    pointings: ac.ICRS | None = Field(
        description="Pointing direction of each of the antennas."
    )  # [num_antenna]
    static_beam: bool = Field(
        default=True,
        description="If True the beam response does not change over time, otherwise it's a function of time and pointing."
    )
    times: at.Time = Field(
        description="Centre times of data windows."
    )  # [num_times]
    ref_time: at.Time = Field(
        description="Reference time of the measurement set."
    )
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
    mount_types: List[
                     Literal['EQUATORIAL', 'ALT-AZ', 'X-Y', 'SPACE-HALCA']
                 ] | Literal['EQUATORIAL', 'ALT-AZ', 'X-Y', 'SPACE-HALCA'] = Field(
        description="Mount types."
    )  # [num_antenna]

    with_autocorr: bool = Field(
        default=True,
        description="Whether to include autocorrelations."
    )

    system_equivalent_flux_density: au.Quantity | None = Field(
        default=None,
        description="System equivalent flux density."
    )

    convention: Literal['physical', 'engineering'] = Field(
        default='physical',
        description="Convention of the data, 'physical' means uvw are computed for antennas_2 - antenna1. "
                    "The RIME model should use the same convention to model visibilities. "
                    "Using 'engineering' means uvw are computed for antenna1 - antenna2."
    )

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(MeasurementSetMeta, self).__init__(**data)
        # Use _check_measurement_set_meta_v0 as instance-wise validator
        _check_measurement_set_meta_v0(self)


def _check_measurement_set_meta_v0(meta: MeasurementSetMeta):
    num_antennas = len(meta.antennas)
    num_times = len(meta.times)
    num_freqs = len(meta.freqs)

    if not meta.array_location.isscalar:
        raise ValueError(f"Expected a scalar EarthLocation, got {meta.array_location}")
    if not meta.phase_center.isscalar:
        raise ValueError(f"Expected a scalar ICRS, got {meta.phase_center}")
    if not meta.channel_width.isscalar:
        raise ValueError(f"Expected a scalar Quantity, got {meta.channel_width}")
    if not meta.integration_time.isscalar:
        raise ValueError(f"Expected a scalar Quantity, got {meta.integration_time}")
    if meta.pointings is not None and meta.pointings.isscalar:
        warnings.warn(f"Expected a vector ICRS, got {meta.pointings.shape}, assuming same pointing for all antennas.")
        meta.pointings = ac.ICRS(
            np.repeat(meta.pointings.ra.deg, num_antennas) * au.deg,
            np.repeat(meta.pointings.dec.deg, num_antennas) * au.deg
        )
    if meta.times.isscalar:
        raise ValueError(f"Expected a vector Time, got {meta.times}")
    meta.times = meta.times.tt  # Use TT time scale
    if meta.freqs.isscalar:
        raise ValueError(f"Expected a vector Quantity, got {meta.freqs}")
    if meta.antennas.isscalar:
        raise ValueError(f"Expected a vector EarthLocation, got {meta.antennas}")
    if meta.antenna_diameters.isscalar:
        warnings.warn(f"Expected antenna_diameters to be a vector, got {meta.antenna_diameters}")
        meta.antenna_diameters = np.repeat(meta.antenna_diameters.value, num_antennas) * meta.antenna_diameters.unit
    if isinstance(meta.mount_types, str):
        warnings.warn(f"Expected mount_types to be a vector, got {meta.mount_types}")
        meta.mount_types = [meta.mount_types] * num_antennas
    elif len(meta.mount_types) == 1:
        warnings.warn(f"Expected mount_types to be a vector, got {meta.mount_types}")
        meta.mount_types = [meta.mount_types[0]] * num_antennas

    if not meta.channel_width.unit.is_equivalent(au.Hz):
        raise ValueError(f"Expected channel width in Hz, got {meta.channel_width}")
    if not meta.integration_time.unit.is_equivalent(au.s):
        raise ValueError(f"Expected integration time in seconds, got {meta.integration_time}")
    if not meta.freqs.unit.is_equivalent(au.Hz):
        raise ValueError(f"Expected frequencies in Hz, got {meta.freqs}")
    if not meta.antenna_diameters.unit.is_equivalent(au.m):
        raise ValueError(f"Expected antenna diameters in meters, got {meta.antenna_diameters}")
    if meta.system_equivalent_flux_density is not None and (
            not meta.system_equivalent_flux_density.unit.is_equivalent(au.Jy)
    ):
        raise ValueError(f"Expected system equivalent flux density in Jy, got {meta.system_equivalent_flux_density}")

    if meta.pointings is not None and meta.pointings.shape != (num_antennas,):
        raise ValueError(f"Expected pointings to have shape ({num_antennas},), got {meta.pointings.shape}")
    if len(meta.antenna_names) != num_antennas:
        raise ValueError(f"Expected antenna_names to have length {num_antennas}, got {len(meta.antenna_names)}")
    if meta.antenna_diameters.shape != (num_antennas,):
        raise ValueError(
            f"Expected antenna_diameters to have shape ({num_antennas},), got {meta.antenna_diameters.shape}")
    if len(meta.mount_types) != num_antennas:
        raise ValueError(f"Expected mount_types to have length {num_antennas}, got {len(meta.mount_types)}")


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
    if isinstance(indices, slice):
        return indices
    steps = np.diff(indices)
    if not np.all(steps == steps[0]):
        raise NotContiguous("Indices must be contiguous.")
    step = steps[0]
    return slice(int(indices[0]), int(indices[-1]) + 1, step)


def _try_get_slice(indices: slice | np.ndarray) -> slice:
    try:
        return _get_slice(indices)
    except NotContiguous:
        return indices


class VisibilityData(NamedTuple):
    """
    Data for a single visibility.
    """
    vis: Optional[ComplexArray] = None  # [rows, num_freqs, 4] the visibility data
    weights: Optional[FloatArray] = None  # [rows, num_freqs, 4] the weights
    flags: Optional[BoolArray] = None  # [rows, num_freqs, 4] the flags


@dataclasses.dataclass(eq=False)
class MeasurementSet:
    ms_folder: str

    def __repr__(self):
        return f"MeasurementSet({self.ms_folder}, meta={self.meta_file}, data={self.data_file})"

    def __post_init__(self):
        self.meta_file = os.path.join(self.ms_folder, "meta.json")
        self.data_file = os.path.join(self.ms_folder, "data.h5")

        if not os.path.exists(self.meta_file):
            raise ValueError(f"Meta file {self.meta_file} does not exist.")

        if not os.path.exists(self.data_file):
            raise ValueError(f"Data file {self.data_file} does not exist.")

        self.meta = MeasurementSetMeta.parse_file(self.meta_file)
        _check_measurement_set_meta(self.meta)

    def get_rows(self, antenna1: np.ndarray | int, antenna2: np.ndarray | int,
                 time_idx: np.ndarray | int) -> np.ndarray:
        """
        Get the row index for the given antenna pair and time index.

        Args:
            antenna1: [num_rows] the first antenna
            antenna2: [num_rows] the second antenna
            time_idx: [num_rows] the time index

        Returns:
            rows: [num_rows] the row indices
        """
        if self.meta.with_autocorr:
            get_antenna_index = partial(_combination_with_replacement_index, n=len(self.meta.antennas))
        else:
            get_antenna_index = partial(_combination_index, n=len(self.meta.antennas))

        time_offset = time_idx * self.block_size

        return get_antenna_index(antenna1, antenna2) + time_offset

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
        Get the number of rows in a single time-slice of antennas (taking into account autocorrelations).
        """
        num_antennas = len(self.meta.antennas)
        if self.meta.with_autocorr:
            return (num_antennas * (num_antennas + 1)) // 2
        return (num_antennas * (num_antennas - 1)) // 2

    @property
    def num_baselines(self) -> int:
        """
        Get the number of baselines in the measurement set.
        """
        return self.block_size

    @cached_property
    def ref_time(self) -> at.Time:
        """
        Get the reference time of the measurement set.
        """
        # Casa convention is to use the first time as the reference time, also for FITS
        return self.meta.ref_time.tt

    def time_to_jnp(self, times: at.Time) -> jax.Array:
        """
        Convert the given times to TT seconds since ref time.

        Args:
            times: the times

        Returns:
            the times in TT seconds since ref time
        """
        return time_to_jnp(times, self.ref_time)

    def is_full_stokes(self) -> bool:
        """
        Check if the measurement set is full stokes.

        Returns:
            True if full stokes, False otherwise
        """
        return len(self.meta.coherencies) == 4

    def clone(self, ms_folder: str, preserve_symbolic_links: bool = False) -> 'MeasurementSet':
        """
        Clone the measurement set to the given folder.

        Args:
            ms_folder: the folder to clone the measurement set to
            preserve_symbolic_links: whether to keep symbolic links, or else copy the pointed-to files
        """
        ms_folder = os.path.abspath(ms_folder)
        copy_fn = shutil.copy2  # Copy with meta data
        shutil.copytree(
            self.ms_folder, ms_folder,
            symlinks=preserve_symbolic_links,
            dirs_exist_ok=False,
            copy_function=copy_fn
        )
        return MeasurementSet(ms_folder=ms_folder)

    @cached_property
    def far_field_delay_engine(self) -> BaseFarFieldDelayEngine:
        """
        Get the far field delay engine for the measurement set.

        Returns:
            the far field delay engine
        """
        return build_far_field_delay_engine(
            antennas=self.meta.antennas,
            phase_center=self.meta.phase_center,
            start_time=self.meta.times[0],
            end_time=self.meta.times[-1],
            ref_time=self.ref_time,
            verbose=True
        )

    @cached_property
    def near_field_delay_engine(self) -> build_near_field_delay_engine:
        """
        Get the near field delay engine for the measurement set.

        Returns:
            the near field delay engine
        """
        return build_near_field_delay_engine(
            antennas=self.meta.antennas,
            start_time=self.meta.times[0],
            end_time=self.meta.times[-1],
            ref_time=self.ref_time,
            verbose=True
        )

    @cached_property
    def geodesic_model(self) -> BaseGeodesicModel:
        """
        Get the geodesic model for the measurement set.

        Returns:
            the geodesic model
        """
        return build_geodesic_model(
            antennas=self.meta.antennas,
            array_location=self.meta.array_location,
            phase_center=self.meta.phase_center,
            obstimes=self.meta.times,
            ref_time=self.ref_time,
            pointings=self.meta.pointings
        )

    @cached_property
    def beam_gain_model(self) -> GainModel:
        """
        Get the beam gain model for the measurement set.

        Returns:
            the beam gain model
        """
        return beam_gain_model_factory(self)

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
        # antenna1: [num_rows]
        # antenna2: [num_rows]
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
            f.create_array("/", "antenna1", atom=tb.Int16Atom(), shape=(num_rows,))
            f.create_array("/", "antenna2", atom=tb.Int16Atom(), shape=(num_rows,))
            f.create_array("/", "time_idx", atom=tb.Int16Atom(), shape=(num_rows,))
            f.create_array("/", "vis", atom=tb.ComplexAtom(itemsize=16),
                           shape=(num_rows, num_freqs, len(meta.coherencies)))
            f.create_array("/", "weights", atom=tb.Float16Atom(), shape=(num_rows, num_freqs, len(meta.coherencies)))
            f.create_array("/", "flags", atom=tb.BoolAtom(), shape=(num_rows, num_freqs, len(meta.coherencies)))

        engine = build_far_field_delay_engine(
            antennas=meta.antennas,
            phase_center=meta.phase_center,
            start_time=meta.times[0],
            end_time=meta.times[-1],
            ref_time=meta.times[0],
            verbose=True
        )

        if meta.with_autocorr:
            baseline_pairs = np.asarray(list(itertools.combinations_with_replacement(range(num_antennas), 2)),
                                        dtype=np.int32)
        else:
            baseline_pairs = np.asarray(list(itertools.combinations(range(num_antennas), 2)),
                                        dtype=np.int32)
        antenna1 = baseline_pairs[:, 0]
        antenna2 = baseline_pairs[:, 1]

        start_row = 0

        compute_uvw_jax = jax.jit(partial(engine.compute_uvw, convention=meta.convention))
        for time_idx in range(num_times):
            # UVW are position(antenna2) - position(antenna1)
            # antenna1, antenna2 are all possible baselines
            time = meta.times[time_idx]

            time_idx = np.full(antenna1.shape, time_idx, dtype=np.int32)

            # Don't use interpolation, so precise.
            uvw = compute_uvw_jax(
                times=jnp.repeat(time_to_jnp(time, meta.times[0])[None], len(antenna1), axis=0),
                antenna1=jnp.asarray(antenna1),
                antenna2=jnp.asarray(antenna2)
            )  # [num_baselines, 3]

            uvw = np.asarray(uvw)

            end_row = start_row + uvw.shape[0]

            with tb.open_file(data_file, "r+") as f:
                f.root.uvw[start_row:end_row] = uvw
                f.root.antenna1[start_row:end_row] = antenna1
                f.root.antenna2[start_row:end_row] = antenna2
                f.root.time_idx[start_row:end_row] = time_idx
                f.root.vis[start_row:end_row] = 0.
                f.root.weights[start_row:end_row] = 0.
                f.root.flags[start_row:end_row] = False
            start_row += uvw.shape[0]

        return MeasurementSet(ms_folder=ms_folder)

    def put(self, data: VisibilityData, antenna1: np.ndarray | int, antenna2: np.ndarray | int, times: at.Time,
            freqs: au.Quantity | None = None):
        """
        Put the visibility data for the given antenna pair and time index.

        Args:
            data: the visibility data being put
            antenna1: [num_rows] the first antenna
            antenna2: [num_rows] the second antenna
            times: the times
            freqs: the frequencies, if None, all frequencies are returned
        """
        time_idx = get_centred_insert_index((times - self.ref_time).sec, (self.meta.times - self.ref_time).sec)
        if freqs is not None:
            freqs_idx = get_centred_insert_index((freqs.value - self.meta.freqs[0]).to('Hz').value,
                                                 (self.meta.freqs - self.meta.freqs[0]).to('Hz').value)
            freqs_idx = _try_get_slice(freqs_idx)
        else:
            freqs_idx = slice(None, None, None)

        rows = self.get_rows(antenna1=antenna1, antenna2=antenna2, time_idx=time_idx)

        num_rows = len(rows)

        rows = np.unique(rows)
        rows = _try_get_slice(rows)

        def _check_data(name, array):
            if len(np.shape(array)) != 3:
                raise ValueError(
                    f"Expected {name} to have shape (num_rows, num_freqs, num_coherencies), got {np.shape(array)}"
                )
            if np.shape(array)[0] != num_rows:
                raise ValueError(
                    f"Expected {name} to have {num_rows} rows, got {np.shape(array)[0]}"
                )
            if freqs is not None and np.shape(array)[1] != len(freqs):
                raise ValueError(
                    f"Expected {name} to have {len(freqs)} frequencies, got {np.shape(array)[1]}"
                )
            if np.shape(array)[2] != len(self.meta.coherencies):
                raise ValueError(
                    f"Expected {name} to have {len(self.meta.coherencies)} coherencies, got {np.shape(array)[2]}"
                )

        with tb.open_file(self.data_file, 'r+') as f:
            if data.vis is not None:
                _check_data("vis", data.vis)
                f.root.vis[rows, freqs_idx, :] = data.vis
            if data.weights is not None:
                _check_data("weights", data.weights)
                f.root.weights[rows, freqs_idx, :] = data.weights
            if data.flags is not None:
                _check_data("flags", data.flags)
                f.root.flags[rows, freqs_idx, :] = data.flags

    def get_uvw(self, row_slice: slice) -> np.ndarray:
        """
        Get the UVW coordinates for the given row slice.

        Args:
            row_slice: the row slice

        Returns:
            uvw: [num_rows, 3] the UVW coordinates
        """
        row_slice = _try_get_slice(row_slice)
        with tb.open_file(self.data_file, 'r') as f:
            return f.root.uvw[row_slice]

    def match(self, antenna1: np.ndarray | int, antenna2: np.ndarray | int, times: at.Time,
              freqs: au.Quantity | None = None) -> VisibilityData:
        """
        Get the visibility data for the given antenna pair, times and frequencies. The shapes of inputs must broadcast.
        I.e. scalars will broadcast.

        Args:
            antenna1: [num_ant] the first antenna
            antenna2: [num_ant] the second antenna
            times: [num_times] the times
            freqs: [num_freqs] the frequencies, if None, all frequencies are returned

        Returns:
            visibility_data: [num_rows, num_freqs, num_coherencies] the visibility data matching the given
            antenna pairs, times and frequencies.
        """

        (i0_time, alpha0_time, i1_time, alpha1_time) = get_interp_indices_and_weights(
            x=(times - self.ref_time).sec, xp=(self.meta.times - self.ref_time).sec
        )
        (i0_time, alpha0_time, i1_time, alpha1_time) = jax.tree.map(
            np.asarray, (i0_time, alpha0_time, i1_time, alpha1_time)
        )
        rows0 = self.get_rows(antenna1=antenna1, antenna2=antenna2, time_idx=i0_time)
        rows1 = self.get_rows(antenna1=antenna1, antenna2=antenna2, time_idx=i1_time)

        # For accessing HDF5 slices are faster
        rows0, inverse_map0 = np.unique(rows0, return_inverse=True)
        rows0 = _try_get_slice(rows0)
        rows1, inverse_map1 = np.unique(rows1, return_inverse=True)
        rows1 = _try_get_slice(rows1)

        def access_non_unique(h5_array, unique_rows, inverse_map):
            return _access_non_unique(h5_array, unique_rows, inverse_map)

        with tb.open_file(self.data_file, 'r') as f:
            vis = (
                    access_non_unique(f.root.vis, rows0, inverse_map0) * alpha0_time[:, None, None]
                    + access_non_unique(f.root.vis, rows1, inverse_map1) * alpha1_time[:, None, None]
            )
            weights = (
                    access_non_unique(f.root.weights, rows0, inverse_map0) * alpha0_time[:, None, None]
                    + access_non_unique(f.root.weights, rows1, inverse_map1) * alpha1_time[:, None, None]
            )
            flags = (
                    access_non_unique(f.root.flags, rows0, inverse_map0) * alpha0_time[:, None, None]
                    + access_non_unique(f.root.flags, rows1, inverse_map1) * alpha1_time[:, None, None]
            )

        if freqs is not None:
            (i0_freq, alpha0_freq, i1_freq, alpha1_freq) = get_interp_indices_and_weights(
                x=freqs.value, xp=self.meta.freqs.value
            )
            (i0_freq, alpha0_freq, i1_freq, alpha1_freq) = jax.tree.map(
                np.asarray, (i0_freq, alpha0_freq, i1_freq, alpha1_freq)
            )
            i0_freq = _try_get_slice(i0_freq)
            i1_freq = _try_get_slice(i1_freq)
            vis = vis[:, i0_freq] * alpha0_freq[:, None] + vis[:, i1_freq] * alpha1_freq[:, None]
            weights = weights[:, i0_freq] * alpha0_freq + weights[:, i1_freq] * alpha1_freq
            flags = flags[:, i0_freq] * alpha0_freq[:, None] + flags[:, i1_freq] * alpha1_freq[:, None]

        # Cast flags to bool, will effective to OR operation
        flags = flags.astype(np.bool_)

        return VisibilityData(
            vis=vis,
            weights=weights,
            flags=flags
        )

    def create_block_generator(self, start_time_idx: int = 0, end_time_idx: int | None = None, vis: bool = True,
                               weights: bool = True, flags: bool = True, num_blocks: int = 1,
                               corrs: Tuple[str] | Tuple[Tuple[str]] | None = None) -> Generator[
        Tuple[at.Time, VisibilityCoords, VisibilityData], VisibilityData | None, None
    ]:
        """
        Create a block generator for the measurement set.

        Args:
            start_time_idx: the start time index, default 0
            end_time_idx: the end time index, default the end
            vis: whether to include visibilities, default True
            weights: whether to include weights, default True
            flags: whether to include flags, default True
            num_blocks: the number of blocks to yield at a time, default 1
            corrs: the coherencies to translate to, default None

        Returns:
            a generator that yields:
                times: [num_blocks] the times
                coords: [num_rows] batched coordinates
                data: [num_rows] batched data

        Raises:
            ValueError: if the number of blocks is not positive
            ValueError: if the total rows is not divisible by the block size and number of blocks
            RuntimeError: if the time index is out of bounds
            RuntimeError: if the row is out of bounds
        """
        if corrs is None:
            corrs = self.meta.coherencies

        def transform_corr_from_fn(coh: jax.Array) -> jax.Array:
            return broadcast_translate_corrs(
                coh.astype(jnp.promote_types(coh.dtype, jnp.float32)), from_corrs=self.meta.coherencies, to_corrs=corrs
            )

        def transform_corr_to_fn(coh: jax.Array) -> jax.Array:
            return broadcast_translate_corrs(
                coh.astype(jnp.promote_types(coh.dtype, jnp.float32)), from_corrs=corrs, to_corrs=self.meta.coherencies
            )

        if num_blocks <= 0:
            raise ValueError(f"Number of blocks {num_blocks} must be positive.")
        if self.meta.with_autocorr:
            start_antenna1 = 0
            start_antenna2 = 0
        else:
            start_antenna1 = 0
            start_antenna2 = 1

        start_row = self.get_rows(start_antenna1, start_antenna2, start_time_idx)

        if end_time_idx is None:
            end_row = self.num_rows
        else:
            end_row = self.get_rows(start_antenna1, start_antenna2, end_time_idx) + self.block_size

        total_rows = end_row - start_row
        if total_rows % (self.block_size * num_blocks) != 0:
            raise ValueError(
                f"Total rows {total_rows} must be divisible by "
                f"block size {self.block_size} and number of blocks {num_blocks}."
            )

        compute_visibility_coords_jit = jax.jit(
            partial(
                self.far_field_delay_engine.compute_visibility_coords,
                with_autocorr=self.meta.with_autocorr,
                convention=self.meta.convention
            )
        )

        def reshape_to_blocks(x):
            return x.reshape((num_blocks, self.num_baselines) + np.shape(x)[1:])

        def reshape_from_blocks(x):
            return x.reshape((num_blocks * self.num_baselines,) + np.shape(x)[2:])

        freqs = quantity_to_jnp(self.meta.freqs)

        with tb.open_file(self.data_file, 'r+') as f:
            from_time_idx = start_time_idx
            for from_row in range(start_row, end_row, self.block_size * num_blocks):
                to_time_idx = from_time_idx + num_blocks
                if to_time_idx > len(self.meta.times):
                    raise RuntimeError(f"Time index {to_time_idx} out of bounds.")
                times_slice = self.meta.times[from_time_idx:to_time_idx]

                to_row = from_row + self.block_size * num_blocks
                if to_row > end_row:
                    raise RuntimeError(f"Row {from_row} + block size {self.block_size} * num blocks {num_blocks}.")

                from_time_idx += num_blocks

                coords = compute_visibility_coords_jit(freqs=freqs, times=self.time_to_jnp(times_slice))

                # We transform to the desired coherencies, and unstack into blocks.
                data = VisibilityData(
                    vis=reshape_to_blocks(mp_policy.cast_to_vis(
                        transform_corr_from_fn(f.root.vis[from_row:to_row]))) if vis is not None else None,
                    weights=reshape_to_blocks(mp_policy.cast_to_weight(
                        transform_corr_from_fn(f.root.weights[from_row:to_row]))) if weights is not None else None,
                    flags=reshape_to_blocks(mp_policy.cast_to_flag(
                        transform_corr_from_fn(f.root.flags[from_row:to_row]))) if flags is not None else None
                )
                response = yield (times_slice, coords, data)
                if response is not None and isinstance(response, VisibilityData):
                    # For each we transform back to the original coherencies, and stack into rows.
                    if response.vis is not None:
                        f.root.vis[from_row:to_row] = np.asarray(
                            reshape_from_blocks(transform_corr_to_fn(response.vis)),
                            np.float32
                        )
                    if response.weights is not None:
                        f.root.weights[from_row:to_row] = np.asarray(
                            reshape_from_blocks(transform_corr_to_fn(response.weights)),
                            np.float16
                        )
                    if response.flags is not None:
                        f.root.flags[from_row:to_row] = np.asarray(
                            reshape_from_blocks(transform_corr_to_fn(response.flags)),
                            np.bool_
                        )


def get_non_unqiue(h5_array, indices, axis=0, indices_sorted: bool = False):
    """
    Access non-unique elements from an array based on the indices provided.

    :param h5_array: the H5 array to access
    :param indices: the indices to access
    :param axis: the axis along which to access the indices
    :param indices_sorted: whether the indices are sorted. If True, the function will be faster.

    :return:
        The non-unique elements from the H5 array based on the indices provided
    """
    if len(np.shape(indices)) != 1:
        raise ValueError("Indices must be a 1D array")
    unique_indices, inverse_map = np.unique(indices, return_inverse=True)
    if indices_sorted and (len(unique_indices) == len(indices)):
        unique_indices = indices
        return _access_non_unique(h5_array, unique_indices, inverse_map, axis, _already_unique=True)
    # Even if already unique, we need to remap the indices to account for ordering.
    return _access_non_unique(h5_array, unique_indices, inverse_map, axis)


def put_non_unique(h5_array, indices, values, axis=0, indices_sorted: bool = False):
    """
    Put non-unique elements into an array based on the indices provided.

    :param h5_array: the H5 array to put into
    :param indices: the indices to put into. Duplicates are allowed. Only the first occurrence will be used.
    :param values: the values to put into the array
    :param axis: the axis along which to put the indices
    :param indices_sorted: whether the indices are sorted. If True, the function will be faster.

    :return:
        The H5 array with the non-unique elements put in
    """
    if len(np.shape(indices)) != 1:
        raise ValueError("Indices must be a 1D array")
    unique_indices = np.unique(indices)
    if indices_sorted and (len(unique_indices) == len(indices)):
        return _put_non_unique(h5_array, unique_indices, values, axis, _already_unique=True)
    return _put_non_unique(h5_array, unique_indices, values, axis)


def _access_non_unique(h5_array, unique_indices, inverse_map, axis=0, _already_unique: bool = False):
    # Prepare an index tuple to handle different axes
    index_tuple = tuple(
        unique_indices if i == axis else slice(None)
        for i in range(h5_array.ndim)
    )

    # Access the unique elements
    unique_get = h5_array[index_tuple]

    # Check if remapping is necessary
    if _already_unique:
        return unique_get

    # Generate an index tuple for reshaping based on the inverse map
    reshape_index_tuple = tuple(
        inverse_map if i == axis else slice(None)
        for i in range(h5_array.ndim)
    )

    # Reshape the unique get array back to the shape of the original input
    return unique_get[reshape_index_tuple]


def _put_non_unique(h5_array, unique_indices, values, axis=0, _already_unique: bool = False):
    # Prepare an index tuple to handle different axes
    index_tuple = tuple(
        unique_indices if i == axis else slice(None)
        for i in range(h5_array.ndim)
    )
    if _already_unique:
        h5_array[index_tuple] = values
        return
    h5_array[index_tuple] = values[index_tuple]


def beam_gain_model_factory(ms: MeasurementSet) -> BaseSphericalInterpolatorGainModel:
    if ms.meta.static_beam:
        model_times = at.Time([ms.meta.times.tt.mean()])
    else:
        model_times = at.Time([ms.meta.times.tt.min(), ms.meta.times.tt.max()])
    return build_beam_gain_model(array_name=ms.meta.array_name, times=model_times, full_stokes=ms.is_full_stokes())
