import dataclasses
import itertools
import os.path
import warnings
from functools import cached_property
from typing import Literal, List, Union, Annotated, NamedTuple, Generator, Tuple

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import tables as pt
from pydantic import Field

from dsa2000_cal.common.coord_utils import earth_location_to_uvw
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


class VisibilityCoords(NamedTuple):
    """
    Coordinates for a single visibility.
    """
    uvw: jax.Array | np.ndarray  # [rows, 3] the uvw coordinates
    time_mjs: jax.Array | np.ndarray  # [rows] the time
    antenna_1: jax.Array | np.ndarray  # [rows] the first antenna
    antenna_2: jax.Array | np.ndarray  # [rows] the second antenna
    time_idx: jax.Array | np.ndarray  # [rows] the time index


class VisibilityData(NamedTuple):
    """
    Data for a single visibility.
    """
    vis: jax.Array | np.ndarray | None = None  # [rows, num_freqs, 4] the visibility data
    weights: jax.Array | np.ndarray | None = None  # [rows, num_freqs, 4] the weights
    flags: jax.Array | np.ndarray | None = None  # [rows, num_freqs, 4] the flags


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

    def get_row(self, antenna_1: int, antenna_2: int, time_idx: int) -> int:
        """
        Get the row index for the given antenna pair and time index.
        """
        with pt.open_file(self.data_file) as f:
            (rows1,) = np.where(f.root.antenna_1[:] == antenna_1)
            (rows2,) = np.where(f.root.antenna_2[rows1] == antenna_2)
            (rows3,) = np.where(f.root.time_idx[rows1[rows2]] == time_idx)

            row = rows1[rows2[rows3]]
            if len(row) == 0:
                raise ValueError(f"No rows found for antenna pair {antenna_1}, {antenna_2} and time index {time_idx}.")
            return row[0]

    @cached_property
    def num_rows(self) -> int:
        """
        Get the number of rows in the measurement set.
        """
        with pt.open_file(self.data_file) as f:
            return f.root.uvw.shape[0]

    @cached_property
    def block_size(self) -> int:
        """
        Get the number of rows in the measurement set.
        """
        num_antennas = len(self.meta.antennas)
        return (num_antennas * (num_antennas - 1)) // 2

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
        num_rows = num_antennas * (num_antennas - 1) // 2 * num_times
        with pt.open_file(data_file, "w") as f:
            f.create_array("/", "uvw", atom=pt.Float32Atom(), shape=(num_rows, 3))
            f.create_array("/", "antenna_1", atom=pt.Int16Atom(), shape=(num_rows,))
            f.create_array("/", "antenna_2", atom=pt.Int16Atom(), shape=(num_rows,))
            f.create_array("/", "time_idx", atom=pt.Int16Atom(), shape=(num_rows,))
            f.create_array("/", "vis", atom=pt.ComplexAtom(itemsize=8),
                           shape=(num_rows, num_freqs, 4))  # single precision complex
            f.create_array("/", "weights", atom=pt.Float16Atom(), shape=(num_rows, num_freqs, 4))
            f.create_array("/", "flags", atom=pt.BoolAtom(), shape=(num_rows, num_freqs, 4))

        start_row = 0
        for time_idx, time in enumerate(meta.times):
            # UVW are position(antenna_2) - position(antenna_1)
            # antenna_1, antenna_2 are all possible baselines

            baseline_pairs = np.asarray(list(itertools.combinations(range(num_antennas), 2)), dtype=np.int32)
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

            with pt.open_file(data_file, "r+") as f:
                f.root.uvw[start_row:end_row] = uvw
                f.root.antenna_1[start_row:end_row] = antenna_1
                f.root.antenna_2[start_row:end_row] = antenna_2
                f.root.time_idx[start_row:end_row] = time_idx
                f.root.vis[start_row:end_row] = 0.
                f.root.weights[start_row:end_row] = 1.
                f.root.flags[start_row:end_row] = False
            start_row += uvw.shape[0]

        return MeasurementSet(ms_folder=ms_folder)

    def create_block_generator(self, start_time_idx: int = 0, end_time_idx: int | None = None,
                               vis: bool = True, weights: bool = True, flags: bool = True) -> Generator[
        Tuple[at.Time, VisibilityCoords, VisibilityData], VisibilityData | None, None
    ]:
        start_row = self.get_row(0, 1, start_time_idx)

        if end_time_idx is None:
            end_row = self.num_rows
        else:
            end_row = self.get_row(0, 1, end_time_idx) + self.block_size

        time_mjs = jnp.asarray(self.meta.times.mjd * 86400, dtype=jnp.float64)

        with pt.open_file(self.data_file, 'r+') as f:
            time_idx = start_time_idx
            for row in range(start_row, end_row, self.block_size):
                time = self.meta.times[time_idx]
                time_idx += 1
                coords = VisibilityCoords(
                    uvw=f.root.uvw[row:row + self.block_size],
                    time_mjs=time_mjs[f.root.time_idx[row:row + self.block_size]],
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


def test_measurement_set():
    meta = MeasurementSetMetaV0(
        array_name="test_array",
        array_location=ac.EarthLocation.from_geodetic(0 * au.deg, 0 * au.deg, 0 * au.m),
        phase_tracking=ac.ICRS(0 * au.deg, 0 * au.deg),
        channel_width=au.Quantity(1, au.Hz),
        integration_time=au.Quantity(1, au.s),
        coherencies='stokes',
        pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
        times=at.Time.now() + np.arange(10) * au.s,
        freqs=au.Quantity([1, 2, 3], au.Hz),
        antennas=ac.EarthLocation.from_geodetic(np.arange(5) * au.deg, np.arange(5) * au.deg, np.arange(5) * au.m),
        antenna_names=[f"antenna_{i}" for i in range(5)],
        antenna_diameters=au.Quantity(np.ones(5), au.m)
    )
    ms = MeasurementSet.create_measurement_set("test_ms", meta)

    assert ms.num_rows == 10 * 5 * 4 // 2

    assert ms.get_row(0, 1, 0) == 0
    assert ms.get_row(0, 1, 1) == ms.block_size
    assert ms.get_row(0, 1, 9) == ms.num_rows - ms.block_size

    gen = ms.create_block_generator()

    gen_response = None
    while True:
        try:
            time, coords, data = gen.send(gen_response)
        except StopIteration:
            break

        assert coords.uvw.shape == (ms.block_size, 3)
        assert coords.time_mjs.shape == (ms.block_size,)
        assert coords.antenna_1.shape == (ms.block_size,)
        assert coords.antenna_2.shape == (ms.block_size,)
        assert coords.time_idx.shape == (ms.block_size,)
        assert data.vis.shape == (ms.block_size, 3, 4)
        assert data.weights.shape == (ms.block_size, 3, 4)
        assert data.flags.shape == (ms.block_size, 3, 4)
        for time_idx in coords.time_idx:
            assert ms.meta.times[time_idx] == time

        gen_response = VisibilityData(
            vis=np.ones((ms.block_size, 3, 4), dtype=np.complex64),
            weights=np.zeros((ms.block_size, 3, 4), dtype=np.float16),
            flags=np.ones((ms.block_size, 3, 4), dtype=np.bool)
        )

    gen = ms.create_block_generator()
    gen_response = None
    while True:
        try:
            time, coords, data = gen.send(gen_response)
        except StopIteration:
            break

        assert np.all(data.vis.real == 1)
        assert np.all(data.weights == 0)
        assert np.all(data.flags)
