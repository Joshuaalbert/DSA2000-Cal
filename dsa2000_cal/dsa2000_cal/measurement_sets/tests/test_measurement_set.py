import itertools

import numpy as np
import pytest
import tables as tb
from astropy import coordinates as ac, units as au, time as at

from dsa2000_cal.measurement_sets.measurement_set import _combination_with_replacement_index, _combination_index, \
    _try_get_slice, _get_slice, NotContiguous, MeasurementSetMetaV0, MeasurementSet, VisibilityData


def test__combination_with_replacementindex():
    example = list(itertools.combinations_with_replacement(range(5), 2))
    for i, (a, b) in enumerate(example):
        assert _combination_with_replacement_index(a, b, 5) == i


def test__combination_index():
    example = list(itertools.combinations(range(5), 2))
    for i, (a, b) in enumerate(example):
        assert _combination_index(a, b, 5) == i


def test__try_get_slice():
    assert _try_get_slice([0, 1, 2, 3, 4]) == slice(0, 5, 1)
    assert _try_get_slice([0, 2, 3]) == [0, 2, 3]


def test__get_slice():
    assert _get_slice([0, 1, 2, 3, 4]) == slice(0, 5, 1)
    assert _get_slice([0, 2, 4]) == slice(0, 5, 2)
    with pytest.raises(NotContiguous, match='Indices must be contiguous.'):
        _get_slice([0, 2, 3, 4])


@pytest.mark.parametrize("with_autocorr", [True, False])
def test_measurement_set(with_autocorr):
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
        antenna_diameters=au.Quantity(np.ones(5), au.m),
        with_autocorr=with_autocorr
    )
    ms = MeasurementSet.create_measurement_set("test_ms", meta)

    if ms.meta.with_autocorr:
        assert ms.num_rows == 10 * 5 * 6 // 2
    else:
        assert ms.num_rows == 10 * 5 * 4 // 2

    with tb.open_file(ms.data_file, 'r') as f:
        rows = ms.get_rows(f.root.antenna_1[:], f.root.antenna_2[:], f.root.time_idx[:])
        assert np.all(rows == np.arange(ms.num_rows))

    gen = ms.create_block_generator()

    gen_response = None
    while True:
        try:
            time, coords, data = gen.send(gen_response)
        except StopIteration:
            break

        assert coords.uvw.shape == (ms.block_size, 3)
        assert coords.time_obs.shape == (ms.block_size,)
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
            flags=np.ones((ms.block_size, 3, 4), dtype=np.bool_)
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

    data = ms.match(antenna_1=0, antenna_2=1, times=ms.meta.times)
    print(data.vis.shape)
