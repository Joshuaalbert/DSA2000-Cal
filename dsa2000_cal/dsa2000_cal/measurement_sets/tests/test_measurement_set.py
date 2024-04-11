import itertools

import numpy as np
import pytest
import tables as tb
from astropy import coordinates as ac, units as au, time as at

from dsa2000_cal.measurement_sets.measurement_set import _combination_with_replacement_index, _combination_index, \
    _try_get_slice, _get_slice, NotContiguous, MeasurementSetMetaV0, MeasurementSet, VisibilityData, \
    _get_centred_insert_index


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
def test_measurement_set_shapes(tmp_path, with_autocorr):
    meta = MeasurementSetMetaV0(
        array_name="test_array",
        array_location=ac.EarthLocation.from_geodetic(0 * au.deg, 0 * au.deg, 0 * au.m),
        phase_tracking=ac.ICRS(0 * au.deg, 0 * au.deg),
        channel_width=au.Quantity(1, au.Hz),
        integration_time=au.Quantity(1, au.s),
        coherencies=['XX', 'XY', 'YX', 'YY'],
        pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
        times=at.Time.now() + np.arange(10) * au.s,
        freqs=au.Quantity([1, 2, 3], au.Hz),
        antennas=ac.EarthLocation.from_geodetic(np.arange(5) * au.deg, np.arange(5) * au.deg, np.arange(5) * au.m),
        antenna_names=[f"antenna_{i}" for i in range(5)],
        antenna_diameters=au.Quantity(np.ones(5), au.m),
        with_autocorr=with_autocorr,
        mount_types='ALT-AZ',
        system_equivalent_flux_density=au.Quantity(1, au.Jy)
    )
    ms = MeasurementSet.create_measurement_set(str(tmp_path / "test_ms"), meta)

    assert len(meta.antenna_diameters) == 5
    assert len(meta.antenna_names) == 5
    assert len(meta.mount_types) == 5
    assert len(meta.pointings) == 5

    if ms.meta.with_autocorr:
        assert ms.num_rows == 10 * 5 * 6 // 2
    else:
        assert ms.num_rows == 10 * 5 * 4 // 2

    with tb.open_file(ms.data_file, 'r') as f:
        rows = ms.get_rows(f.root.antenna_1[:], f.root.antenna_2[:], f.root.time_idx[:])
        assert np.all(rows == np.arange(ms.num_rows))


@pytest.mark.parametrize("with_autocorr", [True, False])
def test_measurement_setting(tmp_path, with_autocorr):
    meta = MeasurementSetMetaV0(
        array_name="test_array",
        array_location=ac.EarthLocation.from_geodetic(0 * au.deg, 0 * au.deg, 0 * au.m),
        phase_tracking=ac.ICRS(0 * au.deg, 0 * au.deg),
        channel_width=au.Quantity(1, au.Hz),
        integration_time=au.Quantity(1, au.s),
        coherencies=['XX', 'XY', 'YX', 'YY'],
        pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
        times=at.Time.now() + np.arange(10) * au.s,
        freqs=au.Quantity([1, 2, 3], au.Hz),
        antennas=ac.EarthLocation.from_geodetic(np.arange(5) * au.deg, np.arange(5) * au.deg, np.arange(5) * au.m),
        antenna_names=[f"antenna_{i}" for i in range(5)],
        antenna_diameters=au.Quantity(np.ones(5), au.m),
        with_autocorr=with_autocorr,
        mount_types='ALT-AZ',
        system_equivalent_flux_density=au.Quantity(1, au.Jy)
    )
    ms = MeasurementSet.create_measurement_set(str(tmp_path / "test_ms"), meta)

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


@pytest.mark.parametrize("with_autocorr", [True, False])
def test_measurement_setting_put(tmp_path, with_autocorr):
    meta = MeasurementSetMetaV0(
        array_name="test_array",
        array_location=ac.EarthLocation.from_geodetic(0 * au.deg, 0 * au.deg, 0 * au.m),
        phase_tracking=ac.ICRS(0 * au.deg, 0 * au.deg),
        channel_width=au.Quantity(1, au.Hz),
        integration_time=au.Quantity(1, au.s),
        coherencies=['XX', 'XY', 'YX', 'YY'],
        pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
        times=at.Time.now() + np.arange(10) * au.s,
        freqs=au.Quantity([1, 2, 3], au.Hz),
        antennas=ac.EarthLocation.from_geodetic(np.arange(5) * au.deg, np.arange(5) * au.deg, np.arange(5) * au.m),
        antenna_names=[f"antenna_{i}" for i in range(5)],
        antenna_diameters=au.Quantity(np.ones(5), au.m),
        with_autocorr=with_autocorr,
        mount_types='ALT-AZ',
        system_equivalent_flux_density=au.Quantity(1, au.Jy)
    )
    ms = MeasurementSet.create_measurement_set(str(tmp_path / "test_ms"), meta)

    gen = ms.create_block_generator()
    gen_response = None
    while True:

        try:
            time, coords, data = gen.send(gen_response)
        except StopIteration:
            break

        ms.put(
            data=VisibilityData(
                vis=2 * np.ones((ms.block_size, 3, 4), dtype=np.complex64),
                weights=2 * np.ones((ms.block_size, 3, 4), dtype=np.float16),
                flags=np.zeros((ms.block_size, 3, 4), dtype=np.bool_)
            ),
            antenna_1=coords.antenna_1,
            antenna_2=coords.antenna_2,
            times=at.Time([time.isot] * ms.block_size, format='isot')
        )

    data = ms.match(antenna_1=0, antenna_2=1, times=ms.meta.times)
    assert data.vis.shape == (len(meta.times), len(meta.freqs), 4)

    data = ms.match(antenna_1=np.asarray([0, 1]), antenna_2=np.asarray([1, 2]), times=ms.meta.times[:2])
    assert data.vis.shape == (2, len(meta.freqs), 4)

    gen = ms.create_block_generator()
    for time, coords, data in gen:
        assert np.all(data.vis.real == 2)
        assert np.all(data.weights == 2)
        assert np.bitwise_not(np.any(data.flags))


def test__get_centred_insert_index():
    time_centres = np.asarray([0.5, 1.5, 2.5])

    times_to_insert = np.asarray([0, 1, 2])
    expected_time_idx = np.asarray([0, 1, 2])
    time_idx = _get_centred_insert_index(times_to_insert, time_centres)
    np.testing.assert_array_equal(time_idx, expected_time_idx)

    times_to_insert = np.asarray([1, 2, 3 - 1e-10])
    expected_time_idx = np.asarray([1, 2, 2])
    time_idx = _get_centred_insert_index(times_to_insert, time_centres)
    np.testing.assert_array_equal(time_idx, expected_time_idx)

    with pytest.raises(ValueError):
        _get_centred_insert_index(np.asarray([3]), time_centres)

    with pytest.raises(ValueError):
        _get_centred_insert_index(np.asarray([0 - 1e-10]), time_centres)
