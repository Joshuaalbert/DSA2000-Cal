import itertools

import numpy as np
import pytest
import tables as tb
from astropy import coordinates as ac, units as au, time as at

from dsa2000_cal.adapter.from_casa_ms import transfer_from_casa
from dsa2000_cal.measurement_sets.measurement_set import _combination_with_replacement_index, _combination_index, \
    _try_get_slice, _get_slice, NotContiguous, MeasurementSetMetaV0, MeasurementSet, VisibilityData, get_non_unqiue, \
    put_non_unique


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

    assert ms.num_rows % ms.block_size == 0

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
            assert ms.meta.x[time_idx] == time

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

    data = ms.match(antenna_1=0, antenna_2=1, times=ms.meta.x)
    assert data.vis.shape == (len(meta.times), len(meta.freqs), 4)

    data = ms.match(antenna_1=np.asarray([0, 1]), antenna_2=np.asarray([1, 2]), times=ms.meta.x[:2])
    assert data.vis.shape == (2, len(meta.freqs), 4)

    gen = ms.create_block_generator()
    for time, coords, data in gen:
        assert np.all(data.vis.real == 2)
        assert np.all(data.weights == 2)
        assert np.bitwise_not(np.any(data.flags))


@pytest.mark.parametrize("with_autocorr", [True, False])
def test_multi_block_gen(tmp_path, with_autocorr):
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

    with pytest.raises(ValueError):
        gen = ms.create_block_generator(num_blocks=0)
        next(gen)
    with pytest.raises(ValueError):
        gen = ms.create_block_generator(num_blocks=4)
        next(gen)

    gen = ms.create_block_generator(num_blocks=2)
    gen_response = None

    block_count = 0
    while True:
        try:
            times, coords, data = gen.send(gen_response)
        except StopIteration:
            break

        block_count += 2
        assert len(times) == 2
        assert coords.uvw.shape == (ms.block_size * 2, 3)
        assert coords.time_obs.shape == (ms.block_size * 2,)
        assert coords.antenna_1.shape == (ms.block_size * 2,)
        assert coords.antenna_2.shape == (ms.block_size * 2,)
        assert coords.time_idx.shape == (ms.block_size * 2,)
        assert data.vis.shape == (ms.block_size * 2, 3, 4)
        assert data.weights.shape == (ms.block_size * 2, 3, 4)
        assert data.flags.shape == (ms.block_size * 2, 3, 4)
        for time_idx in coords.time_idx:
            assert ms.meta.x[time_idx] in times
        for time in times:
            assert time in ms.meta.x[coords.time_idx]

        gen_response = VisibilityData(
            vis=np.ones((ms.block_size * 2, 3, 4), dtype=np.complex64),
            weights=np.zeros((ms.block_size * 2, 3, 4), dtype=np.float16),
            flags=np.ones((ms.block_size * 2, 3, 4), dtype=np.bool_)
        )

    assert block_count == 10


def test_get_non_unique():
    h5_array = np.array([[1, 2], [3, 4], [5, 6]])
    indices = np.array([0, 1, 0])
    result = get_non_unqiue(h5_array, indices, axis=0)
    np.testing.assert_allclose(result, np.array([[1, 2], [3, 4], [1, 2]]))

    indices = np.array([0, 1])
    result = get_non_unqiue(h5_array, indices, axis=0, indices_sorted=True)
    np.testing.assert_allclose(result, np.array([[1, 2], [3, 4]]))

    indices = np.array([0, 1, 0])
    result = get_non_unqiue(h5_array, indices, axis=1)
    np.testing.assert_allclose(result, np.array([[1, 2, 1], [3, 4, 3], [5, 6, 5]]))

    indices = np.array([0, 1])
    result = get_non_unqiue(h5_array, indices, axis=1, indices_sorted=True)
    np.testing.assert_allclose(result, np.array([[1, 2], [3, 4], [5, 6]]))


def test_put_non_unique():
    h5_array = np.array([[1, 2], [3, 4], [5, 6]])
    indices = np.array([0, 1, 0])
    values = np.array([[7, 8], [9, 10], [11, 12]])
    put_non_unique(h5_array, indices, values, axis=0)
    np.testing.assert_allclose(h5_array, np.array([[7, 8], [9, 10], [5, 6]]))

    h5_array = np.array([[1, 2], [3, 4], [5, 6]])
    indices = np.array([0, 1])
    values = np.array([[7, 8], [9, 10]])
    put_non_unique(h5_array, indices, values, axis=0, indices_sorted=True)
    np.testing.assert_allclose(h5_array, np.array([[7, 8], [9, 10], [5, 6]]))

    h5_array = np.array([[1, 2], [3, 4], [5, 6]])
    indices = np.array([0, 1, 0])
    values = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    put_non_unique(h5_array, indices, values, axis=1)
    np.testing.assert_allclose(h5_array, np.array([[7, 8], [10, 11], [13, 14]]))

    h5_array = np.array([[1, 2], [3, 4], [5, 6]])
    indices = np.array([0, 1])
    values = np.array([[7, 8], [10, 11], [13, 14]])
    put_non_unique(h5_array, indices, values, axis=1, indices_sorted=True)
    np.testing.assert_allclose(h5_array, np.array([[7, 8], [10, 11], [13, 14]]))


def test_transfer_from_casa():
    casa_file = '~/data/forward_modelling/data_dir/lwa01.ms'
    ms_folder = '~/data/forward_modelling/data_dir/lwa01_ms'
    ms = transfer_from_casa(
        ms_folder=ms_folder,
        casa_ms=casa_file,
        convention='casa'  # Or else UVW coordinates are very wrong.
    )
