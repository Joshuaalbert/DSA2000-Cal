import os

import numpy as np
import tables as tb
from astropy import time as at, coordinates as ac, units as au

from dsa2000_fm.measurement_sets.measurement_set import MeasurementSetMeta
from dsa2000_casa.adapter.from_casa_ms import transfer_from_casa
from dsa2000_casa.adapter.to_casa_ms import create_makems_config

try:
    from pyrap import tables as pt


    def test_transfer_from_casa(tmp_path):
        ms = transfer_from_casa(ms_folder=str(tmp_path / 'new_ms_folder'), casa_ms='visibilities.ms')
        print(ms)

        # Compare UVW with MS
        with pt.table('visibilities.ms') as t:
            uvw = t.getcol('UVW')[:]
            antenna1 = t.getcol('ANTENNA1')[:]
            antenna2 = t.getcol('ANTENNA2')[:]
            times_mjs = t.getcol('TIME')[:]
            times = at.Time(times_mjs / 86400., format='mjd', scale='utc')
            with tb.open_file(ms.data_file, 'r') as f:
                uvw_ms = f.root.uvw[:]
                antenna1_ms = f.root.antenna1[:]
                antenna2_ms = f.root.antenna2[:]
                time_idx_ms = f.root.time_idx[:]
                times_ms = ms.meta.times[time_idx_ms]

            print(uvw[:10])
            print(uvw_ms[:10])
            print("UVW Diff StdDev", np.std(uvw - uvw_ms))
            print("UVW Diff Max", np.max(np.abs(uvw - uvw_ms)))

            assert np.all(antenna1 == antenna1_ms)
            assert np.all(antenna2 == antenna2_ms)
            assert np.all((times - times_ms).sec < 1e-6)  # 1 us error
            assert np.std(uvw - uvw_ms) < 0.019  # 1.9 cm error
            assert np.max(np.abs(uvw - uvw_ms)) < 0.11  # 11 cm error


    def test_create_makems_config(tmp_path):
        meta = MeasurementSetMeta(
            array_name="test_array",
            array_location=ac.EarthLocation.from_geodetic(0 * au.deg, 0 * au.deg, 0 * au.m),
            phase_center=ac.ICRS(0 * au.deg, 0 * au.deg),
            channel_width=au.Quantity(1, au.Hz),
            integration_time=au.Quantity(1, au.s),
            coherencies=('XX', 'XY', 'YX', 'YY'),
            pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
            times=at.Time.now() + np.arange(10) * au.s,
            freqs=au.Quantity([1, 2, 3], au.Hz),
            antennas=ac.EarthLocation.from_geodetic(np.arange(5) * au.deg, np.arange(5) * au.deg, np.arange(5) * au.m),
            antenna_names=[f"antenna_{i}" for i in range(5)],
            antenna_diameters=au.Quantity(np.ones(5), au.m),
            with_autocorr=False,
            mount_types='ALT-AZ',
            ref_time=at.Time.now()
        )
        config_file = create_makems_config(
            casa_ms=str(tmp_path / 'test.ms'),
            meta=meta
        )
        assert os.path.exists(config_file)
        os.remove(config_file)
except ImportError:
    pass
