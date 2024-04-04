import astropy.time as at
import numpy as np
import pyrap.tables as pt
import tables as tb

from dsa2000_cal.adapter.from_casa_ms import transfer_from_casa


def test_transfer_from_casa(tmp_path):
    ms = transfer_from_casa(ms_folder=str(tmp_path / 'new_ms_folder'), casa_ms='visibilities.ms')
    print(ms)

    # Compare UVW with MS
    with pt.table('visibilities.ms') as t:
        uvw = t.getcol('UVW')[:]
        antenna_1 = t.getcol('ANTENNA1')[:]
        antenna_2 = t.getcol('ANTENNA2')[:]
        times_mjs = t.getcol('TIME')[:]
        times = at.Time(times_mjs / 86400., format='mjd', scale='utc')
        with tb.open_file(ms.data_file, 'r') as f:
            uvw_ms = f.root.uvw[:]
            antenna_1_ms = f.root.antenna_1[:]
            antenna_2_ms = f.root.antenna_2[:]
            time_idx_ms = f.root.time_idx[:]
            times_ms = ms.meta.times[time_idx_ms]

        print(uvw[:10])
        print(uvw_ms[:10])
        print("UVW Diff StdDev", np.std(uvw - uvw_ms))
        print("UVW Diff Max", np.max(np.abs(uvw - uvw_ms)))

        assert np.all(antenna_1 == antenna_1_ms)
        assert np.all(antenna_2 == antenna_2_ms)
        assert np.all((times - times_ms).sec < 1e-6)  # 1 us error
        assert np.std(uvw - uvw_ms) < 0.019  # 1.9 cm error
        assert np.max(np.abs(uvw - uvw_ms)) < 0.11  # 11 cm error
