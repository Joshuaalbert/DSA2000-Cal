import os

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
from astropy import io
from astropy.io import fits
from astropy.wcs import WCS

from dsa2000_cal.assets.mocks.mock_data import MockData
from dsa2000_cal.faint_sky_model import repoint_fits, down_sample_fits, prepare_gain_fits


def test_repoint_fits():
    pointing_centre = ac.ICRS(ra=ac.Angle("19h59m28.356s"), dec=ac.Angle("+40d44m02.10s"))

    MockData()

    ds_output_file = "down_sampled_faint_sky_model.fits"

    down_sample_fits(
        fits_file=MockData().faint_sky_model(),
        output_file=ds_output_file,
        desired_ra_size=128,
        desired_dec_size=128
    )

    # Check the dimensions of the output file
    with fits.open(ds_output_file) as hdu:
        data = hdu[0].data
        assert data.shape[2:] == (128, 128)

    rp_output_file = "repointed_down_sampled_faint_sky_model.fits"
    repoint_fits(
        fits_file=ds_output_file,
        output_file=rp_output_file,
        pointing_centre=pointing_centre
    )

    # Check if the output file is created
    assert os.path.exists(rp_output_file)

    # Check the pointing center of the output file
    with fits.open(rp_output_file) as hdu:
        wcs = WCS(hdu[0].header)
        assert np.isclose(wcs.wcs.crval[0], pointing_centre.ra.deg, atol=1e-6)
        assert np.isclose(wcs.wcs.crval[1], pointing_centre.dec.deg, atol=1e-6)

    # Cleanup
    os.remove(ds_output_file)
    os.remove(rp_output_file)


def mock_data():
    pointing_centre = ac.ICRS(ra=180 * au.degree, dec=0 * au.degree)
    gains = np.random.randn(10, 5, 3, 4, 2, 2) + 1j * np.random.randn(10, 5, 3, 4, 2, 2)
    directions = ac.concatenate([ac.ICRS(ra=ra * au.degree, dec=0 * au.degree) for ra in [178, 180, 182]])
    freq_hz = np.array([1e8, 1.1e8, 1.2e8, 1.3e8])
    times = at.Time(['2023-10-18T00:00:00', '2023-10-18T01:00:00', '2023-10-18T02:00:00', '2023-10-18T03:00:00',
                     '2023-10-18T04:00:00', '2023-10-18T05:00:00', '2023-10-18T06:00:00', '2023-10-18T07:00:00',
                     '2023-10-18T08:00:00', '2023-10-18T09:00:00'])
    num_pix = 32
    return pointing_centre, gains, directions, freq_hz, times, num_pix


def test_output_file_created(tmp_path):
    pointing_centre, gains, directions, freq_hz, times, num_pix = mock_data()
    output_file = tmp_path / "test_output.fits"

    prepare_gain_fits(output_file, pointing_centre, gains, directions, freq_hz, times, num_pix)

    assert output_file.exists()


def test_fits_file_content(tmp_path):
    pointing_centre, gains, directions, freq_hz, times, num_pix = mock_data()
    output_file = tmp_path / "test_output.fits"

    prepare_gain_fits(output_file, pointing_centre, gains, directions, freq_hz, times, num_pix)

    with io.fits.open(output_file) as hdu_list:
        header = hdu_list[0].header
        data = hdu_list[0].data

        # Check if WCS data in header is as expected
        assert header['CTYPE1'] == 'RA---SIN'
        assert header['CTYPE2'] == 'DEC--SIN'
        # ... add other checks

        # Check the shape of the data
        assert data.shape == (num_pix, num_pix, 4, 10, 4, 10)

        # Check some statistics on the data if necessary
        assert data.mean() != 0
        # ... add other checks
