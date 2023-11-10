import os

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
from astropy import io
from astropy.io import fits
from astropy.wcs import WCS

from dsa2000_cal.assets.mocks.mock_data import MockData
from dsa2000_cal.faint_sky_model import repoint_fits, down_sample_fits, prepare_gain_fits, haversine, \
    nearest_neighbors_sphere


def test_repoint_fits():
    pointing_centre = ac.ICRS(ra=ac.Angle("19h59m28.356s"), dec=ac.Angle("+40d44m02.10s"))

    MockData()

    ds_output_file = "downsampled_mock_faint_sky-model.fits"

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
        assert not np.any(np.isnan(data))

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
        assert not np.any(np.isnan(data))

    # Cleanup
    os.remove(ds_output_file)
    os.remove(rp_output_file)


def test_basic_functionality():
    ra1, dec1 = np.radians(0), np.radians(0)
    ra2, dec2 = np.radians(0), np.radians(1)
    distance = haversine(ra1, dec1, ra2, dec2)
    assert np.isclose(distance, np.radians(1), rtol=1e-5)


def test_identical_points():
    ra1, dec1 = np.radians(45), np.radians(45)
    ra2, dec2 = np.radians(45), np.radians(45)
    distance = haversine(ra1, dec1, ra2, dec2)
    assert np.isclose(distance, 0, rtol=1e-5)


def test_array_of_coordinates():
    ra1 = np.radians([0, 0, 0])
    dec1 = np.radians([0, 1, 2])
    ra2 = np.radians([0, 0, 0])
    dec2 = np.radians([1, 2, 3])
    distances = haversine(ra1, dec1, ra2, dec2)
    expected = np.radians([1, 1, 1])
    assert np.allclose(distances, expected, rtol=1e-5)


def test_antipodal_points():
    ra1, dec1 = np.radians(0), np.radians(90)  # North pole
    ra2, dec2 = np.radians(0), np.radians(-90)  # South pole
    distance = haversine(ra1, dec1, ra2, dec2)
    assert np.isclose(distance, np.pi, rtol=1e-5)  # Should be half the circumference of the circle (pi radians)


def test_wrapping():
    # Define two points close in RA but on either side of the RA=0 boundary.
    coord1 = ac.ICRS(ra=359.5 * au.degree, dec=0 * au.degree)
    coord2 = ac.ICRS(ra=0.5 * au.degree, dec=0 * au.degree)

    # Expected separation is 1 degree, but we have to handle the RA wrapping.
    expected_distance = np.radians(1)  # Convert 1 degree to radians

    distance = haversine(coord1.ra.rad, coord1.dec.rad, coord2.ra.rad, coord2.dec.rad)
    assert np.isclose(distance, expected_distance, rtol=1e-5)


def test_nearest_neighbor():
    coords1 = ac.ICRS([10, 20] * au.degree, [-45, 45] * au.degree)
    coords2 = ac.ICRS([11, 21, 10.5, 19.5] * au.degree, [-44, 46, -45.5, 44.5] * au.degree)

    nearest_indices, distances = nearest_neighbors_sphere(coords1, coords2)

    # For the given example:
    # - The nearest neighbor to (10°, -45°) is (10.5°, -45.5°)
    # - The nearest neighbor to (20°, 45°) is (19.5°, 44.5°)
    expected_indices = np.array([2, 3])
    # Convert angles to radians for haversine calculations
    ra1_rad, dec1_rad = np.radians(10), np.radians(-45)
    ra2_rad, dec2_rad = np.radians(10.5), np.radians(-45.5)
    ra3_rad, dec3_rad = np.radians(20), np.radians(45)
    ra4_rad, dec4_rad = np.radians(19.5), np.radians(44.5)

    # Calculate expected distances
    expected_distances = np.array([
        haversine(ra1_rad, dec1_rad, ra2_rad, dec2_rad),
        haversine(ra3_rad, dec3_rad, ra4_rad, dec4_rad)
    ])
    assert np.array_equal(nearest_indices, expected_indices)
    assert np.allclose(distances, expected_distances, rtol=1e-5)


def mock_data():
    pointing_centre = ac.ICRS(ra=80 * au.degree, dec=0 * au.degree)
    # [num_time, num_ant, num_dir, num_freq, 2, 2]
    gains = np.random.randn(10, 5, 3, 7, 2, 2) + 1j * np.random.randn(10, 5, 3, 7, 2, 2)
    directions = ac.concatenate(
        [ac.ICRS(ra=ra * au.degree, dec=np.random.normal() * au.degree) for ra in [178, 180, 182]])
    freq_hz = np.linspace(700e6, 1400e6, 7)
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

    # Open the FITS file
    with fits.open(output_file) as hdul:
        # Get the header of the primary HDU (or specify another index if needed)
        header = hdul[0].header

        # Print the entire header
        print(header)



def test_fits_file_content(tmp_path):
    pointing_centre, gains, directions, freq_hz, times, num_pix = mock_data()
    output_file = "test_output.fits"

    prepare_gain_fits(output_file, pointing_centre, gains, directions, freq_hz, times, num_pix)

    with io.fits.open(output_file) as hdu_list:
        header = hdu_list[0].header
        print("\n".join([f"{k} = {header[k]}" for k in header]))
        data = hdu_list[0].data

        # Check if WCS data in header is as expected
        assert header['CTYPE1'] == 'RA---SIN'
        assert header['CTYPE2'] == 'DEC--SIN'
        assert header['BITPIX'] == -32
        # ... add other checks

        # Check the shape of the data
        assert data.shape == (num_pix, num_pix, 4, 5, 7, 10)

        # Check some statistics on the data if necessary
        assert data.mean() != 0
        # ... add other checks

        assert not np.any(np.isnan(data))
