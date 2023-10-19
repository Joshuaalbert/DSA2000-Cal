from typing import List, Tuple

import astropy.time as at
import astropy.units as au
import numpy as np
from astropy import coordinates as ac, io, wcs
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy.ndimage import zoom


def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    naxis = f[0].header['NAXIS']
    if naxis < 2:
        raise ValueError('Can\'t make map from this')
    if naxis == 2:
        return fits.PrimaryHDU(header=f[0].header, data=f[0].data)

    w = WCS(f[0].header)
    wn = WCS(naxis=2)

    wn.wcs.crpix[0] = w.wcs.crpix[0]
    wn.wcs.crpix[1] = w.wcs.crpix[1]
    wn.wcs.cdelt = w.wcs.cdelt[0:2]
    wn.wcs.crval = w.wcs.crval[0:2]
    wn.wcs.ctype[0] = w.wcs.ctype[0]
    wn.wcs.ctype[1] = w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"] = 2
    copy = ('EQUINOX', 'EPOCH', 'BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'TELESCOP', 'OBSERVER')
    for k in copy:
        r = f[0].header.get(k)
        if r is not None:
            header[k] = r

    slice = []
    for i in range(naxis, 0, -1):
        if i <= 2:
            slice.append(np.s_[:], )
        else:
            slice.append(0)

    hdu = fits.PrimaryHDU(header=header, data=f[0].data[tuple(slice)])
    return hdu


def down_sample_fits(fits_file: str, output_file: str, desired_ra_size: int, desired_dec_size: int):
    """
    Down-samples the fits file to the desired size.

    Args:
        fits_file: the path to the fits file
        output_file: the path to the output fits file
        desired_ra_size: the desired RA size
        desired_dec_size: the desired DEC size
    """
    hdu = fits.open(fits_file)[0]
    data = hdu.data
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaNs")
    output = np.zeros((data.shape[0], data.shape[1], desired_dec_size, desired_ra_size), dtype=data.dtype)
    ra_downsample_ratio = desired_ra_size / data.shape[3]
    dec_downsample_ratio = desired_dec_size / data.shape[2]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Assuming data shape corresponds to [FREQ, STOKES, DEC, RA]
            data_2d = data[i, j, :, :]  # Example for the first frequency and first Stokes parameter
            zoom(data_2d, (dec_downsample_ratio, ra_downsample_ratio), output=output[i, j, :, :])

    original_wcs = WCS(hdu.header)  # .celestial ensures you only get RA and DEC part of the WCS
    new_wcs = original_wcs.deepcopy()

    new_wcs.wcs.cdelt[0] /= ra_downsample_ratio
    new_wcs.wcs.cdelt[1] /= dec_downsample_ratio
    new_wcs.wcs.crpix[0] *= ra_downsample_ratio
    new_wcs.wcs.crpix[1] *= dec_downsample_ratio

    hdu_new = fits.PrimaryHDU(data=output, header=new_wcs.to_header())
    hdu_new.writeto(output_file, overwrite=True)


def repoint_fits(fits_file: str, output_file: str, pointing_centre: ac.ICRS):
    """
    Re-points the fits file to the pointing centre.

    Args:
        fits_file: the path to the fits file
        output_file: the path to the output fits file
        pointing_centre: the pointing centre in ICRS coordinates
    """
    # Load the FITS file
    hdu = fits.open(fits_file)[0]
    original_wcs = WCS(hdu.header)

    # Create a new WCS based on the desired center direction
    new_wcs = WCS(naxis=4)
    new_wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
    # Assuming you want to keep STOKES and FREQ as 1
    new_wcs.wcs.crval = [pointing_centre.ra.deg, pointing_centre.dec.deg, 1, 1]
    # Assuming data shape corresponds to [FREQ, STOKES, DEC, RA]
    new_wcs.wcs.crpix = [hdu.data.shape[3] / 2, hdu.data.shape[2] / 2, 1, 1]
    new_wcs.wcs.cdelt = [original_wcs.pixel_scale_matrix[1, 1], original_wcs.pixel_scale_matrix[0, 0], 1, 1]

    new_wcs.wcs.set()
    reprojected_data, _ = reproject_interp(hdu, new_wcs, shape_out=hdu.data.shape)
    hdu_new = fits.PrimaryHDU(data=reprojected_data, header=new_wcs.to_header())
    hdu_new.writeto(output_file, overwrite=True)


def haversine(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    """
    Calculate the great circle separation between two points on the sky using the haversine formula.

    Args:
        ra1: RA (or longitude) of the first point in radians
        dec1: DEC (or latitude) of the first point in radians
        ra2: RA (or longitude) of the second point in radians
        dec2: DEC (or latitude) of the second point in radians

    Returns:
        The great circle separation in radians
    """
    delta_ra = np.abs(ra2 - ra1)
    delta_ra = np.where(delta_ra > np.pi, 2 * np.pi - delta_ra, delta_ra)
    delta_dec = dec2 - dec1

    a = np.sin(delta_dec / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(delta_ra / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return c


def nearest_neighbors_sphere(coords1: ac.ICRS, coords2: ac.ICRS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the nearest neighbors on a sphere.

    Args:
        coords1: the coordinates [M]
        coords2: set of coordinates to search through [N]

    Returns:
        The nearest indices [M] and distances [M]
    """
    # Extract RA and Dec, and convert them to radians
    ra1 = coords1.ra.rad
    dec1 = coords1.dec.rad
    ra2 = coords2.ra.rad
    dec2 = coords2.dec.rad

    # For each coordinate in coords1 find the nearest in coords2
    dists = haversine(ra1[:, None], dec1[:, None], ra2[None, :], dec2[None, :])
    nearest_indices = np.argmin(dists, axis=1)
    distances = dists[np.arange(len(nearest_indices)), nearest_indices]

    return nearest_indices, distances


def prepare_gain_fits(output_file: str, pointing_centre: ac.ICRS,
                      gains: np.ndarray, directions: ac.ICRS,
                      freq_hz: np.ndarray, times: at.Time, num_pix: int):
    """
    Given an input gain array, prepare a gain fits file for use with a-term correction in WSClean.
    Axes should be [RA, DEC, MATRIX, ANTENNA, FREQ, TIME]
    MATRIX has 4 components: {real XX, imaginary XX, real YY, imaginary YY}
    Args:
        output_file: the path to the output fits file
        pointing_centre: the pointing centre
        gains: the gain array [num_time, num_ant, num_dir, num_freq, 2, 2]
        directions: the directions [num_dir]
        freq_hz: the frequencies [num_freq]
        times: the times [num_time]
        num_pix: the screen resolution in pixels
    References:
        [1] https://wsclean.readthedocs.io/en/latest/a_term_correction.html#diagonal-gain-correction
    """
    Nt, Na, Nd, Nf, _, _ = gains.shape
    if len(directions) != Nd:
        raise ValueError("Number of directions does not match gains")
    if len(freq_hz) != Nf:
        raise ValueError("Number of frequencies does not match gains")
    if len(times) != Nt:
        raise ValueError("Number of times does not match gains")

    # Determine the range of RA and DEC and pad by 20%
    ra_values = [dir.ra.deg for dir in directions]
    dec_values = [dir.dec.deg for dir in directions]
    ra_range = max(ra_values) - min(ra_values)
    dec_range = max(dec_values) - min(dec_values)
    if ra_range == 0:
        raise ValueError("RA range is 0")
    if dec_range == 0:
        raise ValueError("DEC range is 0")

    ra_padding = ra_range * 0.2
    dec_padding = dec_range * 0.2

    ra_min = min(ra_values) - ra_padding
    ra_max = max(ra_values) + ra_padding
    dec_min = min(dec_values) - dec_padding
    dec_max = max(dec_values) + dec_padding

    # Create the new WCS
    w = wcs.WCS(naxis=6)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "MATRIX", "ANTENNA", "FREQ", "TIME"]
    w.wcs.crpix = [num_pix // 2, num_pix // 2, 1, 1, 1, 1]
    w.wcs.cdelt = [(ra_max - ra_min) / num_pix, (dec_max - dec_min) / num_pix, 1, 1, freq_hz[1] - freq_hz[0],
                   (times[1].mjd - times[0].mjd) * 86400]
    w.wcs.crval = [pointing_centre.ra.deg, pointing_centre.dec.deg, 1, 1, freq_hz[0], times[0].mjd * 86400]
    w.wcs.set()
    # Extract pixel directions from the WCS as ICRS coordinates
    x = np.arange(num_pix)
    y = np.arange(num_pix)
    X = np.meshgrid(x, y, np.arange(1), np.arange(1), np.arange(1), np.arange(1), indexing='ij')
    coords_pix = np.stack([x.flatten() for x in X], axis=1)
    coords_world = w.all_pix2world(coords_pix, 0)
    ra = coords_world[:, 0]
    dec = coords_world[:, 1]

    coords = ac.ICRS(ra=ra * au.deg, dec=dec * au.deg)
    nn_indices, nn_dist = nearest_neighbors_sphere(coords1=coords, coords2=directions)

    # Split gains into real and imaginary parts
    gains = np.transpose(gains, (2, 1, 3, 0, 4, 5))  # [num_dir, num_ant, num_freq, num_time, 2, 2]
    gains_real = np.real(gains)
    gains_imag = np.imag(gains)

    # Prepare data for FITS
    data = np.zeros((num_pix, num_pix, 4, Na, Nf, Nt))

    for (i, j), nn_idx in zip(coords_pix[:, :2], nn_indices):
        # Assign real and imaginary parts of XX and YY to appropriate positions in the data array
        data[i, j, 0, :, :, :] = gains_real[nn_idx, :, :, :, 0, 0]
        data[:, :, 1, :, :, :] = gains_imag[nn_idx, :, :, :, 0, 0]
        data[:, :, 2, :, :, :] = gains_real[nn_idx, :, :, :, 1, 1]
        data[:, :, 3, :, :, :] = gains_imag[nn_idx, :, :, :, 1, 1]

    # Store the gains in the fits file
    hdu = io.fits.PrimaryHDU(data, header=w.to_header())
    hdu.writeto(output_file, overwrite=True)


def write_diagonal_a_term_correction_file(a_term_file: str, diagonal_gain_fits_files: List[str]):
    """
    Write the diagonal a-term correction file.

    Args:
        a_term_file: the path to the a-term file
        diagonal_gain_fits_files: the paths to the diagonal gain fits files
    """
    with open(a_term_file, 'w') as f:
        # The fourierfit, klfit aterms are new since 3.2
        f.write("aterms = [ diagonal ]\n")
        # The diagonal correction has parameters 'images' and 'window'.
        # This fits must match what's in the run config.
        f.write(f"diagonal.images = [ {' '.join(diagonal_gain_fits_files)} ]\n")
        # The window parameter is new since 2.7
        # It supports tukey, hann, raised_hann, rectangular or gaussian.
        # If not specified, raised_hann is used, which generally performs best.
        f.write("diagonal.window = raised_hann")
