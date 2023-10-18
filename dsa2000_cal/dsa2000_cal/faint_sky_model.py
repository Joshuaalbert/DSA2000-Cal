import astropy.time as at
import numpy as np
from astropy import coordinates as ac, io, wcs
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy.interpolate import griddata
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
    # Determine the range of RA and DEC and pad by 20%
    ra_values = [dir.ra.deg for dir in directions]
    dec_values = [dir.dec.deg for dir in directions]
    ra_range = max(ra_values) - min(ra_values)
    dec_range = max(dec_values) - min(dec_values)

    ra_padding = ra_range * 0.2
    dec_padding = dec_range * 0.2

    ra_min = min(ra_values) - ra_padding
    ra_max = max(ra_values) + ra_padding
    dec_min = min(dec_values) - dec_padding
    dec_max = max(dec_values) + dec_padding

    # Create the new WCS
    w = wcs.WCS(naxis=6)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "MATRIX", "ANTENNA", "FREQ", "TIME"]
    w.wcs.crpix = [num_pix / 2, num_pix / 2, 1, 1, 1, 1]
    w.wcs.cdelt = [(ra_max - ra_min) / num_pix, (dec_max - dec_min) / num_pix, 1, 1, freq_hz[1] - freq_hz[0],
                   (times[1].mjd - times[0].mjd) * 86400]
    w.wcs.crval = [pointing_centre.ra.deg, pointing_centre.dec.deg, 1, 1, freq_hz[0], times[0].mjd * 86400]

    # Extract pixel directions from the WCS as ICRS coordinates
    x = np.linspace(ra_min, ra_max, num_pix)
    y = np.linspace(dec_min, dec_max, num_pix)
    x, y = np.meshgrid(x, y)
    ra, dec, _, _, _, _ = w.all_pix2world(x, y, 1, 1, 1, 1, 0)
    coords = ac.SkyCoord(ra, dec, frame='icrs', unit='deg')

    # Grid the gains using nearest method
    grid_points = [(dir.ra.deg, dir.dec.deg) for dir in directions]

    # Split gains into real and imaginary parts
    gains_real = np.real(gains)
    gains_imag = np.imag(gains)

    values_real = gains_real.reshape(-1, 2, 2)
    values_imag = gains_imag.reshape(-1, 2, 2)

    grid_gains_real = griddata(grid_points, values_real, (coords.ra.deg, coords.dec.deg), method='nearest').reshape(
        num_pix, num_pix, len(gains), 2, 2)
    grid_gains_imag = griddata(grid_points, values_imag, (coords.ra.deg, coords.dec.deg), method='nearest').reshape(
        num_pix, num_pix, len(gains), 2, 2)

    # Prepare data for FITS
    data = np.zeros((num_pix, num_pix, 4, len(gains), len(freq_hz), len(times)))

    # Assign real and imaginary parts of XX and YY to appropriate positions in the data array
    data[:, :, 0, :, :, :] = grid_gains_real[:, :, :, :, 0, 0]
    data[:, :, 1, :, :, :] = grid_gains_imag[:, :, :, :, 0, 0]
    data[:, :, 2, :, :, :] = grid_gains_real[:, :, :, :, 1, 1]
    data[:, :, 3, :, :, :] = grid_gains_imag[:, :, :, :, 1, 1]

    # Store the gains in the fits file
    hdu = io.fits.PrimaryHDU(data, header=w.to_header())
    hdu.writeto(output_file, overwrite=True)
