import numpy as np
from astropy import coordinates as ac
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
    new_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "STOKES", "FREQ"]
    # Assuming you want to keep STOKES and FREQ as 1
    new_wcs.wcs.crval = [pointing_centre.ra.deg, pointing_centre.dec.deg, 1, 1]
    # Assuming data shape corresponds to [FREQ, STOKES, DEC, RA]
    new_wcs.wcs.crpix = [hdu.data.shape[3] / 2, hdu.data.shape[2] / 2, 1, 1]
    new_wcs.wcs.cdelt = [original_wcs.pixel_scale_matrix[1, 1], original_wcs.pixel_scale_matrix[0, 0], 1, 1]
    #
    # new_wcs = WCS(naxis=2)
    # new_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # new_wcs.wcs.crval = [pointing_centre.ra.deg, pointing_centre.dec.deg]
    # new_wcs.wcs.crpix = [hdu.data.shape[1] / 2, hdu.data.shape[0] / 2]  # Assuming 2D data
    # new_wcs.wcs.cdelt = np.array([original_wcs.pixel_scale_matrix[1, 1], original_wcs.pixel_scale_matrix[0, 0]])
    new_wcs.wcs.set()
    reprojected_data, _ = reproject_interp(hdu, new_wcs, shape_out=hdu.data.shape)
    hdu_new = fits.PrimaryHDU(data=reprojected_data, header=new_wcs.to_header())
    hdu_new.writeto(output_file, overwrite=True)
