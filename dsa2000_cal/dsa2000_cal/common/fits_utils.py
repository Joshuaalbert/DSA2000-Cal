import logging
import os
from typing import List, Tuple, Literal

import astropy.time as at
import astropy.units as au
import numpy as np
from astropy import coordinates as ac, io, wcs
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import zoom

from dsa2000_cal.common.quantity_utils import quantity_to_np
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel

logger = logging.getLogger(__name__)


def flatten(f):
    """
    Flatten a FITS file to a 2D image, by dropping all axes except the first two.

    Args:
        f: the FITS file

    Returns:
        the flattened image
    """

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


def transform_to_wsclean_model(fits_file: str, output_file: str, pointing_centre: ac.ICRS, ref_freq_hz: float,
                               bandwidth_hz: float):
    """
    Re-points the fits file to the pointing centre, and transposes axes to match the output of wsclean.

    Note, the projection might change, but we don't take that into account.

    Args:
        fits_file: the path to the fits file of an image to turn into a model
        output_file: the path to the output fits file
        pointing_centre: the pointing centre in ICRS coordinates
        ref_freq_hz: reference frequency in Hz (lowest bound of bandwidth)
        bandwidth_hz: the bandwidth of model
    """
    # Load the FITS file
    with fits.open(fits_file) as hdu:

        original_wcs = WCS(hdu[0].header)
        # new_wcs = original_wcs.deepcopy()

        # Create a new WCS based on the desired center direction
        new_wcs = WCS(naxis=4)
        new_wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
        # Get permutation of axes
        perm = []
        for ctype in new_wcs.wcs.ctype:
            if ctype not in original_wcs.wcs.ctype:
                raise ValueError(f"Could not find {ctype} in {fits_file}")
            perm.append(list(original_wcs.wcs.ctype).index(ctype))
        # Apply perm. Note: because python is column-major we need to reverse the perm
        # print(perm)

        data = np.transpose(hdu[0].data.T, perm).T.copy()  # [Ns, Nf, Ndec, Nra]
        # print(data.shape)
        Ns, Nf, Ndec, Nra = data.shape
        new_wcs.wcs.crval = [pointing_centre.ra.deg, pointing_centre.dec.deg, ref_freq_hz, 1]
        new_wcs.wcs.crpix = [Nra / 2 + 1, Ndec / 2 + 1, 1, 1]
        new_wcs.wcs.cdelt = [
            original_wcs.pixel_scale_matrix[0, 0],
            original_wcs.pixel_scale_matrix[1, 1],
            bandwidth_hz,
            1
        ]
        new_wcs.wcs.cunit = [
            'deg',
            'deg',
            'Hz',
            ''
        ]
        new_wcs.wcs.set()
        # reprojected_data, _ = reproject_interp(hdu[0], new_wcs, shape_out=hdu[0].data.shape)
        hdu_new = fits.PrimaryHDU(data=data, header=new_wcs.to_header())
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

    ra_padding = ra_range * 0.2
    dec_padding = dec_range * 0.2
    if ra_range == 0:
        ra_padding = 1.
        print("RA range is 0, padding by 1 deg")
    if dec_range == 0:
        dec_padding = 1.
        print("DEC range is 0, padding by 1 deg")

    ra_min = min(ra_values) - ra_padding
    ra_max = max(ra_values) + ra_padding
    dec_min = min(dec_values) - dec_padding
    dec_max = max(dec_values) + dec_padding

    # Create the new WCS
    w = wcs.WCS(naxis=6)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "MATRIX", "ANTENNA", "FREQ", "TIME"]
    w.wcs.crpix = [num_pix / 2 + 1, num_pix / 2 + 1, 1, 1, 1, 1]
    w.wcs.cdelt = [-(ra_max - ra_min) / num_pix, (dec_max - dec_min) / num_pix, 1, 1, freq_hz[1] - freq_hz[0],
                   (times[1].mjd - times[0].mjd) * 86400]
    w.wcs.crval = [pointing_centre.ra.deg, pointing_centre.dec.deg, 1, 1, freq_hz[0], times[0].mjd * 86400]
    w.wcs.set()
    # Extract pixel directions from the WCS as ICRS coordinates
    x = np.arange(-num_pix / 2, num_pix / 2) + 1
    y = np.arange(-num_pix / 2, num_pix / 2) + 1
    X = np.meshgrid(x, y, np.arange(1), np.arange(1), np.arange(1), np.arange(1), indexing='ij')
    coords_pix = np.stack([x.flatten() for x in X], axis=1)
    coords_world = w.all_pix2world(coords_pix, 0)
    ra = coords_world[:, 0]
    dec = coords_world[:, 1]

    coords = ac.ICRS(ra=ra * au.deg, dec=dec * au.deg)
    nn_indices, nn_dist = nearest_neighbors_sphere(coords1=coords, coords2=directions)

    # Prepare data for FITS: row-major to transposed column-major
    data = np.zeros((Nt, Nf, Na, 4, num_pix, num_pix), dtype=np.float32)
    # [num_time, num_ant, num_dir, num_freq, 2, 2] -> [num_time, num_freq, num_ant, 2, 2, num_dir]
    gains = np.transpose(gains, (0, 3, 1, 4, 5, 2))

    # Split gains into real and imaginary parts
    gains_real = np.real(gains).astype(np.float32)
    gains_imag = np.imag(gains).astype(np.float32)

    for (i, j), nn_idx in zip(coords_pix[:, :2], nn_indices):
        i = int(i + num_pix / 2 - 1)
        j = int(j + num_pix / 2 - 1)
        # Assign real and imaginary parts of XX and YY to appropriate positions in the data array
        data[:, :, :, 0, j, i] = gains_real[:, :, :, 0, 0, nn_idx]
        data[:, :, :, 1, j, i] = gains_imag[:, :, :, 0, 0, nn_idx]
        data[:, :, :, 2, j, i] = gains_real[:, :, :, 1, 1, nn_idx]
        data[:, :, :, 3, j, i] = gains_imag[:, :, :, 1, 1, nn_idx]

    # Store the gains in the fits file
    hdu = io.fits.PrimaryHDU(data, header=w.to_header())
    hdu.header['EXTEND'] = True
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


class ImageModel(SerialisableBaseModel):
    phase_tracking: ac.ICRS
    obs_time: at.Time
    dl: au.Quantity
    dm: au.Quantity
    freqs: au.Quantity  # [num_freqs]
    bandwidth: au.Quantity
    coherencies: List[Literal[
        'XX', 'XY', 'YX', 'YY',
        'I', 'Q', 'U', 'V',
        'RR', 'RL', 'LR', 'LL'
    ]]  # [num_coherencies]
    beam_major: au.Quantity
    beam_minor: au.Quantity
    beam_pa: au.Quantity
    unit: Literal['JY/BEAM', 'JY/PIXEL'] = 'JY/PIXEL'
    object_name: str = 'undefined'
    image: au.Quantity  # [num_l, num_m, num_freqs, num_coherencies]

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(ImageModel, self).__init__(**data)
        # Use _check_measurement_set_meta_v0 as instance-wise validator
        _check_image_model(self)

    def save_image_to_fits(self, file_path: str, overwrite: bool = False):
        """
        Saves an image to FITS using SIN projection.

        Args:
            file_path: the path to the output FITS file
            overwrite: whether to overwrite the file if it already exists
        """
        save_image_to_fits(file_path=file_path, image_model=self, overwrite=overwrite)


def _check_image_model(image_model: ImageModel):
    # Check units
    if not image_model.dl.unit.is_equivalent(au.dimensionless_unscaled):
        raise ValueError("dl must be in dimensionless_unscaled")
    if not image_model.dm.unit.is_equivalent(au.dimensionless_unscaled):
        raise ValueError("dm must be in dimensionless_unscaled")
    if not image_model.freqs.unit.is_equivalent(au.Hz):
        raise ValueError("freqs must be in Hz")
    if not image_model.bandwidth.unit.is_equivalent(au.Hz):
        raise ValueError("bandwidth must be in Hz")
    if not image_model.image.unit.is_equivalent(au.Jy):
        raise ValueError("image must be in Jy")
    if not image_model.beam_major.unit.is_equivalent(au.deg):
        raise ValueError("beam_major must be in degrees")
    if not image_model.beam_minor.unit.is_equivalent(au.deg):
        raise ValueError("beam_minor must be in degrees")
    if not image_model.beam_pa.unit.is_equivalent(au.deg):
        raise ValueError("beam_pa must be in degrees")
    # Check shapes
    if not image_model.phase_tracking.isscalar:
        raise ValueError("phase_tracking must be scalar")
    if not image_model.dl.isscalar:
        raise ValueError("dl must be scalar")
    if not image_model.dm.isscalar:
        raise ValueError("dm must be scalar")
    if not image_model.freqs.isscalar:
        image_model.freqs = image_model.freqs.reshape((-1,))
    if image_model.image.isscalar:
        raise ValueError("image must be a 2D array")
    if not image_model.beam_major.isscalar:
        raise ValueError("beam_major must be scalar")
    if not image_model.beam_minor.isscalar:
        raise ValueError("beam_minor must be scalar")
    if not image_model.beam_pa.isscalar:
        raise ValueError("beam_pa must be scalar")
    if len(image_model.image.shape) != 4:
        raise ValueError(f"image shape must be (num_l, num_m, num_freqs, num_coherencies), "
                         f"got {image_model.image.shape}")
    if image_model.image.shape[2] != len(image_model.freqs):
        raise ValueError(f"num_freqs must match image[2] shape, "
                         f"got {image_model.image.shape[2]} != {len(image_model.freqs)}")
    if image_model.image.shape[3] != len(image_model.coherencies):
        raise ValueError(f"num_coherencies must match image[3] shape, "
                         f"got {image_model.image.shape[3]} != {len(image_model.coherencies)}")
    if image_model.coherencies not in [
        ['XX', 'XY', 'YX', 'YY'],
        ['I', 'Q', 'U', 'V'],
        ['RR', 'RL', 'LR', 'LL'],
        ['I']
    ]:
        raise ValueError(f"coherencies format {image_model.coherencies} is invalid.")
    # Ensure freqs are uniformly spaced
    dfreq = np.diff(image_model.freqs.to(au.Hz).value)
    if len(dfreq) > 0 and not np.allclose(dfreq, dfreq[0], atol=1e-6):
        raise ValueError("freqs must be uniformly spaced")
    # dl is negative
    if image_model.dl.value <= 0:
        raise ValueError("dl is always positive.")
    # dm is positive
    if image_model.dm.value <= 0:
        raise ValueError("dm is always positive.")
    # Check beam
    if image_model.beam_major.value <= 0:
        raise ValueError("beam_major must be positive")
    if image_model.beam_minor.value <= 0:
        raise ValueError("beam_minor must be positive")


def save_image_to_fits(file_path: str, image_model: ImageModel, overwrite: bool = False):
    """
    Saves an image to FITS using SIN projection.

    Args:
        file_path: the path to the output FITS file
        image_model: the image model

    Raises:
        FileExistsError: if the file already exists and overwrite is False
    """

    # SIMPLE  =                    T / file does conform to FITS standard
    # BITPIX  =                  -32 / number of bits per data pixel
    # NAXIS   =                    4 / number of data axes
    # NAXIS1  =                 1300 / length of data axis 1
    # NAXIS2  =                 1300 / length of data axis 2
    # NAXIS3  =                    1 / length of data axis 3
    # NAXIS4  =                    1 / length of data axis 4
    # EXTEND  =                    T / FITS dataset may contain extensions
    # COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy
    # COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H
    # BSCALE  =                   1.
    # BZERO   =                   0.
    # BUNIT   = 'JY/BEAM '           / Units are in Jansky per beam
    # BMAJ    =  0.00264824850407388
    # BMIN    =  0.00205277482609264
    # BPA     =     88.7325624575301
    # EQUINOX =                2000. / J2000
    # BTYPE   = 'Intensity'
    # TELESCOP= 'LOFAR   '
    # OBSERVER= 'unknown '
    # OBJECT  = 'BEAM_0  '
    # ORIGIN  = 'WSClean '           / W-stacking imager written by Andre Offringa
    # CTYPE1  = 'RA---SIN'           / Right ascension angle cosine
    # CRPIX1  =                 651.
    # CRVAL1  =    -9.13375000000008
    # CDELT1  = -0.000555555555555556
    # CUNIT1  = 'deg     '
    # CTYPE2  = 'DEC--SIN'           / Declination angle cosine
    # CRPIX2  =                 651.
    # CRVAL2  =        58.8117777778
    # CDELT2  = 0.000555555555555556
    # CUNIT2  = 'deg     '
    # CTYPE3  = 'FREQ    '           / Central frequency
    # CRPIX3  =                   1.
    # CRVAL3  =     53807067.8710938
    # CDELT3  =            47656250.
    # CUNIT3  = 'Hz      '
    # CTYPE4  = 'STOKES  '
    # CRPIX4  =                   1.
    # CRVAL4  =                   1.
    # CDELT4  =                   1.
    # CUNIT4  = '        '
    # SPECSYS = 'TOPOCENT'
    # DATE-OBS= '2015-08-26T16:30:05.0'
    # END
    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(f"File {file_path} already exists")

    # Create the WCS
    w = wcs.WCS(naxis=4)  # 4D image [l, m, freq, coherency]
    w.wcs.ctype = ["RA--SIN", "DEC-SIN", "FREQ", "STOKES"]
    if (np.shape(image_model.image)[0] % 2 == 1) or (np.shape(image_model.image)[1] % 2 == 1):
        raise ValueError("Image must have an even number of pixels in the direction")
    w.wcs.crpix = [image_model.image.shape[0] / 2 + 1, image_model.image.shape[1] / 2 + 1, 1, 1]
    dfreq = image_model.bandwidth / len(image_model.freqs)
    w.wcs.cdelt = [
        -image_model.dl.value * 180. / np.pi,
        image_model.dm.value * 180. / np.pi,
        dfreq.to('Hz').value,
        1
    ]
    w.wcs.crval = [
        image_model.phase_tracking.ra.deg,
        image_model.phase_tracking.dec.deg,
        image_model.freqs[0].to('Hz').value,
        1
    ]
    w.wcs.cunit = ['deg', 'deg', 'Hz', '']
    w.wcs.set()
    # Set beam
    header = w.to_header()
    header["BUNIT"] = image_model.unit
    header["BSCALE"] = 1
    header["BZERO"] = 0
    header["BTYPE"] = "Intensity"
    header["BMAJ"] = image_model.beam_major.to(au.deg).value
    header["BMIN"] = image_model.beam_minor.to(au.deg).value
    header["BPA"] = image_model.beam_pa.to(au.deg).value
    header["TELESCOP"] = "DSA2000"
    header["OBSERVER"] = "unknown"
    header["OBJECT"] = image_model.object_name
    header["RADESYS"] = 'ICRS'
    header["SPECSYS"] = "TOPOCENT"
    header["MJD-OBS"] = image_model.obs_time.mjd

    # Save the image
    data = quantity_to_np(image_model.image, 'Jy')  # [num_l, num_m, num_freqs, num_coherencies]
    # Reverse l axis
    data = data[::-1, :, :, :]
    # Transpose data
    data = data.T  # [num_coherencies, num_freqs, num_m, num_l]
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(file_path, overwrite=overwrite)
