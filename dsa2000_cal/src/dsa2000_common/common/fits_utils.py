import logging
import os
from typing import Tuple, Literal

import astropy.time as at
import astropy.units as au
import numpy as np
from astropy import coordinates as ac, wcs
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import zoom

from dsa2000_common.common.quantity_utils import quantity_to_np
from dsa2000_common.common.serialise_utils import SerialisableBaseModel

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


class ImageModel(SerialisableBaseModel):
    phase_center: ac.ICRS
    obs_time: at.Time
    dl: au.Quantity
    dm: au.Quantity
    freqs: au.Quantity  # [num_freqs]
    bandwidth: au.Quantity
    coherencies: Tuple[str, ...]  # [num_coherencies]
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
    if not image_model.dl.unit.is_equivalent(au.rad):
        raise ValueError("dl must be in rad")
    if not image_model.dm.unit.is_equivalent(au.rad):
        raise ValueError("dm must be in rad")
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
    if not image_model.phase_center.isscalar:
        raise ValueError("phase_center must be scalar")
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


def save_image_to_fits(file_path: str, image_model: ImageModel, overwrite: bool = False, radian_angles: bool = False,
                       casa_compat_center_location: bool = False):
    """
    Saves an image to FITS using SIN projection.

    Args:
        file_path: the path to the output FITS file
        image_model: the image model

    Raises:
        FileExistsError: if the file already exists and overwrite is False
    """
    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(f"File {file_path} already exists")

    # Create the WCS
    w = wcs.WCS(naxis=4)  # 4D image [l, m, freq, coherency]
    # 4-3 form
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    if (np.shape(image_model.image)[0] % 2 != 0) or (np.shape(image_model.image)[1] % 2 != 0):
        raise ValueError("Image must have an even number of pixels in each direction")

    if not casa_compat_center_location:
        # Since we flip the l axis, the reference pixel is the other side
        # To see this:
        # index         : 0     1       2       3 (These indices are the array coordinates in unflipped order)
        # l (dl units)  : -2    -1      0       1 (the l formula of wgridder is l[i] = (-1/2 N + i) dl + l0)
        # pixel-0based  : 3     2       1       0 (each pixel is negative -- which is why we flip l)
        # pixel-1based  : 4     3       2       1 (in fits file we just add one to the pixel-0based)
        # So l[N//2] == 0 in pixel-0based is N//2 - 1, which is N//2 in pixel-1based
        crpix_l = image_model.image.shape[0] // 2
        crpix_m = image_model.image.shape[1] // 2 + 1
    else:
        # Alternatively, if we want the 0-based center pixel to be at N//2, then we want l[N//2 - 1] == 0
        # To see this:
        # index         : 0     1       2       3       4       6
        # l (dl units)  : -3    -2      -1      0       1       2
        # pixel-0based  : 5     4       3       2       1       0
        # pixel-1based  : 6     5       4       3       2       1
        # So l[N//2 - 1] == 0 can be achived by slicing the array from 2:
        # index         : 0     1       2       3
        # l (dl units)  : -1    0       1       2
        # pixel-0based  : 3     2       1       0
        # pixel-1based  : 4     3       2       1
        image_model.image = image_model.image[2:, ...]
        crpix_l = image_model.image.shape[0] // 2 + 1
        crpix_m = image_model.image.shape[1] // 2 + 1

    w.wcs.crpix = [crpix_l, crpix_m, 1, 1]
    dfreq = image_model.bandwidth / len(image_model.freqs)
    if radian_angles:
        angle_unit = 'rad'
    else:
        angle_unit = 'deg'
    w.wcs.cdelt = [
        -image_model.dl.to(angle_unit).value,
        image_model.dm.to(angle_unit).value,
        dfreq.to('Hz').value,
        1
    ]
    w.wcs.crval = [
        image_model.phase_center.ra.to(angle_unit).value,
        image_model.phase_center.dec.to(angle_unit).value,
        image_model.freqs[0].to('Hz').value,
        1  # {1: I, 2: Q, 3: U, 4: V}
    ]
    w.wcs.cunit = [angle_unit, angle_unit, 'Hz', '']
    w.wcs.set()
    # Set beam
    header = w.to_header()
    header["BUNIT"] = image_model.unit
    header["BSCALE"] = 1
    header["BZERO"] = 0
    header["BTYPE"] = "Intensity"
    header["BMAJ"] = image_model.beam_major.to("deg").value
    header["BMIN"] = image_model.beam_minor.to("deg").value
    header["BPA"] = image_model.beam_pa.to("deg").value
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
