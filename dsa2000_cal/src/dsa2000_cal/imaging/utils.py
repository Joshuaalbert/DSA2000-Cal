import itertools

import astropy.units as au
import numpy as np
from astropy import constants

from dsa2000_cal.antenna_model.antenna_model_utils import get_dish_model_beam_widths
from dsa2000_cal.assets.content_registry import fill_registries, NoMatchFound
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.fourier_utils import find_optimal_fft_size
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_np
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMeta


def get_image_parameters(meta: MeasurementSetMeta,
                         field_of_view: au.Quantity | None = None,
                         oversample_factor: float = 5.):
    """
    Get the image parameters for imaging

    Args:
        meta: a MeasurementSetMeta instance
        field_of_view: the field of view in degrees
        oversample_factor: the oversampling factor, higher is more accurate but bigger image

    Returns:
        num_pixel: the number of pixels in the image
        dl: the pixel size in the l direction
        dm: the pixel size in the m direction
        center_l: the center of the image in l direction
        center_m: the center of the image in m direction
    """
    wavelengths = quantity_to_np(constants.c / meta.freqs)
    diameter = np.min(quantity_to_np(meta.antenna_diameters))
    if field_of_view is not None:
        field_of_view = field_of_view
    else:
        # Try to get HPFW from the actual beam
        try:
            fill_registries()
            antenna_model = array_registry.get_instance(
                array_registry.get_match(meta.array_name)).get_antenna_model()
            _freqs, _beam_widths = get_dish_model_beam_widths(antenna_model)
            field_of_view = np.max(np.interp(meta.freqs, _freqs, _beam_widths))
        except NoMatchFound as e:
            print(f"Failed to get beam width from antenna model: {e}")
            field_of_view = au.Quantity(
                1.22 * np.max(wavelengths) / diameter,
                au.rad
            )
            print(f"Using diffraction limit: {field_of_view}")
    # D/ 4F = 1.22 wavelength / D ==> F = D^2 / (4 * 1.22 * wavelength)
    effective_focal_length = diameter ** 2 / (4 * 1.22 * np.max(wavelengths))
    print(f"Effective focal length: {effective_focal_length}")

    # Get the maximum baseline length
    min_wavelength = np.min(wavelengths)
    antennas_itrs = meta.antennas.get_itrs().cartesian.xyz.to(au.m).value.T
    antenna1, antenna2 = np.asarray(list(itertools.combinations_with_replacement(range(len(antennas_itrs)), 2)),
                                    dtype=mp_policy.index_dtype).T
    uvw = np.linalg.norm(antennas_itrs[antenna2] - antennas_itrs[antenna1], axis=-1)  # [num_baselines]
    max_baseline = np.max(uvw)

    # Number of pixels
    diffraction_limit_resolution = 1.22 * min_wavelength / max_baseline
    pixel_size = (diffraction_limit_resolution / oversample_factor) * au.rad
    num_pixel = find_optimal_fft_size(
        int(field_of_view / pixel_size)
    )

    dl = pixel_size.to('rad')
    dm = pixel_size.to('rad')

    center_l = 0. * au.rad
    center_m = 0. * au.rad

    print(f"Center x: {center_l}, Center y: {center_m}")
    print(f"Image size: {num_pixel} x {num_pixel}")
    print(f"Pixel size: {dl} x {dm}")
    return num_pixel, dl, dm, center_l, center_m
