import logging
from typing import Tuple

import astropy.units as au
import numpy as np
import pylab as plt

from dsa2000_cal.abc import AbstractAntennaModel

logger = logging.getLogger(__name__)

"""
This is a valid FITS file header for a quartical beam:

SIMPLE  =                    T / conforms to FITS standard                      
BITPIX  =                  -64 / array data type                                
NAXIS   =                    3 / number of array dimensions                     
NAXIS1  =                  513                                                  
NAXIS2  =                  513                                                  
NAXIS3  =                    3                                                  
EXTEND  =                    T                                                  
DATE    = '2015-05-20 12:40:12.507624'                                          
DATE-OBS= '2015-05-20 12:40:12.507624'                                          
ORIGIN  = 'GFOSTER '                                                            
TELESCOP= 'VLA     '                                                            
OBJECT  = 'beam    '                                                            
EQUINOX =               2000.0                                                  
CTYPE1  = 'X       '           / points right on the sky                        
CUNIT1  = 'DEG     '                                                            
CDELT1  =             0.011082 / degrees                                        
CRPIX1  =                  257 / reference pixel (one relative)                 
CRVAL1  =      0.0110828777007                                                  
CTYPE2  = 'Y       '           / points up on the sky                           
CUNIT2  = 'DEG     '                                                            
CDELT2  =             0.011082 / degrees                                        
CRPIX2  =                  257 / reference pixel (one relative)                 
CRVAL2  =   -2.14349358381E-07                                                  
CTYPE3  = 'FREQ    '                                                            
CDELT3  =     30303030.3030303 / frequency step in Hz                           
CRPIX3  =                    1 / reference frequency postion                    
CRVAL3  =         1280000000.0 / reference frequency                            
CTYPE4  = 'STOKES  '                                                            
CDELT4  =                    1                                                  
CRPIX4  =                    1                                                  
CRVAL4  =                   -5                                                  
GFREQ1  =         1280000000.0                                                  
GFREQ2  =         1306120000.0                                                  
GFREQ3  =         1333330000.0                                                  
END 
"""


def bore_sight_coords_to_pixel_coords(theta: float, phi: float) -> Tuple[float, float]:
    """
    Convert theta and phi to pixel coordinates.

    Args:
        theta: theta in degrees
        phi: phi in degrees

    Returns:
        x: x pixel coordinate right on the sky
        y: y pixel coordinate up on the sky
    """
    a = np.sin(theta) * np.cos(phi)
    b = np.sin(theta) * np.sin(phi)
    c = np.cos(theta)

    x = np.arctan2(b, a)
    y = np.arctan2(c, np.sqrt(a ** 2 + b ** 2))
    return x, y


def pixel_coords_to_bore_sight_coords(x: float, y: float) -> Tuple[float, float]:
    """
    Convert pixel coordinates to theta and phi.

    Args:
        x: x pixel coordinate right on the sky
        y: y pixel coordinate up on the sky

    Returns:
        theta: theta in degrees
        phi: phi in degrees
    """
    a = np.cos(y) * np.cos(x)
    b = np.cos(y) * np.sin(x)
    c = np.sin(y)

    theta = np.arccos(c)
    phi = np.arctan2(b, a)
    return theta, phi


def find_num_pixels(antenna_model: AbstractAntennaModel, beam_width: float, test_threshold: float) -> int:
    """
    Find the number of pixels required to represent the beam model.

    Args:
        antenna_model: antenna model
        beam_width: beam width in degrees
        test_threshold: threshold value to use to determine beam part to plot

    Returns:
        number of pixels
    """
    voltage_gain = antenna_model.get_voltage_gain()  # [num_freq]
    amplitude = antenna_model.get_amplitude()[..., 0, 0] / voltage_gain  # [theta, phi, freq]
    circular_mean = np.mean(amplitude, axis=1)
    theta = antenna_model.get_theta()
    freqs = antenna_model.get_freqs()

    def cost_function(num_pix: int) -> float:
        # Max deviation
        theta_propose = np.linspace(0., beam_width, num_pix)
        k = np.searchsorted(theta, beam_width)
        theta_test = theta[:k]
        max_deviation = []
        for i in range(len(freqs)):
            beam_propose = np.interp(theta_propose, theta, circular_mean[:, i])
            f_test = lambda theta: np.interp(theta, theta_propose, beam_propose)
            max_deviation.append(np.max(np.abs(circular_mean[:k, i] - f_test(theta_test))))
        return np.max(max_deviation)

    for b in np.arange(3, 11):
        num_pix = 2 ** b + 1
        if cost_function(num_pix) < test_threshold:
            return num_pix


def get_beam_widths(amplitude: au.Quantity, theta: au.Quantity, freqs: au.Quantity, threshold: float = 0.5) -> Tuple[
    au.Quantity, au.Quantity]:
    """
    Get the antenna beam width in degrees for a given threshold. Uses the circularised beam.

    Args:
        amplitude: [theta, phi, freq]
        theta: [theta]
        freqs: [freq]
        threshold: float, threshold value to use to determine beam width, 0.5 is half-power beam width

    Returns:
        freqs: [freq]
        beam_widths: [freq]
    """
    if not amplitude.unit.is_equivalent(au.dimensionless_unscaled):
        raise ValueError(f"Expected amplitude to be dimensionless but got {amplitude.unit}"
                         f" with shape {amplitude.shape}.")
    if not theta.unit.is_equivalent(au.deg):
        raise ValueError(f"Expected theta to be in degrees but got {theta.unit}.")
    if not freqs.unit.is_equivalent(au.Hz):
        raise ValueError(f"Expected freqs to be in Hz but got {freqs.unit}.")

    circular_mean = np.mean(amplitude**2, axis=1)  # [theta, freq]
    circular_mean /= np.max(circular_mean, axis=0, keepdims=True)
    freq_order = np.argsort(freqs)

    beam_widths = []
    for i in freq_order:
        mask = circular_mean[:, i] > threshold
        mask[:-1] = np.logical_or(mask[:-1], np.diff(circular_mean[:, i]) > 0.)
        argmin = np.argmin(mask)
        th = theta[argmin]
        beam_widths.append(th * 2)

    freqs = freqs[freq_order]
    beam_widths = au.Quantity(beam_widths)
    return freqs, beam_widths


def get_dish_model_beam_widths(antenna_model: AbstractAntennaModel, threshold: float = 0.5) -> Tuple[
    au.Quantity, au.Quantity]:
    """
    Get the antenna recpetion beam width in degrees.

    Args:
        antenna_model: antenna model
        threshold: threshold value to use to determine beam width, 0.5 is half-power beam width

    Returns:
        freqs: [freq] the frequencies
        beam_widths: [freq] the beam widths at the frequencies
    """
    amplitude = antenna_model.get_amplitude()  # [theta, phi, freq, 2, 2]
    freqs = antenna_model.get_freqs()
    theta = antenna_model.get_theta()
    # Use XX to define beamwidth
    return get_beam_widths(amplitude[..., 0, 0], theta, freqs, threshold=threshold)


def plot_beam_profile(antenna_model: AbstractAntennaModel, threshold: float = 0.01):
    """
    Plot the circular beam.

    Args:
        antenna_model: antenna model
        threshold: threshold value to use to determine beam part to plot
    """

    amplitude = antenna_model.get_amplitude()[..., 0, 0]  # [theta, phi, freq]
    circular_mean = np.mean(amplitude, axis=1)  # [theta, freq]
    circular_mean /= np.max(circular_mean, axis=0, keepdims=True)
    theta = antenna_model.get_theta()
    norm = plt.Normalize(vmin=antenna_model.get_freqs().value.min(), vmax=antenna_model.get_freqs().value.max())
    freqs, beam_widths = get_dish_model_beam_widths(antenna_model, threshold=threshold)
    for i, freq in enumerate(freqs):
        k = int(np.interp(beam_widths[i].value / 2., theta.value, np.arange(len(theta))))
        plt.plot(theta[:k], circular_mean[:k, i],
                 label=freq, color=plt.get_cmap('PuOr_r')(norm(freq.value)))
        print(f"Freq: {i}, {freq}, theta: {beam_widths[i] / 2.}")
    plt.xlabel('Theta (deg)')
    plt.ylabel('Amplitude')
    plt.title(f"'{antenna_model.__class__.__name__}' Half-width of beam")
    # Put a colorbar to the right of the plot
    sm = plt.cm.ScalarMappable(cmap='PuOr_r', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Frequency (Hz)', ax=plt.gca())
    # plt.legend()
    plt.show()
