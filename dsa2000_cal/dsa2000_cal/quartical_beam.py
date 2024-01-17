import logging
from typing import Tuple

import astropy.units as au
import numpy as np
import pylab as plt
from astropy import coordinates as ac, io, wcs
from astropy.io import fits

from dsa2000_cal.abc import AbstractAntennaBeam, AbstractAntennaModel

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
    voltage_gain = antenna_model.get_voltage_gain()
    amplitude = antenna_model.get_amplitude() / voltage_gain
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


def get_beam_width(antenna_model: AbstractAntennaModel, threshold: float = 0.01) -> float:
    """
    Get the antenna recpetion beam width in degrees.

    Args:
        antenna_model: antenna model
        threshold: threshold value to use to determine beam width

    Returns:
        beam width in degrees
    """
    voltage_gain = antenna_model.get_voltage_gain()
    amplitude = antenna_model.get_amplitude() / voltage_gain
    circular_mean = np.mean(amplitude, axis=1)
    theta = antenna_model.get_theta()
    max_theta = 0.
    for i, freq in enumerate(antenna_model.get_freqs()):
        for k, th in enumerate(theta):
            if circular_mean[k, i] < threshold:
                break
        max_theta = max(max_theta, th)
    return max_theta


def plot_circular_beam(antenna_model: AbstractAntennaModel, theshold: float = 0.01):
    """
    Plot the circular beam.

    Args:
        antenna_model: antenna model
        threshold: threshold value to use to determine beam part to plot
    """

    voltage_gain = antenna_model.get_voltage_gain()
    amplitude = antenna_model.get_amplitude() / voltage_gain
    circular_mean = np.mean(amplitude, axis=1)
    theta = antenna_model.get_theta()
    norm = plt.Normalize(vmin=antenna_model.get_freqs().min(), vmax=antenna_model.get_freqs().max())
    for i, freq in enumerate(antenna_model.get_freqs()):
        for k, th in enumerate(theta):
            if circular_mean[k, i] < theshold:
                break
        plt.plot(theta[:k], circular_mean[:k, i], label=freq, color=plt.cm.get_cmap('jet_r')(norm(freq)))
        print(f"Freq: {i}, {freq}, theta: {k}, {th}")
    plt.xlabel('Theta (deg)')
    plt.ylabel('Amplitude')
    plt.title(f"'{antenna_model.__class__.__name__}' Beam")
    plt.legend()
    plt.show()


def create_quartical_beam(antenna_beam: AbstractAntennaBeam, output_beam_fits: str,
                          threshold: float = 0.01):
    antenna_model = antenna_beam.get_model()
    beam_width = get_beam_width(antenna_model, threshold=threshold)
    num_pix = find_num_pixels(antenna_model=antenna_model, beam_width=beam_width, test_threshold=threshold)
    # We compute the X,Y be pointing the antenna in same direction as phase-tracking centre and computing ~L,M coords (negated L)
    beam_width = get_beam_width(antenna_model, threshold=0.01)
    pix_delta = beam_width / num_pix
    logger.info(f"Beam width: {beam_width} deg, num pixels: {num_pix}, pixel delta: {pix_delta} deg")

    beam_freqs = antenna_beam.get_model().get_freqs()
    freq_delta = np.mean(np.diff(beam_freqs))
    logger.info(f"Num freqs: {len(beam_freqs)}, freq delta: {freq_delta} Hz")

    w = wcs.WCS(naxis=3)
    w.wcs.ctype = ["X", "Y", "FREQ"]
    w.wcs.cunit = ["DEG", "DEG", "Hz"]

    # 1 2 3 -> 2 = 3 / 2 + 0.5
    # 1 2 3 4 -> 2.5 = 4 / 2 + 0.5
    w.wcs.crpix = [num_pix / 2 + 0.5, num_pix / 2 + 0.5, 1]
    w.wcs.cdelt = [pix_delta, pix_delta, freq_delta]
    w.wcs.crval = [0., 0., beam_freqs[0]]
    w.wcs.set()

    # Now we interpolate the beam model to the pixel coordinates
    pointing = ac.ICRS(ra=0. * au.deg, dec=0. * au.deg)  # Any pointing will do.
    source_ra = np.linspace(-beam_width / 2., beam_width / 2., num_pix)
    source_dec = np.linspace(-beam_width / 2., beam_width / 2., num_pix)
    RA, DEC = np.meshgrid(source_ra, source_dec, indexing='ij')
    sources = ac.ICRS(ra=RA.flatten() * au.deg, dec=DEC.flatten() * au.deg)

    data = np.zeros((num_pix, num_pix, len(beam_freqs)), dtype=np.float32)
    for i, freq in enumerate(beam_freqs):
        data[:, :, i] = antenna_model.compute_amplitude(
            pointing=pointing,
            sources=sources,
            freq_hz=freq,
            enu_frame=enu_frame,
            pol='X').reshape((num_pix, num_pix))

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

    hdu.header['OBJECT'] = 'beam'
    hdu.header['EQUINOX'] = 2000.0
    hdu.header['ORIGIN'] = 'GFOSTER'
    hdu.header['TELESCOP'] = 'VLA'
    hdu.header['EXTEND'] = True

    for i in range(len(beam_freqs)):
        hdu.header[f"GFREQ{i + 1}"] = beam_freqs[i]

    hdu.writeto(output_beam_fits, overwrite=True)
