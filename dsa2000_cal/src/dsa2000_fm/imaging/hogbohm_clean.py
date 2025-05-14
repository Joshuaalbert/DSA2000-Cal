import os

import numba
import numpy as np
from astropy.convolution import convolve_fft
from astropy.io import fits
from numba import njit, prange

from dsa2000_common.common.logging import dsa_logger


def deconvolve_image(
        image_fits: str,
        psf_fits: str,
        model_output: str,
        residual_output: str,
        gain: float = 0.1,
        niter: int = 1000,
        threshold: float = None,
        restore_beam: bool = False,
        kappa: float = 4
) -> None:
    """
    Perform Hogbom CLEAN on a 4D FITS (l, m, freqs, stokes), with optional beam restoration.

    Assumes image and PSF both have shape (num_l, num_m, num_freq, nstokes).
    Note: data is stored transposed in FITS, so the memory order is (stokes, freq, m, l).
    """

    numba.set_num_threads(os.cpu_count())
    # Load data and headers
    # Note it's transposed in FITS
    dirty = fits.getdata(image_fits).astype(np.float64)
    header = fits.getheader(image_fits)
    psf = fits.getdata(psf_fits).astype(np.float64)
    psf_header = fits.getheader(psf_fits)

    # Reference pixel from PSF header (1-based in FITS)
    crpix1 = int(psf_header['CRPIX1'])  # l-axis reference pixel 1-based
    crpix2 = int(psf_header['CRPIX2'])  # m-axis reference pixel 1-based
    # Convert to 0-based center indices for numpy arrays
    li0_psf = crpix1 - 1
    mi0_psf = crpix2 - 1

    nstokes, nfreq, num_m, num_l = dirty.shape
    # Initialize arrays
    model = np.zeros_like(dirty)
    residual = dirty.copy()

    dynamic_threshold = threshold is None

    # Clean per stokes/freq plane
    iteration = 0
    for s in range(nstokes):
        for f in range(nfreq):
            # Extract 2D slices
            residual_plane = np.ascontiguousarray(residual[s, f])
            psf_plane = np.ascontiguousarray(psf[s, f])
            model_plane = model[s, f]
            if dynamic_threshold:
                threshold = kappa * np.std(residual_plane)
            while iteration < niter:
                iteration += 1
                # Find peak in current residual, note the transposed order
                mi0, li0 = np.unravel_index(np.argmax(residual_plane), residual_plane.shape)
                peak_val = residual_plane[mi0, li0]
                if peak_val <= 0:
                    break
                if dynamic_threshold and peak_val < threshold:
                    threshold = kappa * np.std(residual_plane)
                if peak_val < threshold:
                    break
                model_plane[mi0, li0] += gain * peak_val
                subtract_psf2d_slice(
                    residuals=residual_plane,
                    psf=psf_plane,
                    gain=gain,
                    peak_val=peak_val,
                    li0=li0,
                    mi0=mi0,
                    li0_psf=li0_psf,
                    mi0_psf=mi0_psf
                )
            # Write back updated slices
            residual[s, f] = residual_plane
            model[s, f] = model_plane
    dsa_logger.info(f"Cleaned {nstokes} x {nfreq} planes with {iteration} / {niter} iterations")

    # Optionally restore beam
    if restore_beam:
        kernel = _make_restore_kernel(header)
        restored = np.zeros_like(model)
        for s in range(nstokes):
            for f in range(nfreq):
                restored[s, f] = convolve_fft(model[s, f], kernel, allow_huge=True)
        model = restored

    # Save outputs
    fits.writeto(model_output, model, header, overwrite=True)
    fits.writeto(residual_output, residual, header, overwrite=True)


@njit(parallel=True)
def _subtract_psf2d_numba(
        residuals: np.ndarray,
        psf: np.ndarray,
        gain: float,
        peak_val: float,
        li0: int,
        mi0: int,
        li0_psf: int,
        mi0_psf: int
) -> None:
    """
    Inplace subtract a scaled PSF kernel from the residual plane at (li0, mi0),
    using reference center (li0_psf, mi0_psf) from CRPIX.

    Args:
        residuals: [num_l, num_m] residuals array in JY/BEAM
        psf: [num_l_psf, num_m_psf] PSF kernel
        gain: scaling factor
        peak_val: peak value of the residuals
        li0: l-dimension peak in residuals 0-based
        mi0: m-dimension peak in residuals 0-based
        li0_psf: l-dimension peak in PSF 0-based
        mi0_psf: m-dimension peak in PSF 0-based
    """
    num_m_psf, num_l_psf = psf.shape
    num_m, num_l = residuals.shape
    for li_psf in prange(num_l_psf):
        li = li0 + (li_psf - li0_psf)
        if li < 0 or li >= num_l:
            continue
        for mi_psf in range(num_m_psf):
            mi = mi0 + (mi_psf - mi0_psf)
            if mi < 0 or mi >= num_m:
                continue
            residuals[mi, li] -= gain * peak_val * psf[mi_psf, li_psf]


def subtract_psf2d_slice(residuals, psf, gain, peak_val, li0, mi0, li0_psf, mi0_psf):
    # About 4x faster than numba one.
    num_m, num_l = residuals.shape
    num_m_psf, num_l_psf = psf.shape

    li_min = li0 - li0_psf
    li_max = li_min + num_l_psf
    mi_min = mi0 - mi0_psf
    mi_max = mi_min + num_m_psf

    l0 = max(0, li_min)
    l1 = min(num_l, li_max)
    m0 = max(0, mi_min)
    m1 = min(num_m, mi_max)

    psf_l0 = l0 - li_min
    psf_l1 = psf_l0 + (l1 - l0)
    psf_m0 = m0 - mi_min
    psf_m1 = psf_m0 + (m1 - m0)

    residuals[m0:m1, l0:l1] -= gain * peak_val * psf[psf_m0:psf_m1, psf_l0:psf_l1]


def _make_restore_kernel(header):
    """
    Build an idealized Gaussian restoring beam kernel using header BMAJ, BMIN, BPA.
    Kernel spans up to 3*BMAJ in pixels (odd size).
    """
    pix_dx = abs(header['CDELT1'])
    pix_dy = abs(header['CDELT2'])
    bmaj = header['BMAJ']
    bmin = header['BMIN']
    pa = np.deg2rad(header['BPA'])
    rx = int(np.ceil(3 * bmaj / pix_dx))
    ry = int(np.ceil(3 * bmin / pix_dy))
    sx = 2 * rx + 1
    sy = 2 * ry + 1
    y, x = np.indices((sy, sx))
    cx0, cy0 = rx, ry
    dx = (x - cx0) * pix_dx
    dy = (y - cy0) * pix_dy
    xp = dx * np.cos(pa) + dy * np.sin(pa)
    yp = -dx * np.sin(pa) + dy * np.cos(pa)
    f2s = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_x = bmin * f2s
    sigma_y = bmaj * f2s
    kern = np.exp(-0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2))
    return kern / np.sum(kern)
#
#
# if __name__ == '__main__':
#
#     # Example usage:
#     deconvolve_image(
#         image_fits='point_sources_dsa1650_a_P305_v2.4.6.fits',
#         psf_fits='point_sources_dsa1650_a_P305_v2.4.6_psf.fits',
#         model_output='point_sources_dsa1650_a_P305_v2.4.6_model.fits',
#         residual_output='point_sources_dsa1650_a_P305_v2.4.6_residual.fits',
#         gain=0.1,
#         niter=1000,
#         threshold=None,
#         restore_beam=False
#     )
