import numpy as np
from astropy.convolution import convolve_fft
from astropy.io import fits
from numba import njit, prange


def deconvolve_image(
        image_fits: str,
        psf_fits: str,
        model_output: str,
        residual_output: str,
        gain: float = 0.1,
        niter: int = 1000,
        threshold: float = None,
        restore_beam: bool = False
) -> None:
    """
    Perform Hogbom CLEAN on a 4D FITS (stokes, freq, m, l), with optional beam restoration.

    Assumes image and PSF both have shape (nstokes, nfreq, ny, nx).
    """
    # Load data
    dirty = fits.getdata(image_fits).astype(np.float64)
    header = fits.getheader(image_fits)
    psf = fits.getdata(psf_fits).astype(np.float64)

    nstokes, nfreq, ny, nx = dirty.shape
    # Initialize arrays
    model = np.zeros_like(dirty)
    residual = dirty.copy()

    # Clean per stokes/freq plane
    for s in range(nstokes):
        for f in range(nfreq):
            # Extract 2D slices
            plane = np.ascontiguousarray(residual[s, f])
            comp = model[s, f]
            beam = np.ascontiguousarray(psf[s, f])
            for _ in range(niter):
                # Find peak
                y0, x0 = np.unravel_index(np.argmax(np.abs(plane)), plane.shape)
                val = plane[y0, x0]
                if threshold is not None and abs(val) < threshold:
                    break
                comp[y0, x0] += gain * val
                _subtract_psf2d(plane, beam, gain, val, x0, y0)
            # Write back updated slices
            residual[s, f] = plane
            model[s, f] = comp

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
def _subtract_psf2d(
        plane: np.ndarray,
        psf2d: np.ndarray,
        gain: float,
        peak_val: float,
        x0: int,
        y0: int
) -> None:
    """
    Subtract a 2D PSF kernel from the residual plane at (y0,x0).
    """
    h, w = psf2d.shape
    cy = h // 2
    cx = w // 2
    ny, nx = plane.shape
    for i in prange(h):
        yy = y0 + (i - cy)
        if yy < 0 or yy >= ny:
            continue
        for j in range(w):
            xx = x0 + (j - cx)
            if xx < 0 or xx >= nx:
                continue
            plane[yy, xx] -= gain * peak_val * psf2d[i, j]


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
