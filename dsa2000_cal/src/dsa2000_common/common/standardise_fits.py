import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, WCSSUB_LONGITUDE, WCSSUB_LATITUDE, WCSSUB_SPECTRAL, WCSSUB_STOKES
from contextlib import contextmanager
import astropy.units as u
import astropy.constants as const

from dsa2000_common.common.logging import dsa_logger

try:
    from reproject import reproject_interp
    HAVE_REPROJECT = True
except ImportError:
    dsa_logger.warning("Reproject not available, skipping reprojection.")
    HAVE_REPROJECT = False

@contextmanager
def standardize_fits(input_file, output_file=None, hdu_index=0, overwrite=True):
    """
    Context manager to open, standardize axes to [RA, DEC, FREQ, STOKES],
    enforce SIN projection, convert units to JY/PIXEL, and optionally write.

    Parameters
    ----------
    input_file : str
        Path to input FITS file.
    output_file : str, optional
        Path to write standardized FITS. If provided, written on exit.
    hdu_index : int, optional
        HDU index to process (default primary HDU=0).
    overwrite : bool, optional
        Overwrite existing output file (default True).

    Yields
    ------
    hdul : astropy.io.fits.HDUList
        Standardized HDUList.
    """
    # --- ENTRY: read and build original WCS ---
    hdul_in = fits.open(input_file)
    hdu = hdul_in[hdu_index]
    data = hdu.data
    header = hdu.header.copy()

    w = WCS(header)
    orig_ctypes = list(w.wcs.ctype)
    prefixes = [ct.split('-')[0] for ct in orig_ctypes]

    # 1) Transpose to header-order
    data_header = data.transpose(tuple(range(data.ndim)[::-1]))

    # 2) Map desired axes logic
    desired = ['RA', 'DEC', 'FREQ', 'STOKES']
    mapping_header = []
    for ax in desired:
        if ax in prefixes:
            mapping_header.append(prefixes.index(ax))
        else:
            mapping_header.append(len(prefixes))
            prefixes.append(ax)

    # 3) Add singleton dims
    while data_header.ndim < len(prefixes):
        data_header = np.expand_dims(data_header, axis=-1)

    # 4) Reorder to logical
    data_logical = data_header.transpose(tuple(mapping_header))

    # 5) Build standardized WCS and enforce SIN
    axes_codes = [WCSSUB_LONGITUDE, WCSSUB_LATITUDE, WCSSUB_SPECTRAL, WCSSUB_STOKES]
    w_std = w.sub(axes_codes)
    for i in (0, 1):
        if not w_std.wcs.ctype[i].endswith('SIN'):
            w_std.wcs.ctype[i] = w_std.wcs.ctype[i][:5] + 'SIN'

    # 6) Create new header and ensure 4 axes
    new_header = w_std.to_header()
    new_header['NAXIS'] = 4
    if 'CTYPE4' not in new_header:
        new_header.set('CTYPE4', 'STOKES', 'Stokes parameter')
        new_header.set('CUNIT4', '', 'Unitless Stokes index')
        new_header.set('CRPIX4', 1.0, 'Reference pixel')
        new_header.set('CRVAL4', 1.0, 'Reference value (I)')
        new_header.set('CDELT4', 1.0, 'Stokes step')

    # 7) Copy non-WCS cards
    for card in header.cards:
        key = card.keyword
        if key in new_header or key.startswith(('CTYPE','CRPIX','CRVAL','CDELT','CUNIT','WCSAXES')):
            continue
        if key in ('HISTORY', 'COMMENT'):
            new_header[key] = card.value
            continue
        try:
            new_header.set(key, card.value, card.comment)
        except Exception:
            continue

    # 8) Optional reprojection to SIN
    if HAVE_REPROJECT and any(not c.endswith('SIN') for c in orig_ctypes[:2]):
        target_header = new_header.copy()
        target_header['CTYPE1'] = 'RA---SIN'
        target_header['CTYPE2'] = 'DEC--SIN'
        data_reproj, _ = reproject_interp((data_logical, w_std), target_header)
        data_logical = data_reproj
        new_header = target_header

    # 9) Back to numpy memory order
    data_out = data_logical.transpose(tuple(range(data_logical.ndim)[::-1]))

    # 10) Unit conversion: ensure JY/PIXEL
    bunit = header.get('BUNIT').upper()
    dsa_logger.info(f"The original BUNIT is {bunit}")
    if bunit != 'JY/PIXEL':
        dsa_logger.info(f"Converting units from {bunit} to JY/PIXEL")
        factor = 1.
        if "/B" in bunit:
            # pixel area (steradian)
            pix_dx = abs(new_header['CDELT1']) * u.deg
            pix_dy = abs(new_header['CDELT2']) * u.deg
            pixel_area = pix_dx.to(u.rad) * pix_dy.to(u.rad)

            # beam area (steradian)
            if 'BMAJ' not in header:
                dsa_logger.warning(f"No BMAJ in header, assuming beam area = pixel area.")
            bmaj = header.get('BMAJ', pix_dx.to('deg').value) * u.deg
            bmin = header.get('BMIN', pix_dy.to('deg').value) * u.deg
            beam_area = 0.25 * np.pi * bmaj.to(u.rad) * bmin.to(u.rad)

            factor *= float(beam_area / pixel_area)
        if "*M/S" in bunit:
            cdelt3 = new_header['CDELT3'] * u.Hz
            crval3 = new_header['CRVAL3'] * u.Hz
            dv = (cdelt3 / crval3) * const.c
            factor /= abs(dv.to('m/s').value)
        data_out *= factor
        new_header['BUNIT'] = 'JY/PIXEL'

    # Build HDUList
    hdu_out = fits.PrimaryHDU(data=data_out, header=new_header)
    hdul = fits.HDUList([hdu_out])

    try:
        yield hdul
    finally:
        # write if requested
        if output_file:
            hdul.writeto(output_file, overwrite=overwrite)
        # close resources
        hdul.close()
        hdul_in.close()


# if __name__ == '__main__':
#     standardize_fits('../../dsa2000_assets/source_models/ncg_5194/NGC_5194_RO_MOM0_THINGS.FITS', 'NGC_5194_RO_MOM0_THINGS_STD.FITS', overwrite=True)
#     standardize_fits('../../dsa2000_assets/source_models/ncg_5194/KATGC-model.fits', 'KATGC-model_STD.fits', overwrite=True)
