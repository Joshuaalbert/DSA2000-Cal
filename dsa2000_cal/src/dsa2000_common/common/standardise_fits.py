from contextlib import contextmanager

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from dsa2000_common.common.logging import dsa_logger

try:
    from reproject import reproject_interp

    HAVE_REPROJECT = True
except ImportError:
    dsa_logger.warning("Reproject not available, skipping reprojection.")
    HAVE_REPROJECT = False


@contextmanager
def standardize_fits(input_file, output_file=None, hdu_index=0, overwrite=True, reproject: bool = False):
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
        The standardized HDUList.
    """
    # --- Read input ---
    hdul_in = fits.open(input_file)
    hdu = hdul_in[hdu_index]
    data = hdu.data
    header = hdu.header.copy()

    # Ensure 4D array: (STOKES,FREQ,DEC,RA) in memory order
    while data.ndim < 4:
        data = np.expand_dims(data, axis=0)

    # Transpose to header order (axis1..NAXIS)
    data_hdr = data.transpose(tuple(range(data.ndim)[::-1]))

    # Identify original WCS
    w_orig = WCS(header)
    orig = w_orig.wcs
    prefixes = [ctype.split('-')[0] for ctype in orig.ctype]

    # Reorder to logical order [RA, DEC, FREQ, STOKES]
    desired = ['RA', 'DEC', 'FREQ', 'STOKES']
    map_hdr = []
    for ax in desired:
        if ax in prefixes:
            map_hdr.append(prefixes.index(ax))
        else:
            map_hdr.append(len(prefixes))
            prefixes.append(ax)
    # Inject singletons
    while data_hdr.ndim < len(prefixes):
        data_hdr = np.expand_dims(data_hdr, axis=-1)
    # Logical data
    data_log = data_hdr.transpose(tuple(map_hdr))

    # --- Build new WCS with exactly 4 axes explicitly ---
    w_std = WCS(naxis=4)
    # Identify original axis indices (or use defaults for new singleton)
    prefixes = prefixes  # from above
    ra_idx = prefixes.index('RA') if 'RA' in prefixes else None
    dec_idx = prefixes.index('DEC') if 'DEC' in prefixes else None
    freq_idx = prefixes.index('FREQ') if 'FREQ' in prefixes else None

    # CTYPE
    w_std.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'STOKES']
    # CRPIX: use original reference pixels or 1 for new axes
    w_std.wcs.crpix = [
        orig.crpix[ra_idx] if ra_idx is not None else 1.0,
        orig.crpix[dec_idx] if dec_idx is not None else 1.0,
        orig.crpix[freq_idx] if freq_idx is not None else 1.0,
        1.0
    ]
    # CDELT: pixel scales
    w_std.wcs.cdelt = [
        orig.cdelt[ra_idx] if ra_idx is not None else 1.0,
        orig.cdelt[dec_idx] if dec_idx is not None else 1.0,
        orig.cdelt[freq_idx] if freq_idx is not None else 1.0,
        1.0
    ]
    # CRVAL: world coordinate at reference pixel
    w_std.wcs.crval = [
        orig.crval[ra_idx] if ra_idx is not None else 0.0,
        orig.crval[dec_idx] if dec_idx is not None else 0.0,
        orig.crval[freq_idx] if freq_idx is not None else 0.0,
        1.0  # Stokes I
    ]
    # CUNIT: units
    w_std.wcs.cunit = [
        orig.cunit[ra_idx] if ra_idx is not None else 'deg',
        orig.cunit[dec_idx] if dec_idx is not None else 'deg',
        orig.cunit[freq_idx] if freq_idx is not None else 'Hz',
        ''
    ]
    # Finalize WCS
    w_std.wcs.set()

    # Generate header from new WCS
    new_header = w_std.to_header()

    # Copy non-WCS cards
    for card in header.cards:
        key = card.keyword
        if key in new_header or key.startswith(('CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'WCSAXES')):
            continue
        if key in ('HISTORY', 'COMMENT'):
            new_header[key] = card.value
        else:
            try:
                new_header.set(key, card.value, card.comment)
            except Exception:
                pass

    # Optional reprojection if needed
    if reproject and HAVE_REPROJECT and not (orig.ctype[0].endswith('SIN') and orig.ctype[1].endswith('SIN')):
        target_hdr = new_header.copy()
        target_hdr['CTYPE1'] = 'RA---SIN'
        target_hdr['CTYPE2'] = 'DEC--SIN'
        data_log, _ = reproject_interp((data_log, w_std), target_hdr)
        new_header = target_hdr

    # Back to native memory order
    data_out = data_log.transpose(tuple(range(data_log.ndim)[::-1]))

    # Unit conversion to JY/PIXEL
    bunit = header.get('BUNIT', '').upper()
    dsa_logger.info(f"Original BUNIT: {bunit}")
    if bunit != 'JY/PIXEL':
        dsa_logger.info(f"Converting units from {bunit} to JY/PIXEL")
        factor = 1.0
        if '/B' in bunit:
            pix_dx = u.Quantity(new_header['CDELT1'], new_header['CUNIT1'])
            pix_dy = u.Quantity(new_header['CDELT2'], new_header['CUNIT2'])
            pixel_area = pix_dx.to(u.rad) * pix_dy.to(u.rad)
            bmaj = header.get('BMAJ', pix_dx.value) * u.deg
            bmin = header.get('BMIN', pix_dy.value) * u.deg
            beam_solid_angle = np.pi / (4 * np.log(2)) * bmaj.to(u.rad) * bmin.to(u.rad)
            pixel_per_beam = abs(float(beam_solid_angle / pixel_area))
            # Jy/Beam / pixel_per_beam = Jy/pixel
            factor /= pixel_per_beam
        if '*M/S' in bunit:
            cd3 = u.Quantity(new_header['CDELT3'], new_header['CUNIT3'])
            cr3 = u.Quantity(new_header['CRVAL3'], new_header['CUNIT3'])
            dv = (cd3 / cr3) * const.c
            factor /= abs(dv.to('m/s').value)
        data_out *= factor
        new_header['BUNIT'] = 'JY/PIXEL'

    # Build HDUList
    hdu_out = fits.PrimaryHDU(data=data_out, header=new_header)
    hdul = fits.HDUList([hdu_out])

    try:
        yield hdul
    finally:
        if output_file:
            hdul.writeto(output_file, overwrite=overwrite)
        hdul.close()
        hdul_in.close()

# if __name__ == '__main__':
#     standardize_fits('../../dsa2000_assets/source_models/ncg_5194/NGC_5194_RO_MOM0_THINGS.FITS', 'NGC_5194_RO_MOM0_THINGS_STD.FITS', overwrite=True)
#     standardize_fits('../../dsa2000_assets/source_models/ncg_5194/KATGC-model.fits', 'KATGC-model_STD.fits', overwrite=True)
