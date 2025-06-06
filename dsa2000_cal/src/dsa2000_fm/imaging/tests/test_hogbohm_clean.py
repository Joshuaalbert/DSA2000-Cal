# test_beam_peak.py

from pathlib import Path

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits

from dsa2000_common.common.ellipse_utils import Gaussian
from dsa2000_fm.imaging.hogbohm_clean import _make_restore_kernel
# --------------------------------------------------------------------------
# Adjust this import to match where deconvolve_image() lives in your repo
from dsa2000_fm.imaging.hogbohm_clean import deconvolve_image


# --------------------------------------------------------------------------


# ----------------------------------------------------------------------
def test_gaussian_total_flux_from_peak_is_consistent():
    """
    total_flux_from_peak(1) should yield a Gaussian whose value at
    the centre (0,0) is 1.  This sanity-checks the analytic formula.
    """
    bmaj = (15.0 * u.arcsec).to(u.rad).value  # 15″ × 10″ beam, PA 25°
    bmin = (10.0 * u.arcsec).to(u.rad).value
    pa = np.deg2rad(25.0)

    total = Gaussian.total_flux_from_peak(1.0, bmaj, bmin)
    g = Gaussian(np.array([0.0, 0.0]), bmaj, bmin, pa, total)

    peak = float(g.compute_flux_density(np.array([0.0, 0.0])))  # DeviceArray → float
    assert np.isclose(peak, 1.0, rtol=1e-12), "Peak deviates from 1 Jy/beam"


# ----------------------------------------------------------------------
def test_restore_kernel_has_unit_peak():
    """
    The restoring-beam kernel built from a FITS header must have a
    numerical peak of 1 so that CLEAN components keep their amplitude.
    """
    hdr = fits.Header()
    hdr["CDELT1"], hdr["CUNIT1"] = -2.0 / 3600, "deg"  # 2″ pixels
    hdr["CDELT2"], hdr["CUNIT2"] = 2.0 / 3600, "deg"
    hdr["BMAJ"] = 15.0 / 3600  # 15″ × 10″ beam
    hdr["BMIN"] = 10.0 / 3600
    hdr["BPA"] = 30.0  # deg

    kernel = _make_restore_kernel(hdr)
    assert np.isclose(kernel.max(), 1.0, rtol=1e-6), "Kernel peak is not 1"


@pytest.mark.parametrize("pix_arcsec, bmaj_arcsec, bmin_arcsec, pa_deg",
                         [(2.0, 15.0, 10.0, 25.0)])  # easy to extend
def test_flux_conserved(tmp_path: Path,
                        pix_arcsec: float,
                        bmaj_arcsec: float,
                        bmin_arcsec: float,
                        pa_deg: float) -> None:
    """
    Create a synthetic dirty image consisting of a single 1 Jy point source,
    run Hogbom CLEAN, and check that the restored image still peaks at 1 Jy beam⁻¹.

    If the restoring kernel is *not* normalised to unit peak the assertion fails.
    """
    # ----------------- parameters & grids -----------------------------------
    nstokes, nfreq, N = 1, 1, 66  # small but comfortably odd
    pix = (pix_arcsec * u.arcsec).to(u.rad).value
    bmaj = (bmaj_arcsec * u.arcsec).to(u.rad).value
    bmin = (bmin_arcsec * u.arcsec).to(u.rad).value
    pa = np.deg2rad(pa_deg)

    # FITS WCS header (only the pieces deconvolve_image actually uses)
    hdr = fits.Header()
    hdr["NAXIS"] = 4
    hdr["NAXIS1"] = N  # l
    hdr["NAXIS2"] = N  # m
    hdr["NAXIS3"] = 1  # freq
    hdr["NAXIS4"] = 1  # stokes
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---SIN", "DEC--SIN"
    hdr["CDELT1"], hdr["CDELT2"] = -pix * 180 / np.pi / 3600, pix * 180 / np.pi / 3600
    hdr["CUNIT1"], hdr["CUNIT2"] = "deg", "deg"
    hdr["CRPIX1"], hdr["CRPIX2"] = N // 2 + 1, N // 2 + 1
    hdr["BMAJ"], hdr["BMIN"], hdr["BPA"] = bmaj * 180 / np.pi / 3600, \
                                           bmin * 180 / np.pi / 3600, \
        pa_deg

    # ----------------- point-source dirty image -----------------------------
    # Build a Gaussian PSF (same one your code uses for the beam)
    total_flux = Gaussian.total_flux_from_peak(1.0, bmaj, bmin)
    g = Gaussian(np.array([0.0, 0.0]), bmaj, bmin, pa, total_flux)

    l = (np.arange(N) - N // 2) * pix
    m = (np.arange(N) - N // 2) * pix
    L, M = np.meshgrid(l, m, indexing="ij")
    psf_plane = np.array(jax.vmap(jax.vmap(g.compute_flux_density))(jnp.stack((L, M), axis=-1)))  # [N, N]
    # Dirty image = PSF (point source of 1 Jy convolved with PSF)
    dirty = np.zeros((nstokes, nfreq, N, N), dtype=np.float64)
    dirty[0, 0] = psf_plane

    # ----------------- write temporary FITS files ---------------------------
    dfile = tmp_path / "dirty.fits"
    psf_file = tmp_path / "psf.fits"
    model_out = tmp_path / "model.fits"
    resid_out = tmp_path / "resid.fits"
    restor_out = tmp_path / "restor.fits"

    fits.writeto(dfile, dirty, hdr, overwrite=True)
    fits.writeto(psf_file, dirty, hdr, overwrite=True)  # PSF same data/shape

    # ----------------- run CLEAN -------------------------------------------
    deconvolve_image(
        image_fits=str(dfile),
        psf_fits=str(psf_file),
        model_output=str(model_out),
        residual_output=str(resid_out),
        restored_output=str(restor_out),
        gain=1.0,  # one subtraction is enough
        niter=10,
        threshold=0.0
    )

    # ----------------- assertion: peak must be 1 Jy/beam --------------------
    restored = fits.getdata(restor_out)
    peak_restored = restored.max()
    assert np.isclose(peak_restored, 1.0, rtol=1e-2), \
        f"Restored peak {peak_restored:.4f} Jy/beam ≠ 1.0 – flux not conserved"
