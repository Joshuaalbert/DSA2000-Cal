import sympy as sp

pi = sp.pi


def perley_lmn_from_icrs(alpha, dec, alpha0, dec0):
    dra = alpha - alpha0

    l = sp.cos(dec) * sp.sin(dra)
    m = sp.sin(dec) * sp.cos(dec0) - sp.cos(dec) * sp.sin(dec0) * sp.cos(dra)
    n = sp.sin(dec) * sp.sin(dec0) + sp.cos(dec) * sp.cos(dec0) * sp.cos(dra)
    return l, m, n


def perley_icrs_from_lmn(l, m, n, alpha0, dec0):
    dec = sp.asin(m * sp.cos(dec0) + n * sp.sin(dec0))
    ra = alpha0 + sp.atan2(l, n * sp.cos(dec0) - m * sp.sin(dec0))
    return ra, dec


def griesen_celestial_to_native(alpha, delta, alpha_p, delta_p, phi_p):
    phi = phi_p + sp.atan2(
        -sp.cos(delta) * sp.sin(alpha - alpha_p),
        sp.sin(delta) * sp.cos(delta_p) - sp.cos(delta) * sp.sin(delta_p) * sp.cos(alpha - alpha_p)
    )
    theta = sp.asin(
        sp.sin(delta) * sp.sin(delta_p) + sp.cos(delta) * sp.cos(delta_p) * sp.cos(alpha - alpha_p)
    )
    return phi, theta


def verify_fits(fits_file):
    """
    Shows each pixel has constant l,m.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy import units as au, coordinates as ac
    import numpy as np

    def perley_lmn_from_icrs(icrs: ac.ICRS, phase_center: ac.ICRS) -> au.Quantity:
        ra = icrs.ra
        dec = icrs.dec
        ra0 = phase_center.ra
        dec0 = phase_center.dec
        dra = ra - ra0

        l = np.cos(dec) * np.sin(dra)
        m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)
        n = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(dra)
        return np.stack([l, m, n], axis=-1) * au.dimensionless_unscaled

    with fits.open(fits_file) as hdul0:
        image = hdul0[0].data[0, 0, :, :]  # [Nm, Nl]
        Nm, Nl = np.shape(image)
        w0 = WCS(hdul0[0].header)
        # RA--SIN and DEC--SIN
        m_pix = np.arange(Nm)
        l_pix = np.arange(Nl)
        M, L = np.meshgrid(m_pix, l_pix, indexing='ij')

        pointing_coord, spectral_coord, stokes_coord = w0.pixel_to_world(
            L, M, 0, 0
        )

        ra0, dec0 = w0.wcs.crval[0], w0.wcs.crval[1]
        phase_center = ac.ICRS(ra0 * au.deg, dec0 * au.deg)

        pointing_coord = pointing_coord.transform_to(ac.ICRS)
        lmn = perley_lmn_from_icrs(pointing_coord, phase_center)

        dm = np.diff(lmn[:, :, 1], axis=0)
        dl = np.diff(lmn[:, :, 0], axis=1)

        assert np.std(dm) < 1e-10
        assert np.std(dl) < 1e-10

        pixel_size_l = au.Quantity(w0.wcs.cdelt[0], au.deg)
        pixel_size_m = au.Quantity(w0.wcs.cdelt[1], au.deg)
        np.testing.assert_allclose(np.mean(dl).value, pixel_size_l.to('rad').value)
        np.testing.assert_allclose(np.mean(dm).value, pixel_size_m.to('rad').value)


def griesen_native_to_celestial(phi, theta, alpha_p, delta_p, phi_p):
    alpha = alpha_p + sp.atan2(
        -sp.cos(theta) * sp.sin(phi - alpha_p),
        sp.sin(theta) * sp.cos(delta_p) - sp.cos(theta) * sp.sin(delta_p) * sp.cos(phi - phi_p)
    )
    delta = sp.asin(
        sp.sin(theta) * sp.sin(delta_p) + sp.cos(theta) * sp.cos(delta_p) * sp.cos(phi - phi_p)
    )
    return alpha, delta


def griesen_xy(phi, theta, theta_c, phi_c, phi_p):
    x_griesen = (180 / pi) * (
            -sp.cos(theta) * sp.sin(phi - phi_p) + sp.cot(theta_c) * sp.sin(phi_c) * (1 - sp.sin(phi - phi_p)))
    y_griesen = -(180 / pi) * (
            -sp.cos(theta) * sp.cos(phi - phi_p) + sp.cot(theta_c) * sp.cos(phi_c) * (1 - sp.sin(phi - phi_p)))
    return x_griesen, y_griesen


def celestial_to_cartesian(ra, dec):
    x = sp.cos(ra) * sp.cos(dec)
    y = sp.sin(ra) * sp.cos(dec)
    z = sp.sin(dec)
    return x, y, z


def xy_from_lmn(l, m, theta_c, phi_c, phi_p, alpha0, dec0):
    n = sp.sqrt(1 - l * l - m * m)
    ra, dec = perley_icrs_from_lmn(l, m, n, alpha0, dec0)
    phi, theta = griesen_celestial_to_native(ra, dec, alpha0, dec0, phi_p)
    x, y = griesen_xy(phi, theta, theta_c, phi_c, phi_p)
    return x, y


def main():
    """
    Verifies that constant change in l,m corresponds to constant change in pixel coordinates.
    """
    # Define symbols
    alpha0, dec0, l, m, dl, dm = sp.symbols('alpha0 dec0 l m dl dm')

    x, y = xy_from_lmn(l, m, theta_c=pi / 2, phi_c=0, phi_p=pi, alpha0=alpha0, dec0=dec0)

    x = sp.simplify(x)
    y = sp.simplify(y)

    print(f"x={x}, y={y}")

    verify_fits('/src/dsa2000_cal/assets/source_models/cas_a/Cas-MFS-model.fits')


if __name__ == '__main__':
    main()
