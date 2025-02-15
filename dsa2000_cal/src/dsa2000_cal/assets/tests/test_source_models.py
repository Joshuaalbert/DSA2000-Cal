import astropy.coordinates as ac
import astropy.time as at

from dsa2000_common.common.coord_utils import lmn_to_icrs


def get_lm_coords_image(fits_file, time: at.Time, phase_center: ac.ICRS):
    import astropy.coordinates as ac
    import astropy.units as au
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from dsa2000_common.common.coord_utils import icrs_to_lmn

    with fits.open(fits_file) as hdul0:
        # image = hdul0[0].data.T[:, :, 0, 0].T # [Nm, Nl]
        image = hdul0[0].data[0, 0, :, :]  # [Nm, Nl]
        w0 = WCS(hdul0[0].header)
        image = au.Quantity(image, 'Jy')
        # RA--SIN and DEC--SIN
        if hdul0[0].header['BUNIT'] == 'JY/PIXEL':
            central_pixel_size_l = au.Quantity(w0.wcs.cdelt[0], au.deg)
            central_pixel_size_m = au.Quantity(w0.wcs.cdelt[1], au.deg)
            pass
        elif hdul0[0].header['BUNIT'] == 'JY/BEAM':
            # Convert to JY/PIXEL
            bmaj = hdul0[0].header['BMAJ'] * au.deg
            bmin = hdul0[0].header['BMIN'] * au.deg
            # bpa = hdul0[0].header['BPA'] * au.deg
            beam_area = (0.25 * np.pi) * bmaj * bmin
            central_pixel_size_l = au.Quantity(w0.wcs.cdelt[0], au.deg)
            central_pixel_size_m = au.Quantity(w0.wcs.cdelt[1], au.deg)
            pixel_area = np.abs(central_pixel_size_l * central_pixel_size_m)

            beam_per_pixel = beam_area / pixel_area
            image *= beam_per_pixel
        else:
            raise ValueError(f"Unknown BUNIT {hdul0[0].header['BUNIT']}")
        centre_l_pix, centre_m_pix = w0.wcs.crpix[0], w0.wcs.crpix[1]
        centre_l, centre_m = w0.wcs.crval[0], w0.wcs.crval[1]
        Nm, Nl = image.shape
        print(
            f"dl={central_pixel_size_l.to('rad')}, dm={central_pixel_size_m.to('rad')}\n"
            f"centre_ra={centre_l}, centre_dec={centre_m}\n"
            f"centre_l_pix={centre_l_pix}, centre_m_pix={centre_m_pix}\n"
            f"num_l={Nl}, num_m={Nm}"
        )
        Nm, Nl = image.shape
        l_pixels = np.arange(Nl) + 1
        m_pixels = np.arange(Nm) + 1
        L_pixels, M_pixels = np.meshgrid(l_pixels, m_pixels, indexing='ij')
        pointing_coord, spectral_coord, stokes_coord = w0.pixel_to_world(
            L_pixels, M_pixels, 0, 0
        )
        pointing_coord = pointing_coord.transform_to(ac.ICRS)
        print(pointing_coord)

        lmn = icrs_to_lmn(pointing_coord, phase_center)
        Nm, Nl = np.shape(image)
        l = lmn[:, :, 0]
        m = lmn[:, :, 1]
        _, dl = np.gradient(l)
        dm, _ = np.gradient(m)
        dA = dl * dm  # [Nm, Nl]

        flux_density = image / dA
        lvec = np.linspace(np.min(l), np.max(l), Nl)
        mvec = np.linspace(np.min(m), np.max(m), Nm)
        M, L = np.meshgrid(mvec, lvec, indexing='ij')
        N = np.sqrt(1. - (L ** 2 + M ** 2))
        LMN = np.stack([L, M, N], axis=-1)  # [Nm, Nl, 3]
        world_coords_grid = lmn_to_icrs(LMN * au.dimensionless_unscaled, phase_center)
        world_coords_grid = ac.SkyCoord(ra=world_coords_grid.ra, dec=world_coords_grid.dec, frame='icrs')
        print(spectral_coord[0, 0], stokes_coord[0, 0])
        array_indices = w0.world_to_array_index(pointing_coord[-1, 0], spectral_coord[-1, 0], stokes_coord[-1, 0])
        # print(array_indices)
        array_indices = w0.world_to_array_index(world_coords_grid[-1, 0], spectral_coord[-1, 0], stokes_coord[-1, 0])
        print(array_indices)

        # multilinear_interp_2d(M.ravel(), L.ravel(), )
        # dl = lvec[1] - lvec[0]
        # dm = mvec[1] - mvec[0]

        return image, lmn
