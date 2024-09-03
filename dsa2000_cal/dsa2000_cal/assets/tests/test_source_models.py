import os

import pytest
from astropy import time as at, coordinates as ac

from dsa2000_cal.assets.source_models.cyg_a.source_model import CygASourceModel
from dsa2000_cal.common.coord_utils import lmn_to_icrs


def test_model():
    for file in CygASourceModel(seed='abc').get_wsclean_fits_files():
        assert os.path.isfile(file)


@pytest.mark.parametrize('source', ['cas_a', 'cyg_a', 'tau_a', 'vir_a'])
def test_orientations(source: str):
    fill_registries()
    time = at.Time('2021-01-01T00:00:00', scale='utc')

    # source_file = source_model_registry.get_instance(source_model_registry.get_match('cas_a')).get_wsclean_source_file()
    # -00:36:28.234,58.50.46.396
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "58d50m46.396s", frame='icrs')
    # phase_tracking = ac.SkyCoord("-00h36m28.234s", "78d50m46.396s", frame='icrs')

    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match(source)).get_wsclean_fits_files()
    # -04:00:28.608,40.43.33.595
    phase_tracking = ac.SkyCoord("-04h00m28.608s", "40d43m33.595s", frame='icrs')

    freqs = au.Quantity([65e6, 77e6], 'Hz')

    fits_sources = FITSSourceModel.from_wsclean_model(wsclean_fits_files=wsclean_fits_files,
                                                      phase_tracking=phase_tracking, freqs=freqs)
    assert isinstance(fits_sources, FITSSourceModel)

    # Visually verified against ds9, that RA increases over column, and DEC increases over rows.
    fits_sources.plot()
    image, lmn = get_lm_coords_image(wsclean_fits_files[0], time=time, phase_tracking=phase_tracking)
    print(lmn.shape)
    print(image.shape)  # [Nm, Nl]
    l = lmn[:, :, 0]
    m = lmn[:, :, 1]
    _, dl = np.gradient(l)
    dm, _ = np.gradient(m)
    dA = dl * dm
    # dl = np.diff(l, axis=1, prepend=l[:, 1] - l[:, 0])
    # dm = np.diff(m, axis=0, prepend=m[1, :] - m[0, :])
    # print(dl)
    import pylab as plt
    plt.imshow(dl, origin='lower',
               extent=(l.min(), l.max(), m.min(), m.max()))
    plt.colorbar()
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('dl(l,m)')
    plt.show()

    # print(dm)
    plt.imshow(dm, origin='lower',
               extent=(l.min(), l.max(), m.min(), m.max()))
    plt.colorbar()
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('dm(l,m)')
    plt.show()

    # print(dA)
    plt.imshow(dA, origin='lower',
               extent=(l.min(), l.max(), m.min(), m.max()))
    plt.colorbar()
    plt.xlabel('l')
    plt.ylabel('m')
    plt.title('dA(l,m)')
    plt.show()


def get_lm_coords_image(fits_file, time: at.Time, phase_tracking: ac.ICRS):
    import astropy.coordinates as ac
    import astropy.units as au
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from dsa2000_cal.common.coord_utils import icrs_to_lmn

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

        lmn = icrs_to_lmn(pointing_coord, phase_tracking)
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
        world_coords_grid = lmn_to_icrs(LMN * au.dimensionless_unscaled, phase_tracking)
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


import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.visibility_model.source_models.celestial.fits_source.fits_source_model import FITSSourceModel


def test_wsclean_component_files():
    fill_registries()
    # Create a sky model for calibration
    for source in ['cas_a', 'cyg_a', 'tau_a', 'vir_a']:
        source_model_asset = source_model_registry.get_instance(source_model_registry.get_match(source))
        time = at.Time('2021-01-01T00:00:00', format='isot', scale='utc')
        freqs = np.linspace(700e6, 2000e6, 1) * au.Hz
        phase_tracking = ac.ICRS(ra=ac.Angle('0h'), dec=ac.Angle('0d'))
        # source_model = WSCleanSourceModel.from_wsclean_model(
        #     wsclean_clean_component_file=source_model_asset.get_wsclean_clean_component_file(),
        #     time=at.Time('2021-01-01T00:00:00', format='isot', scale='utc'),
        #     freqs=np.linspace(700e6, 2000e6, 2) * au.Hz,
        #     phase_tracking=ac.ICRS(ra=ac.Angle('0h'), dec=ac.Angle('0d'))
        # )

        fits_model = FITSSourceModel.from_wsclean_model(source_model_asset.get_wsclean_fits_files(),
                                                        phase_tracking, freqs, ignore_out_of_bounds=True)
        fits_model.plot()
