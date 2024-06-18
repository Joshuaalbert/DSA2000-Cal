import dataclasses
import os
from typing import List

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.numpy as jnp
import numpy as np
import pylab as plt
from astropy.coordinates import StokesCoord
from astropy.io import fits
from astropy.wcs import WCS
from jax._src.typing import SupportsDType

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.interp_utils import get_centred_insert_index
from dsa2000_cal.common.quantity_utils import quantity_to_np


@dataclasses.dataclass(eq=False)
class FitsStokesISourceModel(AbstractSourceModel):
    """
    Predict vis from Stokes I images.
    """
    freqs: au.Quantity  # [num_freqs] Frequencies
    images: List[au.Quantity]  # num_freqs list of [Nl, Nm] images
    dl: au.Quantity  # [num_freqs] l pixel size (increasing)
    dm: au.Quantity  # [num_freqs] m pixel size
    l0: au.Quantity  # [num_freqs] image centre l coordinates (usually 0, but need not be)
    m0: au.Quantity  # [num_freqs] image centre m coordinates (usually 0, but need not be)

    epsilon: float = 1e-4  # accuracy at which the wgridder computation should be done. Must be larger than 2e-13.
    # If dirty has type numpy.float32, it must be larger than 1e-5.
    num_threads: int | None = 1

    dtype: SupportsDType = jnp.complex64

    def flux_weighted_lmn(self) -> au.Quantity:
        Nm, Nl = self.images[0].shape
        lvec = np.arange(-Nl // 2, Nl // 2) * self.dl[0] + self.l0[0]
        mvec = np.arange(-Nm // 2, Nm // 2) * self.dm[0] + self.m0[0]
        M, L = np.meshgrid(mvec, lvec, indexing='ij')
        flux_model = self.images[0]  # [Nm, Nl]
        m_avg = np.sum(M * flux_model) / np.sum(flux_model)
        l_avg = np.sum(L * flux_model) / np.sum(flux_model)
        lmn = np.asarray([l_avg, m_avg, np.sqrt(1. - l_avg ** 2 - m_avg ** 2)]) * au.dimensionless_unscaled
        return lmn

    def __post_init__(self):
        if not self.freqs.unit.is_equivalent(au.Hz):
            raise ValueError(f"Expected freqs to be in Hz, got {self.freqs.unit}")
        if not self.l0.unit.is_equivalent(au.dimensionless_unscaled):
            raise ValueError(f"Expected l0 to be dimensionless, got {self.l0.unit}")
        if not self.m0.unit.is_equivalent(au.dimensionless_unscaled):
            raise ValueError(f"Expected m0 to be dimensionless, got {self.m0.unit}")
        if not self.dl.unit.is_equivalent(au.dimensionless_unscaled):
            raise ValueError(f"Expected dl to be dimensionless, got {self.dl.unit}")
        if not self.dm.unit.is_equivalent(au.dimensionless_unscaled):
            raise ValueError(f"Expected dm to be dimensionless, got {self.dm.unit}")
        for image in self.images:
            if not image.unit.is_equivalent(au.Jy):
                raise ValueError(f"Expected image to be in Jy, got {image.unit}")

        # Ensure all are 1D vectors
        if self.freqs.isscalar:
            self.freqs = self.freqs.reshape((-1,))
        if self.l0.isscalar:
            self.l0 = self.l0.reshape((-1,))
        if self.m0.isscalar:
            self.m0 = self.m0.reshape((-1,))
        if self.dl.isscalar:
            self.dl = self.dl.reshape((-1,))
        if self.dm.isscalar:
            self.dm = self.dm.reshape((-1,))

        self.num_freqs = self.freqs.shape[0]

        if not all([x.shape == (self.num_freqs,) for x in [self.l0,
                                                           self.m0,
                                                           self.dl,
                                                           self.dm]]):
            raise ValueError("All inputs must have the same shape")

        if len(self.images) != self.num_freqs:
            raise ValueError("Expected one image per frequency")
        for image in self.images:
            if len(image.shape) != 2:
                raise ValueError("Expected image to have 2 dimensions")

        if self.num_threads is None:
            # use 1/2 all available threads
            num_cpus = os.cpu_count()
            self.num_threads = num_cpus // 2

    @staticmethod
    def from_wsclean_model(wsclean_fits_files: List[str],
                           time: at.Time,
                           phase_tracking: ac.ICRS,
                           freqs: au.Quantity,
                           ignore_out_of_bounds: bool = False,
                           **kwargs) -> 'FitsStokesISourceModel':
        """
        Create a FitsSourceModel from a wsclean model file.

        Args:
            wsclean_fits_files: list of tuples of frequency and fits file
            time: the time of the observation
            phase_tracking: the phase tracking center
            freqs: the frequencies to use
            **kwargs:

        Returns:
            FitsSourceModel
        """

        wsclean_fits_freqs_and_fits = []
        for fits_file in wsclean_fits_files:
            # Get frequency from header, open with astropy
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                # Try to find freq
                if 'FREQ' in header:
                    frequency = header['FREQ'] * au.Hz
                elif 'RESTFRQ' in header:
                    frequency = header['RESTFRQ'] * au.Hz
                elif 'CRVAL3' in header:  # Assuming the frequency is in the third axis
                    frequency = header['CRVAL3'] * au.Hz
                else:
                    raise KeyError("Frequency information not found in FITS header.")
                wsclean_fits_freqs_and_fits.append((frequency, fits_file))

        # Sort by freq
        wsclean_fits_freqs_and_fits = sorted(wsclean_fits_freqs_and_fits, key=lambda x: x[0])

        # Each fits file is a 2D image, and we linearly interpolate between frequencies
        available_freqs = au.Quantity([freq for freq, _ in wsclean_fits_freqs_and_fits])
        i0 = get_centred_insert_index(insert_value=quantity_to_np(freqs),
                                      grid_centres=quantity_to_np(available_freqs),
                                      ignore_out_of_bounds=ignore_out_of_bounds)

        images = []
        l0s = []
        m0s = []
        dls = []
        dms = []

        for freq_idx in range(len(freqs)):
            # interpolate between the two closest frequencies
            with fits.open(wsclean_fits_freqs_and_fits[i0[freq_idx]][1]) as hdul0:
                # image = hdul0[0].data.T[:, :, 0, 0].T # [Nm, Nl]
                image = hdul0[0].data[0, 0, :, :]  # [Nm, Nl]
                w0 = WCS(hdul0[0].header)
                image = au.Quantity(image, 'Jy')
                # RA--SIN and DEC--SIN
                cos_dec = np.cos(w0.wcs.crval[1] * np.pi / 180.)
                if hdul0[0].header['BUNIT'] == 'JY/PIXEL':
                    pixel_size_l = au.Quantity(w0.wcs.cdelt[0], au.deg)
                    pixel_size_m = cos_dec * au.Quantity(w0.wcs.cdelt[1], au.deg)
                    pass
                elif hdul0[0].header['BUNIT'] == 'JY/BEAM':
                    # Convert to JY/PIXEL
                    bmaj = hdul0[0].header['BMAJ'] * au.deg
                    bmin = hdul0[0].header['BMIN'] * au.deg
                    beam_area = (np.pi / (2. * np.log(2))) * bmaj * bmin
                    pixel_size_l = au.Quantity(w0.wcs.cdelt[0], au.deg)
                    pixel_size_m = cos_dec * au.Quantity(w0.wcs.cdelt[1], au.deg)
                    pixel_area = np.abs(pixel_size_l * pixel_size_m)

                    beam_per_pixel = beam_area / pixel_area
                    image *= beam_per_pixel
                else:
                    raise ValueError(f"Unknown BUNIT {hdul0[0].header['BUNIT']}")
                centre_l_pix, centre_m_pix = w0.wcs.crpix[0], w0.wcs.crpix[1]
                centre_l, centre_m = w0.wcs.crval[0], w0.wcs.crval[1]
                Nm, Nl = image.shape
                print(f"dl={pixel_size_l.to('rad')}, dm={pixel_size_m.to('rad')}\n"
                      f"centre_ra={centre_l}, centre_dec={centre_m}\n"
                      f"centre_l_pix={centre_l_pix}, centre_m_pix={centre_m_pix}\n"
                      f"num_l={Nl}, num_m={Nm}")
                pointing_coord, spectral_coord, stokes_coord = w0.pixel_to_world(
                    centre_l_pix, centre_m_pix, 0, 0
                )
                # Increasing y is increasing m
                # Increasing x is decreasing l
                if stokes_coord != StokesCoord("I"):
                    raise ValueError(f"Expected Stokes I, got {stokes_coord}")
                center_icrs = pointing_coord.transform_to(ac.ICRS)
                lmn0 = icrs_to_lmn(center_icrs, phase_tracking)
                l0, m0 = lmn0[:2]
                # I don't trust the wcs cdelt values, so I will compute them.
                # compute l over for m[centre_y_pix] and take mean diff

                # Order is l, m, freq, stokes
                # compute l over for m=centre_m and take mean diff
                pointing_coord, _, _ = w0.pixel_to_world(
                    [1, Nl], [centre_m_pix, centre_m_pix], 0, 0
                )
                pointing_icrs = pointing_coord.transform_to(ac.ICRS)
                lmn1 = icrs_to_lmn(pointing_icrs, phase_tracking)
                l1 = lmn1[:, 0]
                dl = (l1[-1] - l1[0]) / (Nl - 1)
                # dl = np.mean(np.diff(l1))
                # compute m over for l[centre_l_pix] and take mean diff
                pointing_coord, _, _ = w0.pixel_to_world(
                    [centre_l_pix, centre_l_pix], [1, Nm], 0, 0
                )
                pointing_icrs = pointing_coord.transform_to(ac.ICRS)
                lmn1 = icrs_to_lmn(pointing_icrs, phase_tracking)
                m1 = lmn1[:, 1]
                dm = (m1[-1] - m1[0]) / (Nm - 1)
                # dm = np.mean(np.diff(m1))
                print(f"My dl={dl}, dm={dm}")

                # Convert to [Nl, Nm], with increasing l
                image = image[:, ::-1].T  # [Nl, Nm]
                dl = -dl

            images.append(image)
            l0s.append(l0)
            m0s.append(m0)
            dls.append(dl)
            dms.append(dm)

        for image in images:
            # Ensure shape is same for each
            if image.shape != images[0].shape:
                raise ValueError("All images must have the same shape")

        return FitsStokesISourceModel(
            freqs=freqs,
            images=images,
            dl=au.Quantity(dls),
            dm=au.Quantity(dms),
            l0=au.Quantity(l0s),
            m0=au.Quantity(m0s),
            **kwargs
        )

    def get_flux_model(self, lvec=None, mvec=None):
        # Use imshow to plot the sky model evaluated over a LM grid
        Nl, Nm = self.images[0].shape
        lvec = (-0.5 * Nl + np.arange(Nl)) * self.dl[0] + self.l0[0]
        mvec = (-0.5 * Nm + np.arange(Nm)) * self.dm[0] + self.m0[0]
        flux_model = self.images[0].T  # [Nm, Nl]
        return lvec, mvec, flux_model

    def plot(self):
        lvec, mvec, flux_model = self.get_flux_model()  # [Nm, Nl]
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

        im = axs.imshow(
            flux_model.value,
            origin='lower',
            extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]),
            cmap='inferno',
            interpolation='none'
        )
        # colorbar
        plt.colorbar(im, ax=axs)
        axs.set_xlabel('l')
        axs.set_ylabel('m')
        plt.show()
