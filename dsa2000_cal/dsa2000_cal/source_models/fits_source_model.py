import dataclasses
import os
from functools import partial
from typing import List, Tuple

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
from astropy.coordinates import StokesCoord
from astropy.io import fits
from astropy.wcs import WCS
from ducc0 import wgridder
from jax import lax
from jax._src.typing import SupportsDType

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights
from dsa2000_cal.common.quantity_utils import quantity_to_np, quantity_to_jnp
from dsa2000_cal.source_models.corr_translation import stokes_to_linear


@dataclasses.dataclass(eq=False)
class FitsSourceModel(AbstractSourceModel):
    """
    Predict vis for point source.
    """
    freqs: au.Quantity  # [num_freqs] Frequencies
    images: List[au.Quantity]  # num_freqs list of [Nm, Nl] images
    dl: au.Quantity  # [num_freqs] l pixel size
    dm: au.Quantity  # [num_freqs] m pixel size
    l0: au.Quantity  # [num_freqs] image centre l coordinates (usually 0, but need not be)
    m0: au.Quantity  # [num_freqs] image centre m coordinates (usually 0, but need not be)

    epsilon: float = 1e-4  # accuracy at which the wgridder computation should be done. Must be larger than 2e-13.
    # If dirty has type numpy.float32, it must be larger than 1e-5.
    num_threads: int | None = 1

    dtype: SupportsDType = jnp.complex64

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
    def from_wsclean_model(wsclean_freqs_and_fits: List[Tuple[au.Quantity, str]], time: at.Time,
                           phase_tracking: ac.ICRS,
                           freqs: au.Quantity, **kwargs) -> 'FitsSourceModel':
        """
        Create a FitsSourceModel from a wsclean model file.

        Args:
            wsclean_freqs_and_fits: list of tuples of frequency and fits file
            time: the time of the observation
            phase_tracking: the phase tracking center
            freqs: the frequencies to use
            **kwargs:

        Returns:

        """
        # Each fits file is a 2D image, and we linearly interpolate between frequencies
        available_freqs = au.Quantity([freq for freq, _ in wsclean_freqs_and_fits])
        (i0, alpha0), (i1, alpha1) = jax.tree_map(np.asarray, get_interp_indices_and_weights(
            quantity_to_jnp(freqs), quantity_to_jnp(available_freqs)
        ))
        images = []
        l0s = []
        m0s = []
        dls = []
        dms = []

        for freq_idx in range(len(freqs)):
            # interpolate between the two closest frequencies

            with fits.open(wsclean_freqs_and_fits[i0[freq_idx]][1]) as hdul0:
                # image = hdul0[0].data.T[:, :, 0, 0].T # [Nm, Nl]
                image = hdul0[0].data[0, 0, :, :]  # [Nm, Nl]
                w0 = WCS(hdul0[0].header)
                image = au.Quantity(image, 'Jy')
                if hdul0[0].header['BUNIT'] == 'JY/PIXEL':
                    bmaj = hdul0[0].header['BMAJ'] * au.deg
                    bmin = hdul0[0].header['BMIN'] * au.deg
                    beam_area = np.pi * bmaj * bmin
                    pixel_size_x = au.Quantity(w0.wcs.cdelt[0], au.deg)
                    pixel_size_y = au.Quantity(w0.wcs.cdelt[1], au.deg)
                    pixel_area = np.abs(pixel_size_x * pixel_size_y)

                    beam_per_pixel = beam_area / pixel_area
                    image = image / beam_per_pixel
                centre_x_pix, centre_y_pix = w0.wcs.crpix[0], w0.wcs.crpix[1]
                pointing_coord, spectral_coord, stokes_coord = w0.pixel_to_world(
                    centre_x_pix, centre_y_pix, 0, 0
                )
                # print(w0.pixel_to_world(centre_x_pix, centre_y_pix, 0, 0))
                # Increasing y is decreasing m
                # Increasing x is increasing l
                if stokes_coord != StokesCoord("I"):
                    raise ValueError(f"Expected Stokes I, got {stokes_coord}")
                center_icrs = pointing_coord.transform_to(ac.ICRS)
                lmn0 = icrs_to_lmn(center_icrs, time, phase_tracking)
                l0, m0 = lmn0[:2]
                # I don't trust the wcs cdelt values, so I will compute them.
                # compute l over for m[centre_y_pix] and take mean diff
                Nm, Nl = image.shape
                pointing_coord, _, _ = w0.pixel_to_world(
                    np.arange(Nl) + 1, centre_y_pix * np.ones(Nl), 0, 0
                )
                pointing_icrs = pointing_coord.transform_to(ac.ICRS)
                lmn1 = icrs_to_lmn(pointing_icrs, time, phase_tracking)
                l1, m1 = lmn1[:, 0], lmn1[:, 1]
                dl = np.mean(np.diff(l1))
                # compute m over for l[centre_x_pix] and take mean diff
                pointing_coord, _, _ = w0.pixel_to_world(
                    centre_x_pix * np.ones(Nm), np.arange(Nm) + 1, 0, 0
                )
                pointing_icrs = pointing_coord.transform_to(ac.ICRS)
                lmn1 = icrs_to_lmn(pointing_icrs, time, phase_tracking)
                l1, m1 = lmn1[:, 0], lmn1[:, 1]
                dm = np.mean(np.diff(m1))
                # For RA--SIN and DEC--SIN we can always do:
                # dl = au.Quantity(w0.wcs.cdelt[0], au.deg).to(au.rad).value * au.dimensionless_unscaled
                # dm = au.Quantity(w0.wcs.cdelt[1], au.deg).to(au.rad).value * au.dimensionless_unscaled
                # print(dl, dm)

            images.append(image)
            l0s.append(l0)
            m0s.append(m0)
            dls.append(dl)
            dms.append(dm)
        return FitsSourceModel(
            freqs=freqs,
            images=images,
            dl=au.Quantity(dls),
            dm=au.Quantity(dms),
            l0=au.Quantity(l0s),
            m0=au.Quantity(m0s),
            **kwargs
        )

    def predict(self, uvw: au.Quantity) -> jax.Array:
        num_rows = np.shape(uvw)[0]
        output_vis = np.zeros((num_rows, self.num_freqs, 4), dtype=self.dtype)
        float_dtype = str(np.ones(1, dtype=self.dtype).real.dtype)
        for freq_idx, (freq, image, dl, dm, l0, m0) in enumerate(
                zip(
                    self.freqs,
                    self.images,
                    self.dl,
                    self.dm,
                    self.l0,
                    self.m0
                )
        ):

            if len(np.shape(image)) != 2:
                raise ValueError(f"Expected image to have 2 dimensions, got {len(np.shape(image))}")
            if np.shape(image)[0] % 2 != 0 or np.shape(image)[1] % 2 != 0:
                raise ValueError("Both dimensions must be even")

            wgridder.dirty2vis(
                uvw=quantity_to_np(uvw, dtype=np.float64),
                freq=quantity_to_np(freq, dtype=np.float64)[None],
                dirty=quantity_to_np(image, 'Jy', dtype=float_dtype),
                pixsize_x=-float(quantity_to_np(dl)),
                pixsize_y=float(quantity_to_np(dm)),
                center_x=float(quantity_to_np(l0)),
                center_y=float(quantity_to_np(m0)),
                epsilon=self.epsilon,
                do_wgridding=True,
                flip_v=False,
                divide_by_n=True,
                sigma_min=1.1,
                sigma_max=2.6,
                nthreads=self.num_threads,
                verbosity=0,
                vis=output_vis[:, freq_idx:freq_idx + 1, 0],
                allow_nshift=True,
                gpu=False
            )

        return self._translate_to_linear(jnp.asarray(output_vis))

    @partial(jax.jit, static_argnums=(0,))
    def _translate_to_linear(self, vis_I: jax.Array):
        shape = np.shape(vis_I)
        vis_I = lax.reshape(vis_I, (shape[0] * shape[1], 4))  # [num_rows * num_freqs, 4]
        vis_linear = jax.vmap(lambda y: stokes_to_linear(y, flat_output=True))(vis_I)
        vis_linear = lax.reshape(vis_linear, shape)
        return vis_linear

    def get_flux_model(self, lvec=None, mvec=None):
        # Use imshow to plot the sky model evaluated over a LM grid
        Nm, Nl = self.images[0].shape
        lvec = np.arange(-Nl // 2, Nl // 2) * self.dl[0] + self.l0[0]
        mvec = np.arange(-Nm // 2, Nm // 2) * self.dm[0] + self.m0[0]
        flux_model = self.images[0]  # [Nm, Nl]
        return lvec, mvec, flux_model

    def plot(self):
        lvec, mvec, flux_model = self.get_flux_model()
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

        im = axs.imshow(flux_model, origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]))
        # colorbar
        plt.colorbar(im, ax=axs)
        axs.set_xlabel('l')
        axs.set_ylabel('m')
        plt.show()
