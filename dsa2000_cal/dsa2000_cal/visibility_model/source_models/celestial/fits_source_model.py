import dataclasses
from functools import partial
from typing import List, NamedTuple, Tuple

import astropy.coordinates as ac
import astropy.units as au
import jax
import numpy as np
import pylab as plt
from astropy.coordinates import StokesCoord
from astropy.io import fits
from astropy.wcs import WCS
from jax import numpy as jnp
from jax._src.typing import SupportsDType

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.common import wgridder
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.corr_translation import stokes_I_to_linear
from dsa2000_cal.common.interp_utils import get_centred_insert_index
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_np, quantity_to_jnp
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.delay_models.far_field import VisibilityCoords


class FITSModelData(NamedTuple):
    """
    Data for predicting with FFT.
    """
    freqs: jax.Array  # [chan]
    image: jax.Array  # [[chan,] Nl, Nm, [2, 2]] in linear or else Stokes-I format.
    gains: jax.Array  # [time, ant, chan[, 2, 2]]
    l0: jax.Array  # [[chan,]]
    m0: jax.Array  # [[chan,]]
    dl: jax.Array  # [[chan,]]
    dm: jax.Array  # [[chan,]]


@partial(jax.jit, static_argnames=['flat_output'])
def stokes_I_image_to_linear(image: jax.Array, flat_output: bool) -> jax.Array:
    """
    Convert a Stokes I image to linear.

    Args:
        image: [Nl, Nm]

    Returns:
        linear: [Nl, Nm, ...]
    """

    @partial(multi_vmap, in_mapping="[Nl,Nm]", out_mapping="[Nl,Nm,...]")
    def f(coh):
        return stokes_I_to_linear(coh, flat_output)

    return f(image)


@dataclasses.dataclass(eq=False)
class FITSSourceModel(AbstractSourceModel):
    """
    Predict vis from Stokes I images.
    """
    freqs: au.Quantity  # [num_freqs] Frequencies
    images: List[au.Quantity]  # num_freqs list of [Nl, Nm[, 2, 2]] images
    dl: au.Quantity  # [num_freqs] l pixel size (increasing)
    dm: au.Quantity  # [num_freqs] m pixel size
    l0: au.Quantity  # [num_freqs] image centre l coordinates (usually 0, but need not be)
    m0: au.Quantity  # [num_freqs] image centre m coordinates (usually 0, but need not be)

    def __add__(self, other: 'FITSSourceModel') -> 'FITSSourceModel':
        raise NotImplementedError("Addition not implemented")

    def is_full_stokes(self) -> bool:
        shape = self.images[0].shape
        return len(shape) == 4 and shape[-2:] == (2, 2)

    def get_model_data(self, gains: jax.Array | None = None) -> FITSModelData:
        """
        Get the model data for the Gaussian source model.

        Args:
            gains: [time, ant, chan[, 2, 2]] the gains to apply

        Returns:
            model_data: the model data
        """
        image = jnp.stack([quantity_to_jnp(image, 'Jy') for image in self.images], axis=0)  # [chan, Nl, Nm[, 2, 2]]
        return FITSModelData(
            freqs=quantity_to_jnp(self.freqs),
            image=image,
            gains=gains,
            l0=quantity_to_jnp(self.l0),
            m0=quantity_to_jnp(self.m0),
            dl=quantity_to_jnp(self.dl),
            dm=quantity_to_jnp(self.dm)
        )

    def get_lmn_sources(self) -> jax.Array:
        """
        Get the lmn coordinates of the sources.

        Returns:
            lmn: [num_sources=1, 3] l, m, n coordinates of the sources
        """
        l0 = np.mean(self.l0)[None]
        m0 = np.mean(self.m0)[None]
        if not np.all(self.l0 == l0):
            raise ValueError("Expected all l0 to be the same")
        if not np.all(self.m0 == m0):
            raise ValueError("Expected all m0 to be the same")
        n0 = np.sqrt(1 - l0 ** 2 - m0 ** 2)
        return jnp.stack(
            [
                quantity_to_jnp(l0),
                quantity_to_jnp(m0),
                quantity_to_jnp(n0)
            ],
            axis=-1
        )

    def total_flux(self) -> au.Quantity:
        return au.Quantity(np.sum(self.images[0]))

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
            if not (len(image.shape) == 2 or len(image.shape) == 4):
                raise ValueError(f"Expected images to be [Nl, Nm, 2, 2] or [Nl, Nm]. Got {image.shape}")
            if image.shape != self.images[0].shape:
                raise ValueError("Expected images to have the same dimensions")

    @staticmethod
    def from_wsclean_model(wsclean_fits_files: List[str], phase_tracking: ac.ICRS, freqs: au.Quantity,
                           ignore_out_of_bounds: bool = False, full_stokes: bool = True) -> 'FITSSourceModel':
        """
        Create a FitsSourceModel from a wsclean model file.

        Args:
            wsclean_fits_files: list of tuples of frequency and fits file
            phase_tracking: the phase tracking center
            freqs: the frequencies to use
            ignore_out_of_bounds: whether to ignore out of bounds frequencies
            full_stokes: whether the images are full stokes
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
                if np.shape(hdul0[0].data)[1] > 1:
                    raise ValueError(f"Expected 1 FREQ parameter, got {np.shape(hdul0[0].data)[1]}")
                image = hdul0[0].data[:, 0, :, :]  # [stokes, Nm, Nl]
                w0 = WCS(hdul0[0].header)
                image = au.Quantity(image, 'Jy')  # [stokes, Nm, Nl]
                # Reverse l axis
                image = image[:, :, ::-1]  # [stokes, Nm, Nl]
                # Transpose
                image = image.T  # [Nl, Nm, stokes]
                Nl, Nm, num_stokes = image.shape
                # RA--SIN and DEC--SIN
                if not (w0.wcs.ctype[0].strip().endswith('SIN') and w0.wcs.ctype[1].strip().endswith('SIN')):
                    raise ValueError(f"Expected SIN projection, got {w0.wcs.ctype}")
                pixel_size_l = au.Quantity(w0.wcs.cdelt[0], au.deg)
                pixel_size_m = au.Quantity(w0.wcs.cdelt[1], au.deg)
                if hdul0[0].header['BUNIT'] == 'JY/PIXEL':
                    pass
                elif hdul0[0].header['BUNIT'] == 'JY/BEAM':
                    # Convert to JY/PIXEL
                    print(f"Converting from JY/BEAM to JY/PIXEL")
                    bmaj = hdul0[0].header['BMAJ'] * au.deg
                    bmin = hdul0[0].header['BMIN'] * au.deg
                    beam_area = (0.25 * np.pi) * bmaj * bmin
                    pixel_area = np.abs(pixel_size_l * pixel_size_m)
                    beam_per_pixel = beam_area / pixel_area
                    image *= beam_per_pixel
                else:
                    raise ValueError(f"Unknown BUNIT {hdul0[0].header['BUNIT']}")
                ra0, dec0 = w0.wcs.crval[0], w0.wcs.crval[1]

                centre_l_pix, centre_m_pix = Nl / 2., Nm / 2.  # 0 1 2 3 -> 1.5
                # Assume pointing is same for all stokes
                pointing_coord, spectral_coord, stokes_coord = w0.pixel_to_world(
                    centre_l_pix, centre_m_pix, 0, 0
                )
                if stokes_coord != StokesCoord("I"):
                    raise ValueError(f"Expected Stokes I, got {stokes_coord}")
                center_icrs = pointing_coord.transform_to(ac.ICRS)
                lmn0 = icrs_to_lmn(center_icrs, phase_tracking)
                l0, m0, n0 = lmn0
                # Order is l, m, freq, stokes
                dl = -pixel_size_l.to('rad').value * au.dimensionless_unscaled  # negative pixel size
                dm = pixel_size_m.to('rad').value * au.dimensionless_unscaled

                print(
                    f"dl={pixel_size_l.to('rad')}, dm={pixel_size_m.to('rad')}\n"
                    f"centre_ra={ra0}, centre_dec={dec0}\n"
                    f"l0={l0}, m0={m0}\n"
                    f"centre_l_pix={centre_l_pix}, centre_m_pix={centre_m_pix}\n"
                    f"num_l={Nl}, num_m={Nm}, num_stokes={num_stokes}"
                )

                # Image is in stokes I, so we can just take the first element
                image = image[:, :, 0]  # [Nl, Nm]
                if full_stokes:
                    # Convert to linear
                    image = np.asarray(
                        stokes_I_image_to_linear(quantity_to_jnp(image, 'Jy'), flat_output=False)
                    ) * au.Jy

            images.append(image)
            l0s.append(l0)
            m0s.append(m0)
            dls.append(dl)
            dms.append(dm)

        for image in images:
            # Ensure shape is same for each
            if image.shape != images[0].shape:
                raise ValueError("All images must have the same shape")

        return FITSSourceModel(
            freqs=freqs,
            images=images,
            dl=au.Quantity(dls),
            dm=au.Quantity(dms),
            l0=au.Quantity(l0s),
            m0=au.Quantity(m0s),
        )

    def get_flux_model(self, lvec=None, mvec=None):
        # Use imshow to plot the sky model evaluated over a LM grid
        if len(self.images[0].shape) == 2:
            Nl, Nm = self.images[0].shape
            flux_model = self.images[0].T  # [Nm, Nl]
        elif len(self.images[0].shape) == 3:
            _, Nl, Nm = self.images[0].shape
            flux_model = self.images[0][0, :, :].T  # [Nm, Nl]
        elif len(self.images[0].shape) == 4:
            Nl, Nm, _, _ = self.images[0].shape
            flux_model = self.images[0][:, :, 0, 0].T  # [Nm, Nl]
        elif len(self.images[0].shape) == 5:
            _, Nl, Nm, _, _ = self.images[0].shape
            flux_model = self.images[0][0, :, :, 0, 0].T  # [Nm, Nl]
        else:
            raise ValueError(f"Expected image shape [[chan,]Nl, Nm[, 2, 2]], got {self.images[0].shape}")
        lvec = (-0.5 * Nl + np.arange(Nl)) * self.dl[0] + self.l0[0]
        mvec = (-0.5 * Nm + np.arange(Nm)) * self.dm[0] + self.m0[0]

        return lvec, mvec, flux_model

    def plot(self, save_file: str = None):
        lvec, mvec, flux_model = self.get_flux_model()  # [Nm, Nl]
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

        im = axs.imshow(
            flux_model.to('Jy').value,
            origin='lower',
            extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]),
            cmap='inferno',
            interpolation='none'
        )
        # colorbar
        plt.colorbar(im, ax=axs)
        axs.set_xlabel('l')
        axs.set_ylabel('m')
        if save_file is not None:
            plt.savefig(save_file)
        plt.show()


@dataclasses.dataclass(eq=False)
class FITSPredict:
    num_threads: int = 1
    epsilon: float = 1e-6
    convention: str = 'physical'
    dtype: SupportsDType = jnp.complex64

    def check_predict_inputs(self, model_data: FITSModelData) -> Tuple[bool, bool, bool]:
        """
        Check the inputs for predict.

        Args:
            model_data: data, see above for shape info.

        Returns:
            image_has_chan: bool
            full_stokes: bool
            is_gains: bool
        """
        num_chan = np.shape(model_data.freqs)[0]
        if len(np.shape(model_data.image)) == 5:  # [chan, Nl, Nm, 2, 2]
            if np.shape(model_data.image)[-2:] == (2, 2):
                full_stokes = True
                if np.shape(model_data.image)[0] == np.shape(model_data.freqs)[0]:
                    image_has_chan = True
                else:
                    raise ValueError(f"Expected image shape (chan, Nl, Nm, 2, 2), got {np.shape(model_data.image)}")
            else:
                raise ValueError(f"Expected image shape (chan, Nl, Nm, 2, 2), got {np.shape(model_data.image)}")
        elif len(np.shape(model_data.image)) == 4:  # [Nl, Nm, 2, 2]
            if np.shape(model_data.image)[-2:] == (2, 2):
                full_stokes = True
                image_has_chan = False
            else:
                raise ValueError(f"Expected image shape (Nl, Nm, 2, 2), got {np.shape(model_data.image)}")
        elif len(np.shape(model_data.image)) == 3:  # [chan, Nl, Nm]
            full_stokes = False
            if np.shape(model_data.image)[0] == np.shape(model_data.freqs)[0]:
                image_has_chan = True
            else:
                raise ValueError(f"Expected image shape (chan, Nl, Nm), got {np.shape(model_data.image)}")
        elif len(np.shape(model_data.image)) == 2:  # [Nl, Nm]
            full_stokes = False
            image_has_chan = False
        else:
            raise ValueError(f"Expected image shape [[chan,] Nl, Nm, [2, 2]], got {np.shape(model_data.image)}")

        is_gains = model_data.gains is not None

        if is_gains:
            time, ant = np.shape(model_data.gains)[:2]
            if full_stokes:
                if np.shape(model_data.gains) != (time, ant, num_chan, 2, 2):
                    raise ValueError(f"Expected gains shape [time, ant, chan, 2, 2], got {np.shape(model_data.gains)}")
            else:
                if np.shape(model_data.gains) != (time, ant, num_chan):
                    raise ValueError(f"Expected gains shape [time, ant], got {np.shape(model_data.gains)}")

        return image_has_chan, full_stokes, is_gains

    def predict(self, model_data: FITSModelData, visibility_coords: VisibilityCoords) -> jax.Array:
        """
        Predict visibilities from faint model data contained in FITS Stokes-I images.
        Predict takes into account frequency depending on the image provided.

        If the image has a frequency dimension then it must match freqs, and we feed in image by image.

        If the image doesn't have a frequency dimension then it must be shaped (Nm, Nl), and we replicate the image
        for all frequencies.

        Similarly, for gains we replicate the gains for all frequencies if they don't have a frequency dimension.

        In all cases, the output frequencies are determined by the freqs argument.


        Args:
            model_data: data, see above for shape info.
            visibility_coords: visibility coordinates.

        Returns:
            visibilities: [row, chan[, 2, 2]]
        """

        image_has_chan, full_stokes, is_gains = self.check_predict_inputs(
            model_data=model_data
        )
        if full_stokes:
            if image_has_chan:  # [chan, Nl, Nm, 2, 2]
                image_mapping = "[c,Nl,Nm,p,q]"
                out_mapping = "[...,c,p,q]"
            else:  # [Nl, Nm, 2, 2]
                image_mapping = "[Nl,Nm,p,q]"
                out_mapping = "[...,p,q]"  # frequency is the last dimension
        else:
            if image_has_chan:
                image_mapping = "[c,Nl,Nm]"
                out_mapping = "[...,c]"
            else:
                image_mapping = "[Nl,Nm]"
                out_mapping = "[...]"  # frequency is the last dimension

        if is_gains:

            _t = visibility_coords.time_idx
            _a1 = visibility_coords.antenna_1
            _a2 = visibility_coords.antenna_2

            if full_stokes:
                g1 = model_data.gains[_t, _a1, :, :, :]
                g2 = model_data.gains[_t, _a2, :, :, :]
                g_mapping = "[r,c,2,2]"
            else:
                g1 = model_data.gains[_t, _a1, :]
                g2 = model_data.gains[_t, _a2, :]
                g_mapping = "[r,c]"
        else:
            g1 = None
            g2 = None
            g_mapping = "[]"

        @partial(multi_vmap,
                 in_mapping=f"[c],{image_mapping},[c],[c],[c],[c],[r,3]",
                 out_mapping=out_mapping,
                 verbose=True)
        def predict(freqs: jax.Array,
                    image: jax.Array,
                    dl: jax.Array, dm: jax.Array, l0: jax.Array, m0: jax.Array,
                    uvw: jax.Array) -> jax.Array:
            """
            Predict visibilities for a single frequency.

            Args:
                freqs: [[num_freqs]]
                image: [Nm, Nl]
                dl: []
                dm: []
                l0: []
                m0: []
                uvw: [rows, 3]

            Returns:
                vis: [num_rows [, num_freqs]]
            """
            if self.convention == 'casa':
                uvw = jnp.negative(uvw)

            squeeze = False
            if np.shape(freqs) == ():
                freqs = freqs[None]
                squeeze = True

            vis = wgridder.dirty2vis(
                uvw=uvw,
                freqs=freqs,
                dirty=image,
                pixsize_m=dm,
                pixsize_l=dl,
                center_m=m0,
                center_l=l0,
                wgt=None,  # Always None
                flip_v=False,
                epsilon=self.epsilon
            )  # [num_rows, num_chan/1]

            if squeeze:
                vis = vis[:, 0]

            return vis  # [num_rows, num_freqs]

        visibilities = predict(
            model_data.freqs, model_data.image,
            model_data.dl, model_data.dm,
            model_data.l0, model_data.m0,
            visibility_coords.uvw
        )  # [row, chan[, p, q]]

        if full_stokes:
            vis_mapping = "[r,c,2,2]"
            out_mapping = "[r,c,...]"
        else:
            vis_mapping = "[r,c]"
            out_mapping = "[r,c]"

        @partial(multi_vmap, in_mapping=f"{g_mapping},{g_mapping},{vis_mapping}", out_mapping=out_mapping, verbose=True)
        def transform(g1, g2, vis):

            if full_stokes:
                return kron_product(g1, vis, g2.conj().T)  # [2, 2]
            else:
                return g1 * vis * g2.conj()

        if is_gains:
            visibilities = transform(g1, g2, visibilities)  # [num_rows, num_freqs[, 2, 2]]

        return visibilities
