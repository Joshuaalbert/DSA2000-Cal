import dataclasses
from functools import partial
from typing import NamedTuple, List

import astropy.coordinates as ac
import astropy.units as au
import jax
import numpy as np
import pylab as plt
from astropy.io import fits
from astropy.wcs import WCS
from jax import numpy as jnp, lax

from dsa2000_cal.adapter.utils import broadcast_translate_corrs
from dsa2000_cal.common import wgridder
from dsa2000_cal.common.array_types import ComplexArray, FloatArray
from dsa2000_cal.common.interp_utils import InterpolatedArray, select_interpolation_points
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np
from dsa2000_cal.common.types import VisibilityCoords
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_cal.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_cal.delay_models.uvw_utils import perley_lmn_from_icrs, perley_icrs_from_lmn
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_cal.visibility_model.source_models.abc import AbstractSourceModel


class FITSModelData(NamedTuple):
    """
    Data for predict.
    """
    image: FloatArray  # [facet, num_l, num_m, [2,2]] in [[xx, xy], [yx, yy]] format or stokes I
    l0: FloatArray  # [facet] central l coordinate of the facet
    m0: FloatArray  # [facet] central m coordinate of the facet
    dl: FloatArray  # [facet] width of pixel in l
    dm: FloatArray  # [facet] width of pixel in m


@dataclasses.dataclass(eq=False)
class BaseFITSSourceModel(AbstractSourceModel[FITSModelData]):
    """
    Predict vis for fits source.
    """
    model_freqs: FloatArray  # [num_model_freqs] Frequencies
    image: FloatArray  # [num_model_freqs, facet, num_l, num_m, [2,2]] in [[xx, xy], [yx, yy]] format or stokes I
    ra: FloatArray  # [num_model_freqs,facet] central ra coordinate of the facet
    dec: FloatArray  # [num_model_freqs,facet] central dec coordinate of the facet
    dl: FloatArray  # [num_model_freqs,facet] width of pixel in l
    dm: FloatArray  # [num_model_freqs,facet] width of pixel in m

    epsilon: float = 1e-6
    convention: str = 'physical'
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return

        # Check shapes match
        if len(np.shape(self.model_freqs)) != 1:
            raise ValueError("model_freqs must be 1D")
        if len(np.shape(self.ra)) != 2:
            raise ValueError(f"ra must be (num_model_freqs, facet) got {np.shape(self.ra)}")
        if np.shape(self.ra)[0] != np.shape(self.model_freqs)[0]:
            raise ValueError("ra must have same length as model_freqs")
        if np.shape(self.image)[:2] != np.shape(self.ra):
            raise ValueError(f"image[:2] shape must be {np.shape(self.ra)}, got {np.shape(self.image)}")
        num_l, num_l = np.shape(self.image)[2:4]
        if num_l % 2 != 0 or num_l % 2 != 0:
            raise ValueError("num_l and num_m must be even")

    def is_full_stokes(self) -> bool:
        return len(np.shape(self.image)) == 6 and np.shape(self.image)[-2:] == (2, 2)

    def get_model_slice(self, freq: FloatArray, time: FloatArray, geodesic_model: BaseGeodesicModel) -> FITSModelData:
        interp = InterpolatedArray(x=self.model_freqs, values=(self.image, self.ra, self.dec, self.dl, self.dm), axis=0)
        image, ra, dec, dl, dm = interp(freq)  # [facet, num_l, num_m, [2,2]]

        def _mask_image(facet_image, ra, dec, dl, dm):
            num_l, num_m = np.shape(facet_image)[:2]
            lmn = geodesic_model.compute_far_field_lmn(ra, dec, time, return_elevation=False)
            l0, m0 = lmn[:2]
            lvec = (-0.5 * num_l + jnp.arange(num_l)) * dl + l0
            mvec = (-0.5 * num_m + jnp.arange(num_m)) * dm + m0
            L, M = jnp.meshgrid(lvec, mvec, indexing='ij')
            L2M2 = L ** 2 + M ** 2
            N = jnp.sqrt(1 - L2M2)
            LMN = jnp.stack([L, M, N], axis=-1)  # [num_l, num_m, 3]
            elevation = geodesic_model.compute_elevation_from_lmn(LMN, time)  # [num_l, num_m]
            elevation_mask = elevation <= 0
            n_mask = L2M2 > 1
            mask = jnp.logical_or(elevation_mask, n_mask)
            if self.is_full_stokes():
                return jnp.where(mask[:, :, None, None], jnp.zeros_like(facet_image), facet_image), l0, m0
            else:
                return jnp.where(mask, jnp.zeros_like(facet_image), facet_image), l0, m0

        image, l0, m0 = jax.vmap(_mask_image)(image, ra, dec, dl, dm)  # [facet, num_l, num_m, [2,2]]

        return FITSModelData(
            image=mp_policy.cast_to_image(image),
            l0=mp_policy.cast_to_angle(l0),
            m0=mp_policy.cast_to_angle(m0),
            dl=mp_policy.cast_to_angle(dl),
            dm=mp_policy.cast_to_angle(dm)
        )

    def predict(
            self,
            visibility_coords: VisibilityCoords,
            gain_model: GainModel | None,
            near_field_delay_engine: BaseNearFieldDelayEngine,
            far_field_delay_engine: BaseFarFieldDelayEngine,
            geodesic_model: BaseGeodesicModel
    ) -> ComplexArray:
        _a1 = visibility_coords.antenna_1  # [B]
        _a2 = visibility_coords.antenna_2  # [B]

        if self.is_full_stokes():
            out_mapping = "[T,~B,C,~P,~Q]"
        else:
            out_mapping = "[T,~B,C]"

        _a1 = visibility_coords.antenna_1
        _a2 = visibility_coords.antenna_2

        # Data will be sharded over frequency so don't reduce over these dimensions, or else communication happens.
        # We want the outer broadcast to be over chan, so we'll do this order.

        @partial(
            multi_vmap,
            in_mapping=f"[T,B,3],[C],[T]",
            out_mapping=out_mapping,
            verbose=True
        )
        def compute_baseline_visibilities_fits(uvw, freq, time):
            """
            Compute visibilities for a single row, channel, accumulating over sources.

            Args:
                uvw: [B, 3]
                freq: []
                time: []

            Returns:
                vis_accumulation: [B, 2, 2] visibility for given baseline, accumulated over all provided directions.
            """

            model_data = self.get_model_slice(
                freq=freq,
                time=time,
                geodesic_model=geodesic_model
            )  # [facet, num_l, num_m[, 2, 2]]

            if gain_model is not None:
                n_phased = jnp.sqrt(1. - model_data.l0 ** 2 - model_data.m0 ** 2)
                lmn_phased = jnp.stack([model_data.l0, model_data.m0, n_phased], axis=-1)  # [facet, 3]
                lmn_geodesic = geodesic_model.compute_far_field_geodesic(
                    times=time[None],
                    lmn_sources=lmn_phased
                )  # [1, num_ant, facet, 3]
                # Compute the gains
                gains = gain_model.compute_gain(
                    freqs=freq[None],
                    times=time[None],
                    lmn_geodesic=lmn_geodesic,
                )  # [1, num_ant, 1, facet,[, 2, 2]]
                g1 = gains[0, visibility_coords.antenna_1, 0, :, ...]  # [B, facet[, 2, 2]]
                g1 = jnp.moveaxis(g1, 1, 0)  # [facet, B[, 2, 2]]
                g2 = gains[0, visibility_coords.antenna_2, 0, :, ...]  # [B, facet[, 2, 2]]
                g2 = jnp.moveaxis(g2, 1, 0)  # [facet, B[, 2, 2]]
            else:
                g1 = g2 = None

            if self.is_full_stokes():
                image_mapping = "[S,Nl,Nm,2,2]"
                gain_mapping = "[S,B,2,2]"
                out_mapping = "[~B,~P,~Q]"
            else:
                image_mapping = "[S,Nl,Nm]"
                gain_mapping = "[S,B]"
                out_mapping = "[~B]"

            @partial(
                multi_vmap,
                in_mapping=f"[B,3],{gain_mapping},{gain_mapping},{image_mapping},[S],[S],[S]",
                out_mapping=out_mapping,
                scan_dims={'S'},
                verbose=True
            )
            def compute_visibilities_fits_over_sources(uvw, g1, g2, image,
                                                       l0, m0, dl, dm):
                """
                Compute visibilities for a single direction, accumulating over sources.

                Args:
                    uvw: [B, 3]
                    g1: [S, B[, 2, 2]]
                    g2: [S, B[, 2, 2]]
                    image: [S,Nl,Nm[, 2, 2]]
                    l0: [S]
                    m0: [S]
                    dl: [S]
                    dm: [S]

                Returns:
                    vis_accumulation: [B[, 2, 2]] visibility for given baseline, accumulated over all provided directions.
                """
                num_baselines = np.shape(uvw)[0]

                def body_fn(accumulate, x):
                    (g1, g2, image, l0, m0, dl, dm) = x
                    delta = self._single_compute_visibilty(uvw, g1, g2, freq, image,
                                                           l0, m0, dl, dm)  # [[2,2]]
                    accumulate += mp_policy.cast_to_vis(delta)
                    return accumulate, ()

                xs = (g1, g2, image, l0, m0, dl, dm)
                init_accumulate = jnp.zeros((num_baselines, 2, 2) if self.is_full_stokes() else (num_baselines,),
                                            dtype=mp_policy.vis_dtype)
                vis_accumulation, _ = lax.scan(body_fn, init_accumulate, xs, unroll=4)
                return vis_accumulation  # [] or [2, 2]

            return compute_visibilities_fits_over_sources(uvw, g1, g2, model_data.image,
                                                          model_data.l0, model_data.m0, model_data.dl, model_data.dm
                                                          )

        visibilities = compute_baseline_visibilities_fits(
            visibility_coords.uvw,
            visibility_coords.freqs,
            visibility_coords.times
        )  # [num_times, num_baselines, num_freqs[,2, 2]]
        return visibilities

    def _single_compute_visibilty(self, uvw, g1, g2, freq, image, l0, m0, dl, dm):
        """
        Compute the visibility from a single direction for a single baseline.

        Args:
            uvw: [B, 3]
            g1: [B, 2, 2]
            g2: [B, 2, 2]
            freq: []
            image: [Nl,Nm[,2, 2]]
            l0: []
            m0: []
            dl: []
            dm: []

        Returns:
            [B[,2, 2]] visibility in given direction for given baseline.
        """

        if self.convention == 'engineering':
            uvw = jnp.negative(uvw)

        if self.is_full_stokes():
            image_mapping = "[Nl,Nm,P,Q]"
            out_mapping = "[~B,P,Q]"
        else:
            image_mapping = "[Nl,Nm]"
            out_mapping = "[~B]"

        @partial(
            multi_vmap,
            in_mapping=f"{image_mapping}",
            out_mapping=out_mapping,
        )
        def wgridder_per_polarisation(image):
            return wgridder.image_to_vis(
                uvw=uvw,
                freqs=freq[None],
                dirty=image,
                pixsize_m=dm,
                pixsize_l=dl,
                center_m=m0,
                center_l=l0,
                epsilon=self.epsilon
            )[:, 0]  # [B]

        vis = wgridder_per_polarisation(image)  # [B[,2,2]]
        if g1 is None and g2 is None:
            return vis

        # Apply the gains
        def apply_gains(vis, g1, g2):
            if self.is_full_stokes():
                return kron_product(g1, vis, g2.conj().T)
            else:
                return g1 * vis * g2.conj()

        vis = jax.vmap(apply_gains)(vis, g1, g2)  # [B[,2,2]]
        return vis

    def plot(self, save_file: str = None, phase_tracking: ac.ICRS | None = None):
        """
        Plot the sky model.

        Args:
            save_file: the file to save the plot to
            phase_tracking: the phase tracking to use for the plot
        """
        # Use imshow to plot the sky model evaluated over a LM grid
        if phase_tracking is None:
            # Won't work near poles or RA=0
            ra0 = np.mean(self.ra) * au.rad
            dec0 = np.mean(self.dec) * au.rad
            phase_tracking = ac.ICRS(ra=ra0, dec=dec0)

        # Plot the first facet
        l, m, n = perley_lmn_from_icrs(self.ra, self.dec, phase_tracking.ra.rad,
                                       phase_tracking.dec.rad)  # [num_model_freqs, facet]
        l0 = l[0, :]
        m0 = m[0, :]
        dl = self.dl[0, :]
        dm = self.dm[0, :]
        if self.is_full_stokes():
            image = self.image[0, :, :, :, 0, 0]  # [facet, num_l, num_m, [2,2]]
        else:
            image = self.image[0, :, :, :]  # [facet, num_l, num_m]
        num_l, num_m = np.shape(image)[-2:]
        lvec = (-0.5 * num_l + np.arange(num_l)) * dl[:, None] + l0[:, None]  # [facet, num_l]
        mvec = (-0.5 * num_m + np.arange(num_m)) * dm[:, None] + m0[:, None]  # [facet, num_m]

        # Evaluate over LM

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

        for facet_idx in range(np.shape(image)[0]):
            im = axs.imshow(
                image[facet_idx, :, :].T,
                origin='lower',
                extent=(lvec[facet_idx, 0], lvec[facet_idx, -1], mvec[facet_idx, 0], mvec[facet_idx, -1]),
                interpolation='nearest'
            )
        axs.set_xlim(lvec.min(), lvec.max())
        axs.set_ylim(mvec.min(), mvec.max())
        # colorbar
        plt.colorbar(im, ax=axs)
        axs.set_xlabel('l [proj.rad]')
        axs.set_ylabel('m [proj.rad]')
        if save_file is not None:
            plt.savefig(save_file)
        plt.show()


# register pytree

def base_fits_source_model_flatten(model: BaseFITSSourceModel):
    return (
        [
            model.model_freqs,
            model.image,
            model.ra,
            model.dec,
            model.dl,
            model.dm
        ],
        (
            model.convention,
            model.epsilon
        )
    )


def base_fits_source_model_unflatten(aux_data, children):
    (model_freqs, image, ra, dec, dl, dm) = children
    (convention, epsilon) = aux_data
    return BaseFITSSourceModel(
        model_freqs=model_freqs,
        image=image,
        ra=ra,
        dec=dec,
        dl=dl,
        dm=dm,
        convention=convention,
        epsilon=epsilon,
        skip_post_init=True
    )


jax.tree_util.register_pytree_node(
    BaseFITSSourceModel,
    base_fits_source_model_flatten,
    base_fits_source_model_unflatten
)


def build_fits_source_model(
        model_freqs: au.Quantity,  # [num_model_freqs] Frequencies
        image: au.Quantity,  # [ num_model_freqs, facets,[,2,2]] Flux amplitude of the source
        ra: au.Quantity,  # [num_model_freqs,facets] central ra coordinate of the facet
        dec: au.Quantity,  # [num_model_freqs,facets] central dec coordinate of the facet
        dl: au.Quantity,  # [num_model_freqs,facets] width of pixel in l
        dm: au.Quantity,  # [num_model_freqs,facets] width of pixel in m
        epsilon: float = 1e-6,
        convention: str = 'physical'
) -> BaseFITSSourceModel:
    """
    Build a point source model.

    Args:
        model_freqs: [num_model_freq] Frequencies
        image: [ num_model_freqs, facets,[,2,2]] Flux amplitude of the source
        ra: [num_model_freqs,facets] central ra coordinate of the facet
        dec: [num_model_freqs,facets] central dec coordinate of the facet
        dl: [num_model_freqs,facets] width of pixel in l
        dm: [num_model_freqs,facets] width of pixel in m
        epsilon: the epsilon value for the wgridder
        convention: the convention for the uvw

    Returns:
        the FITSSourceModel
    """
    sort_order = np.argsort(model_freqs)
    model_freqs = quantity_to_jnp(model_freqs[sort_order], 'Hz')
    image = quantity_to_jnp(image[sort_order], 'Jy')
    ra = quantity_to_jnp(ra[sort_order], 'rad')
    dec = quantity_to_jnp(dec[sort_order], 'rad')
    dl = quantity_to_jnp(dl[sort_order], 'rad')
    dm = quantity_to_jnp(dm[sort_order], 'rad')

    return BaseFITSSourceModel(
        model_freqs=model_freqs,
        image=image,
        ra=ra,
        dec=dec,
        dl=dl,
        dm=dm,
        epsilon=epsilon,
        convention=convention
    )


def build_fits_source_model_from_wsclean_components(
        wsclean_fits_files: List[str],
        model_freqs: au.Quantity,
        full_stokes: bool = True,
        repoint_centre: ac.ICRS | None = None,
        crop_box_size: au.Quantity | None = None,
        num_facets_per_side: int = 1
) -> BaseFITSSourceModel:
    """
    Create a FitsSourceModel from a wsclean model file.

    Args:
        wsclean_fits_files: list of tuples of frequency and fits file
        model_freqs: [num_model_freqs] the frequencies desired. Note: these might be different from what is produced.
        full_stokes: whether the images are full stokes
        repoint_centre: the repoint centre of image if provided
        num_facets_per_side: the number of facets per side

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

    select_idx = select_interpolation_points(
        desired_freqs=quantity_to_np(model_freqs, 'MHz'),
        model_freqs=quantity_to_np(available_freqs, 'MHz')
    )
    # hereafter model freqs is the selected freqs
    model_freqs = available_freqs[select_idx]

    images = []
    ras = []
    decs = []
    dls = []
    dms = []

    for freq_idx in range(len(model_freqs)):
        # interpolate between the two closest frequencies
        select_file_idx = select_idx[freq_idx]
        freq, fits_file = wsclean_fits_freqs_and_fits[select_file_idx]
        with fits.open(fits_file) as hdul0:
            # image = hdul0[0].data.T[:, :, 0, 0].T # [Nm, Nl]
            if np.shape(hdul0[0].data)[1] > 1:
                raise ValueError(f"Expected 1 FREQ parameter, got {np.shape(hdul0[0].data)[1]}")
            image = hdul0[0].data[:, 0, :, :]  # [stokes, Nm, Nl]
            w0 = WCS(hdul0[0].header)
            image = au.Quantity(image, 'Jy')  # [stokes, Nm, Nl]
            # Reverse l axis, so l is increasing
            image = image[:, :, ::-1]  # [stokes, Nm, Nl]
            # Transpose
            image = image.T  # [Nl, Nm, stokes]
            Nl, Nm, num_stokes = image.shape
            if Nl % 2 != 0 or Nm % 2 != 0:
                raise ValueError("Nl and Nm must be even.")
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

            centre_l_pix, centre_m_pix = Nl // 2, Nm // 2  # 0 1 2 3 -> 2 (not 1.5. To make facetting work)
            # Assume pointing is same for all stokes
            pointing_coord, spectral_coord, stokes_coord = w0.pixel_to_world(
                centre_l_pix, centre_m_pix, 0, 0
            )
            if repoint_centre is None:
                # Use WCS conversion instead using crval directly to avoid issues with convention-breaking FITS files
                # I.e. CRVAL might not always the centre of the image
                # ra0, dec0 = w0.wcs.crval[0], w0.wcs.crval[1]
                center_icrs = pointing_coord.transform_to(ac.ICRS)
            else:
                center_icrs = repoint_centre

            # Extract centre ra/dec
            ra0 = center_icrs.ra.to('rad')
            dec0 = center_icrs.dec.to('rad')
            # Order is l, m, freq, stokes
            dl = -pixel_size_l.to('rad')  # negative pixel size
            dm = pixel_size_m.to('rad')

            if crop_box_size is not None:
                # crop to box of given size
                if not crop_box_size.unit.is_equivalent(au.rad):
                    raise ValueError(f"crop_box_size must be in radians, got {crop_box_size.unit}")
                ra_right, dec_right = ac.offset_by(ra0, dec0, np.pi / 2 * au.rad, 0.5 * crop_box_size)
                ra_left, dec_left = ac.offset_by(ra0, dec0, -np.pi / 2 * au.rad, 0.5 * crop_box_size)
                ra_top, dec_top = ac.offset_by(ra0, dec0, 0 * au.rad, 0.5 * crop_box_size)
                ra_bottom, dec_bottom = ac.offset_by(ra0, dec0, np.pi * au.rad, 0.5 * crop_box_size)
                # Get l,m for right, left, top, bottom
                l_right, m_right, _ = perley_lmn_from_icrs(quantity_to_np(ra_right), quantity_to_np(dec_right),
                                                           quantity_to_np(ra0), quantity_to_np(dec0))
                l_left, m_left, _ = perley_lmn_from_icrs(quantity_to_np(ra_left), quantity_to_np(dec_left),
                                                         quantity_to_np(ra0), quantity_to_np(dec0))
                l_top, m_top, _ = perley_lmn_from_icrs(quantity_to_np(ra_top), quantity_to_np(dec_top),
                                                       quantity_to_np(ra0), quantity_to_np(dec0))
                l_bottom, m_bottom, _ = perley_lmn_from_icrs(quantity_to_np(ra_bottom), quantity_to_np(dec_bottom),
                                                             quantity_to_np(ra0), quantity_to_np(dec0))

                # Get the indices for the box
                lvec = ((-0.5 * Nl + np.arange(Nl)) * dl).to('rad').value
                mvec = ((-0.5 * Nm + np.arange(Nm)) * dm).to('rad').value
                l_left_idx = np.searchsorted(lvec, l_left)
                l_right_idx = np.clip(np.searchsorted(lvec, l_right, side='right') - 1, 0, Nl - 1)
                l_slice = slice(l_left_idx, l_right_idx)
                m_bottom_idx = np.clip(np.searchsorted(mvec, m_bottom, side='right') - 1, 0, Nm - 1)
                m_top_idx = np.searchsorted(mvec, m_top)
                m_slice = slice(m_bottom_idx, m_top_idx)
                image = image[l_slice, m_slice, :]  # [Nl, Nm, stokes]
                Nl, Nm, num_stokes = image.shape

            print(
                f"freq={freq.to('MHz')}, dl={pixel_size_l.to('rad')}, dm={pixel_size_m.to('rad')}\n"
                f"centre_ra={ra0}, centre_dec={dec0}\n"
                f"centre_l_pix={centre_l_pix}, centre_m_pix={centre_m_pix}\n"
                f"num_l={Nl}, num_m={Nm}, num_stokes={num_stokes}"
            )

            if full_stokes:
                from_corrs = tuple(stokes_coord.symbol.tolist())
                image = np.asarray(
                    broadcast_translate_corrs(
                        quantity_to_jnp(image, "Jy"),
                        from_corrs=from_corrs,
                        to_corrs=(("XX", "XY"), ("YX", "YY"))
                    )
                ) * au.Jy
            else:
                # Image is in stokes I, so we can just take the first element
                from_corrs = tuple(stokes_coord.symbol.tolist())
                image = np.asarray(
                    broadcast_translate_corrs(
                        quantity_to_jnp(image, "Jy"),
                        from_corrs=from_corrs,
                        to_corrs=("I",)
                    )[:, :, 0]  # Take the I component
                ) * au.Jy

        images.append(image)
        ras.append(ra0)
        decs.append(dec0)
        dls.append(dl)
        dms.append(dm)

    images = au.Quantity(np.stack(images, axis=0))  # [num_model_freqs, num_l, num_m, [2,2]]
    ras = au.Quantity(ras)  # [num_model_freqs]
    decs = au.Quantity(decs)  # [num_model_freqs]
    dls = au.Quantity(dls)  # [num_model_freqs]
    dms = au.Quantity(dms)  # [num_model_freqs]

    images, ras, decs, dls, dms = divide_fits_into_facets(images, ras, decs, dls, dms, num_facets_per_side)

    return build_fits_source_model(
        model_freqs=model_freqs,
        image=images,
        ra=ras,
        dec=decs,
        dl=dls,
        dm=dms
    )


def divide_fits_into_facets(images: au.Quantity, ras: au.Quantity, decs: au.Quantity, dls: au.Quantity,
                            dms: au.Quantity, num_facets_per_side: int):
    """
    Divide the fits images into facets, assuming l0=m0=0 in the centre of the input image.

    Args:
        images: [num_model_freqs, num_l, num_m, [2,2]] in [[xx, xy], [yx, yy]] format or stokes I
        ras: [num_model_freqs] the central ra coordinate of the image
        decs: [num_model_freqs] the central dec coordinate of the image
        dls: [num_model_freqs] the width of pixel in l
        dms: [num_model_freqs] the width of pixel in m
        num_facets_per_side: int, number of facets per side

    Returns:
        images: [num_model_freqs, num_facets, num_l_facet, num_m_facet, [2,2]] in [[xx, xy], [yx, yy]] format or stokes I
        ras: [num_model_freqs, num_facets] the central ra coordinate of the facet
        decs: [num_model_freqs, num_facets] the central dec coordinate of the facet
        dls: [num_model_freqs, num_facets] the width of pixel in l
        dms: [num_model_freqs, num_facets] the width of pixel in m
    """
    # Now break up into facets.
    num_l, num_m = np.shape(images)[1:3]
    # check if we can factor into num_facets_per_side facets
    if num_l % num_facets_per_side != 0 or num_m % num_facets_per_side != 0:
        raise ValueError(f"num_l ({num_l}) and num_m ({num_m}) "
                         f"must be divisible by num_facets_per_side={num_facets_per_side}")
    # Create slices per facet
    slices = []
    num_l_facet = num_l // num_facets_per_side
    num_m_facet = num_m // num_facets_per_side
    if num_l_facet % 2 != 0 or num_m_facet % 2 != 0:
        raise ValueError(f"Size of each resulting facet must be even, got {(num_l_facet, num_m_facet)}.")
    for l_idx in range(num_facets_per_side):
        for m_idx in range(num_facets_per_side):
            l_start = l_idx * num_l_facet
            l_end = (l_idx + 1) * num_l_facet
            m_start = m_idx * num_m_facet
            m_end = (m_idx + 1) * num_m_facet
            slices.append((slice(l_start, l_end), slice(m_start, m_end)))
    # Get slices for each facet
    facet_images = []
    facet_ras = []
    facet_decs = []
    facet_dls = []
    facet_dms = []
    # l0=m0=0 in centre of full image
    lvec = (-0.5 * num_l + np.arange(num_l)) * dls[:, None]  # [num_model_freqs, num_l]
    mvec = (-0.5 * num_m + np.arange(num_m)) * dms[:, None]  # [num_model_freqs, num_m]
    for l_slice, m_slice in slices:
        facet_images.append(images[:, l_slice, m_slice, ...])  # [num_model_freqs, num_l_facet, num_m_facet, [2,2]]
        lvec_facet = lvec[:, l_slice]  # [num_model_freqs, num_l_facet]
        mvec_facet = mvec[:, m_slice]  # [num_model_freqs, num_m_facet]
        # 0 1 2 3 -> 2 , 4//2, so l0 is seen as N//2
        l0_facet = lvec_facet[:, num_l_facet // 2]  # [num_model_freqs]
        m0_facet = mvec_facet[:, num_m_facet // 2]  # [num_model_freqs]
        n0_facet = np.sqrt(1 - l0_facet.to('rad').value ** 2 - m0_facet.to('rad').value ** 2) * au.rad
        ra0_facet, dec0_facet = perley_icrs_from_lmn(
            quantity_to_jnp(l0_facet, 'rad'),
            quantity_to_jnp(m0_facet, 'rad'),
            quantity_to_jnp(n0_facet, 'rad'),
            quantity_to_jnp(ras, 'rad'),
            quantity_to_jnp(decs, 'rad')
        )  # [num_model_freqs]

        facet_ras.append(ra0_facet * au.rad)  # [num_model_freqs]
        facet_decs.append(dec0_facet * au.rad)  # [num_model_freqs]
        facet_dls.append(dls)  # [num_model_freqs]
        facet_dms.append(dms)  # [num_model_freqs]
    images = au.Quantity(
        np.stack(facet_images, axis=1)
    )  # [num_model_freqs, num_facets, num_l_facet, num_m_facet, [2,2]]
    ras = au.Quantity(np.stack(facet_ras, axis=1))  # [num_model_freqs, num_facets]
    decs = au.Quantity(np.stack(facet_decs, axis=1))  # [num_model_freqs, num_facets]
    dls = au.Quantity(np.stack(facet_dls, axis=1))  # [num_model_freqs, num_facets]
    dms = au.Quantity(np.stack(facet_dms, axis=1))  # [num_model_freqs, num_facets]
    return images, ras, decs, dls, dms
