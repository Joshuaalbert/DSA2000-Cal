import dataclasses
import pickle
import warnings
from functools import partial
from typing import List, Tuple, Any

import astropy.coordinates as ac
import astropy.units as au
import jax
import numpy as np
import pylab as plt
from astropy.io import fits
from astropy.wcs import WCS
from jax import numpy as jnp, lax

from dsa2000_cal.common import wgridder
from dsa2000_cal.common.array_types import ComplexArray, FloatArray
from dsa2000_cal.common.corr_utils import broadcast_translate_corrs
from dsa2000_cal.common.interp_utils import InterpolatedArray, select_interpolation_points
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.pure_callback_utils import construct_threaded_callback
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np
from dsa2000_cal.common.types import VisibilityCoords
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_common.delay_models import BaseFarFieldDelayEngine
from dsa2000_common.delay_models import BaseNearFieldDelayEngine
from dsa2000_common.delay_models import perley_lmn_from_icrs, perley_icrs_from_lmn
from dsa2000_common.gain_models import GainModel
from dsa2000_common.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_cal.visibility_model.source_models.abc import AbstractSourceModel


def _prepare_model(freq, time, antenna1, antenna2, model_freqs, image, ra, dec, dl, dm,
                   geodesic_model: BaseGeodesicModel,
                   gain_model: GainModel | None, is_full_stokes: bool):
    interp = InterpolatedArray(x=model_freqs, values=(image, ra, dec, dl, dm),
                               axis=0)
    _image, _ra, _dec, _dl, _dm = interp(freq)  # [num_l, num_m, [2,2]], [],...

    num_l, num_m = np.shape(_image)[:2]
    lmn = geodesic_model.compute_far_field_lmn(_ra, _dec, time, return_elevation=False)
    l0, m0 = lmn[:2]
    lvec = (-0.5 * num_l + jnp.arange(num_l)) * _dl + l0
    mvec = (-0.5 * num_m + jnp.arange(num_m)) * _dm + m0
    L, M = jnp.meshgrid(lvec, mvec, indexing='ij')
    L2M2 = L ** 2 + M ** 2
    N = jnp.sqrt(1 - L2M2)
    LMN = jnp.stack([L, M, N], axis=-1)  # [num_l, num_m, 3]
    elevation = geodesic_model.compute_elevation_from_lmn(LMN, time)  # [num_l, num_m]
    elevation_mask = elevation <= 0
    n_mask = L2M2 > 1
    mask = jnp.logical_or(elevation_mask, n_mask)
    if is_full_stokes:
        masked_image = jnp.where(mask[:, :, None, None], 0, _image)  # [Nl,Nm[,2, 2]]
    else:
        masked_image = jnp.where(mask, 0, _image)  # [Nl,Nm[,2, 2]]

    lmn_geodesic = geodesic_model.compute_far_field_geodesic(
        times=time[None],
        lmn_sources=lmn[None, :]
    )  # [1, num_ant, 1, 3]
    if gain_model is not None:
        # Compute the gains
        gains = gain_model.compute_gain(
            freqs=freq[None],
            times=time[None],
            lmn_geodesic=lmn_geodesic,
        )  # [1, num_ant, 1, 1,[, 2, 2]]
        g1 = gains[0, antenna1, 0, 0, ...]  # [B[, 2, 2]]
        g2 = gains[0, antenna2, 0, 0, ...]  # [B[, 2, 2]]
    else:
        g1 = g2 = None
    return g1, g2, masked_image, l0, m0, _dl, _dm


_prepare_model_jit = jax.jit(_prepare_model, static_argnames=['is_full_stokes'])


@dataclasses.dataclass(eq=False)
class BaseFITSSourceModel(AbstractSourceModel):
    """
    Predict vis for fits source.
    """
    model_freqs: FloatArray  # [num_model_freqs] Frequencies
    image: FloatArray  # [facet,num_model_freqs, num_l, num_m, [2,2]] in [[xx, xy], [yx, yy]] format or stokes I
    ra: FloatArray  # [facet,num_model_freqs] central ra coordinate of the facet
    dec: FloatArray  # [facet,num_model_freqs] central dec coordinate of the facet
    dl: FloatArray  # [facet,num_model_freqs] width of pixel in l
    dm: FloatArray  # [facet,num_model_freqs] width of pixel in m

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
        if np.shape(self.ra)[1] != np.shape(self.model_freqs)[0]:
            raise ValueError("ra must have same length as model_freqs")
        if np.shape(self.image)[:2] != np.shape(self.ra):
            raise ValueError(f"image[:2] shape must be {np.shape(self.ra)}, got {np.shape(self.image)}")
        num_l, num_l = np.shape(self.image)[2:4]
        if num_l % 2 != 0 or num_l % 2 != 0:
            raise ValueError("num_l and num_m must be even")

    def is_full_stokes(self) -> bool:
        return len(np.shape(self.image)) == 6 and np.shape(self.image)[-2:] == (2, 2)

    def predict_np(
            self,
            visibility_coords: VisibilityCoords,
            gain_model: GainModel,
            near_field_delay_engine: BaseNearFieldDelayEngine,
            far_field_delay_engine: BaseFarFieldDelayEngine,
            geodesic_model: BaseGeodesicModel
    ) -> ComplexArray:

        # convert coords to np
        visibility_coords = jax.tree.map(np.asarray, visibility_coords)

        def scan_np(f, init, xs):
            carry = init

            map_size = set(np.shape(x)[0] for x in jax.tree.leaves(xs))
            if len(map_size) != 1:
                raise ValueError("All inputs must have the same size")
            map_size = list(map_size)[0]
            for i in range(map_size):
                x = jax.tree.map(lambda x: np.asarray(x[i]), xs)
                carry, _ = f(carry, x)
            return carry, ()

        # Uses a similar approach to predict, but with numpy instead of JAX, which allows usin wgridder direct access to buffers rather than duplicating buffers, which causes blow-up.
        def body_fn(accumulate, x):
            (ra, dec, dl, dm, image) = x

            def compute_visibilities_fits_single_source(freq, time, uvw, antenna1, antenna2):
                g1, g2, masked_image, l0, m0, _dl, _dm = jax.tree.map(
                    np.asarray,
                    _prepare_model_jit(
                        freq, time, antenna1, antenna2,
                        self.model_freqs, image, ra, dec, dl, dm,
                        geodesic_model, gain_model, self.is_full_stokes()
                    )
                )
                return self._single_compute_visibilty_np(
                    uvw, g1, g2, freq, masked_image, l0, m0, _dl, _dm
                )  # [B[,2, 2]]

            # Kernel input shapes:
            # freq: []
            # time: []
            # uvw: [B, 3]
            # antenna1: [B]
            # antenna2: [B]
            freqs = visibility_coords.freqs[None, :]  # [1, C]
            times = visibility_coords.times[:, None]  # [T, 1]
            uvw = visibility_coords.uvw[:, None, :, :]  # [T, 1, B, 3]
            antenna1 = visibility_coords.antenna1[None, None, :]  # [1, 1, B]
            antenna2 = visibility_coords.antenna2[None, None, :]  # [1, 1, B]

            total_num_execs = T * C * (4 if self.is_full_stokes() else 1)
            num_threads = min(32, total_num_execs)
            num_threads_inner = (4 if self.is_full_stokes() else 1)
            num_threads_outer = max(1, num_threads // num_threads_inner)
            cb = construct_threaded_callback(compute_visibilities_fits_single_source, 0, 0, 2, 1, 1,
                                             num_threads=num_threads_outer)

            delta = cb(freqs, times, uvw, antenna1, antenna2)  # [T, C, B[, 2, 2]]
            # out_mapping = '[T,~B,C[,~P,~Q]]'
            delta = np.moveaxis(delta, 2, 1)  # [T, B, C[, 2, 2]]
            accumulate += mp_policy.cast_to_vis(delta)
            return accumulate, ()

        T = np.shape(visibility_coords.times)[0]
        B = np.shape(visibility_coords.antenna1)[0]
        C = np.shape(visibility_coords.freqs)[0]
        if self.is_full_stokes():
            init_accumulate = np.zeros((T, B, C, 2, 2), dtype=mp_policy.vis_dtype)

        else:
            init_accumulate = np.zeros((T, B, C), dtype=mp_policy.vis_dtype)
        xs = jax.tree.map(np.asarray, (self.ra, self.dec, self.dl, self.dm, self.image))
        # sum over sources
        vis_accumulation, _ = scan_np(body_fn, init_accumulate, xs)
        return vis_accumulation  # [T, B, C[, 2, 2]]

    def predict(
            self,
            visibility_coords: VisibilityCoords,
            gain_model: GainModel | None,
            near_field_delay_engine: BaseNearFieldDelayEngine,
            far_field_delay_engine: BaseFarFieldDelayEngine,
            geodesic_model: BaseGeodesicModel
    ) -> ComplexArray:
        def body_fn(accumulate, x):
            (ra, dec, dl, dm, image) = x

            # vmap over time and freqs
            if self.is_full_stokes():
                out_mapping = '[T,~B,C,~P,~Q]'
            else:
                out_mapping = '[T,~B,C]'

            @partial(
                multi_vmap,
                in_mapping=f'[C],[T],[T,B,3],[B],[B]',
                out_mapping=out_mapping,
                scan_dims={'C', 'T'},
                verbose=True
            )
            def compute_visibilities_fits_single_source(freq, time, uvw, antenna1, antenna2):
                g1, g2, masked_image, l0, m0, _dl, _dm = _prepare_model(
                    freq, time, antenna1, antenna2,
                    self.model_freqs, image, ra, dec, dl, dm,
                    geodesic_model, gain_model, self.is_full_stokes()
                )
                return self._single_compute_visibilty(uvw, g1, g2, freq, masked_image, l0, m0, _dl, _dm)  # [B[,2, 2]]

            delta = compute_visibilities_fits_single_source(
                visibility_coords.freqs,
                visibility_coords.times,
                visibility_coords.uvw,
                visibility_coords.antenna1,
                visibility_coords.antenna2
            )

            accumulate += mp_policy.cast_to_vis(delta)
            return accumulate, ()

        T = np.shape(visibility_coords.times)[0]
        B = np.shape(visibility_coords.antenna1)[0]
        C = np.shape(visibility_coords.freqs)[0]
        if self.is_full_stokes():
            init_accumulate = jnp.zeros((T, B, C, 2, 2), dtype=mp_policy.vis_dtype)

        else:
            init_accumulate = jnp.zeros((T, B, C), dtype=mp_policy.vis_dtype)
        xs = (self.ra, self.dec, self.dl, self.dm, self.image)

        # sum over sources
        vis_accumulation, _ = lax.scan(body_fn, init_accumulate, xs)
        return vis_accumulation  # [T, B, C[, 2, 2]]

    def _single_compute_visibilty_np(self, uvw, g1, g2, freq, image, l0, m0, dl, dm):
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
        uvw = np.array(uvw)
        if self.convention == 'engineering':
            uvw = np.negative(uvw)

        def wgridder_per_polarisation(image):
            return wgridder.image_to_vis_np(
                uvw=uvw,
                freqs=freq[None],
                dirty=image,
                pixsize_m=dm,
                pixsize_l=dl,
                center_m=m0,
                center_l=l0,
                epsilon=self.epsilon,
                num_threads=1
            )[:, 0]  # [B]

        if self.is_full_stokes():
            # image_mapping = "[P,Q,Nl,Nm]"
            # out_mapping = "[~B,P,Q]"
            image = np.transpose(image, (2, 3, 0, 1))  # [P,Q,Nl,Nm]
        else:
            # image_mapping = "[Nl,Nm]"
            # out_mapping = "[~B]"
            pass

        cb = construct_threaded_callback(wgridder_per_polarisation, 2, num_threads=4 if self.is_full_stokes() else 1)
        vis = cb(image)  # [[2, 2],B]
        vis = np.moveaxis(vis, -1, 0)  # [B, [2, 2]]

        if g1 is None and g2 is None:
            return vis

        # Apply the gains
        def apply_gains(vis, g1, g2):
            if self.is_full_stokes():
                return kron_product(g1, vis, g2.conj().T)
            else:
                return g1 * vis * g2.conj()

        vis = jax.jit(jax.vmap(apply_gains))(vis, g1, g2)  # [B[,2,2]]
        return np.asarray(vis)

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

    def plot(self, save_file: str = None, phase_center: ac.ICRS | None = None):
        """
        Plot the sky model.

        Args:
            save_file: the file to save the plot to
            phase_center: the phase tracking to use for the plot
        """
        # Use imshow to plot the sky model evaluated over a LM grid
        if phase_center is None:
            # Won't work near poles or RA=0
            ra0 = np.mean(self.ra) * au.rad
            dec0 = np.mean(self.dec) * au.rad
            phase_center = ac.ICRS(ra=ra0, dec=dec0)

        # Plot the first facet
        l, m, n = perley_lmn_from_icrs(self.ra, self.dec, phase_center.ra.rad,
                                       phase_center.dec.rad)  # [num_model_freqs, facet]
        l0 = l[:, 0]
        m0 = m[:, 0]
        dl = self.dl[:, 0]
        dm = self.dm[:, 0]
        if self.is_full_stokes():
            image = self.image[:, 0, :, :, 0, 0]  # [facet, num_l, num_m, [2,2]]
        else:
            image = self.image[:, 0, :, :]  # [facet, num_l, num_m]
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

    def save(self, filename: str):
        """
        Serialise the model to file.

        Args:
            filename: the filename
        """
        if not filename.endswith('.pkl'):
            warnings.warn(f"Filename {filename} does not end with .pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        """
        Load the model from file.

        Args:
            filename: the filename

        Returns:
            the model
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        children, aux_data = self.flatten(self)
        children_np = jax.tree.map(np.asarray, children)
        serialised = (aux_data, children_np)
        return (self._deserialise, (serialised,))

    @classmethod
    def _deserialise(cls, serialised):
        # Create a new instance, bypassing __init__ and setting the actor directly
        (aux_data, children_np) = serialised
        children_jax = jax.tree.map(jnp.asarray, children_np)
        return cls.unflatten(aux_data, children_jax)

    @classmethod
    def register_pytree(cls):
        jax.tree_util.register_pytree_node(cls, cls.flatten, cls.unflatten)

    # an abstract classmethod
    @classmethod
    def flatten(cls, this: "BaseFITSSourceModel") -> Tuple[List[Any], Tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        return (
            [
                this.model_freqs,
                this.image,
                this.ra,
                this.dec,
                this.dl,
                this.dm
            ],
            (
                this.convention,
                this.epsilon
            )
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "BaseFITSSourceModel":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
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


BaseFITSSourceModel.register_pytree()


def build_fits_source_model(
        model_freqs: au.Quantity,
        image: au.Quantity,
        ra: au.Quantity,
        dec: au.Quantity,
        dl: au.Quantity,
        dm: au.Quantity,
        epsilon: float = 1e-6,
        convention: str = 'physical'
) -> BaseFITSSourceModel:
    """
    Build a point source model.

    Args:
        model_freqs: [num_model_freq] Frequencies
        image: [facets,num_model_freqs, num_l, num_m, [2,2]] in [[xx, xy], [yx, yy]] Flux amplitude of the source
        ra: [facets,num_model_freqs] central ra coordinate of the facet
        dec: [facets,num_model_freqs] central dec coordinate of the facet
        dl: [facets,num_model_freqs] width of pixel in l
        dm: [facets,num_model_freqs] width of pixel in m
        epsilon: the epsilon value for the wgridder
        convention: the convention for the uvw

    Returns:
        the FITSSourceModel
    """
    sort_order = np.argsort(model_freqs)
    model_freqs = quantity_to_jnp(model_freqs[sort_order], 'Hz')
    image = quantity_to_jnp(image[:, sort_order], 'Jy')
    ra = quantity_to_jnp(ra[:, sort_order], 'rad')
    dec = quantity_to_jnp(dec[:, sort_order], 'rad')
    dl = quantity_to_jnp(dl[:, sort_order], 'rad')
    dm = quantity_to_jnp(dm[:, sort_order], 'rad')

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


def build_calibration_fits_source_models_from_wsclean(
        wsclean_fits_files: List[str],
        model_freqs: au.Quantity,
        full_stokes: bool = True,
        repoint_centre: ac.ICRS | None = None,
        crop_box_size: au.Quantity | None = None,
        num_facets: int = 1,
):
    """
    Build a calibration source model from wsclean components.

    Args:
        wsclean_fits_files: the wsclean fits files
        model_freqs: the model frequencies
        full_stokes: whether the model should be full stokes
        repoint_centre: the repoint centre
        crop_box_size: the crop box size
        num_facets: the number of facets

    Returns:
        the calibration source model shaped [
    """
    sky_model = build_fits_source_model_from_wsclean_components(
        wsclean_fits_files=wsclean_fits_files,
        model_freqs=model_freqs,
        full_stokes=full_stokes,
        repoint_centre=repoint_centre,
        crop_box_size=crop_box_size,
        num_facets_per_side=num_facets
    )
    # Turn each facet into a 1 facet sky model
    sky_models = []
    for facet_idx in range(num_facets):
        image = sky_model.image[facet_idx:facet_idx + 1]  # [facet=1,num_model_freqs, num_l, num_m, [2,2]]
        ra = sky_model.ra[facet_idx:facet_idx + 1]  # [facet=1,num_model_freqs]
        dec = sky_model.dec[facet_idx:facet_idx + 1]  # [facet=1,num_model_freqs]
        dl = sky_model.dl[facet_idx:facet_idx + 1]  # [facet=1,num_model_freqs]
        dm = sky_model.dm[facet_idx:facet_idx + 1]  # [facet=1,num_model_freqs]
        sky_models.append(
            BaseFITSSourceModel(
                model_freqs=sky_model.model_freqs,
                image=image,
                ra=ra,
                dec=dec,
                dl=dl,
                dm=dm,
                epsilon=sky_model.epsilon,
                convention=sky_model.convention
            )
        )
    return sky_models


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
                # Assume Hz
                frequency = header['FREQ'] * au.Hz
            elif 'RESTFRQ' in header:
                # Assume Hz
                frequency = header['RESTFRQ'] * au.Hz
            elif 'CRVAL3' in header:  # Assuming the frequency is in the third axis
                if header['CTYPE3'].strip().upper() != 'FREQ':
                    raise ValueError(f"Expected CTYPE3 to be FREQ, got {header['CTYPE3']}")
                unit = header['CUNIT3']
                frequency = au.Quantity(header['CRVAL3'], unit=unit)
            else:
                raise KeyError("Frequency information not found in FITS header.")
            wsclean_fits_freqs_and_fits.append((frequency, fits_file))
    print(f"Found {len(wsclean_fits_freqs_and_fits)} fits files")

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
    print(f"Selecting frequencies: {model_freqs}")

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
            if hdul0[0].header['BUNIT'].upper() == 'JY/PIXEL':
                pass
            elif hdul0[0].header['BUNIT'].upper() == 'JY/BEAM':
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
                l_right, m_right, _ = perley_lmn_from_icrs(quantity_to_np(ra_right),
                                                           quantity_to_np(dec_right),
                                                           quantity_to_np(ra0), quantity_to_np(dec0))
                l_left, m_left, _ = perley_lmn_from_icrs(quantity_to_np(ra_left), quantity_to_np(dec_left),
                                                         quantity_to_np(ra0), quantity_to_np(dec0))
                l_top, m_top, _ = perley_lmn_from_icrs(quantity_to_np(ra_top), quantity_to_np(dec_top),
                                                       quantity_to_np(ra0), quantity_to_np(dec0))
                l_bottom, m_bottom, _ = perley_lmn_from_icrs(quantity_to_np(ra_bottom),
                                                             quantity_to_np(dec_bottom),
                                                             quantity_to_np(ra0),
                                                             quantity_to_np(dec0))

                # Get the indices for the box
                lvec = ((-0.5 * Nl + np.arange(Nl)) * dl).to('rad').value
                mvec = ((-0.5 * Nm + np.arange(Nm)) * dm).to('rad').value

                l_left_idx = np.clip(np.searchsorted(lvec, l_left, side='right') - 1, 0, Nl - 1)
                l_right_idx = np.clip(np.searchsorted(lvec, l_right, side='right') - 1, 0, Nl - 1)
                # ensure even
                if (l_right_idx - l_left_idx) % 2 != 0:
                    l_right_idx += 1
                l_slice = slice(l_left_idx, l_right_idx)
                m_bottom_idx = np.clip(np.searchsorted(mvec, m_bottom, side='right') - 1, 0, Nm - 1)
                m_top_idx = np.clip(np.searchsorted(mvec, m_top, side='right') - 1, 0, Nm - 1)
                # ensure even
                if (m_top_idx - m_bottom_idx) % 2 != 0:
                    m_top_idx += 1
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
        images: [num_facets,num_model_freqs, num_l_facet, num_m_facet, [2,2]] in [[xx, xy], [yx, yy]] format or stokes I
        ras: [num_facets,num_model_freqs] the central ra coordinate of the facet
        decs: [num_facets,num_model_freqs] the central dec coordinate of the facet
        dls: [num_facets,num_model_freqs] the width of pixel in l
        dms: [num_facets,num_model_freqs] the width of pixel in m
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
    pad_l = False
    if num_l_facet % 2 != 0:
        warnings.warn("num_l_facet is not even, padding to make it even.")
        pad_l = True
    pad_m = False
    if num_m_facet % 2 != 0:
        warnings.warn("num_m_facet is not even, padding to make it even.")
        pad_m = True

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
        facet_image = images[:, l_slice, m_slice, ...]  # [num_model_freqs, num_l_facet, num_m_facet, [2,2]]
        lvec_facet = lvec[:, l_slice]  # [num_model_freqs, num_l_facet]
        mvec_facet = mvec[:, m_slice]  # [num_model_freqs, num_m_facet]
        # 0 1 2 3 -> 2 , 4//2, so l0 is seen as N//2
        l0_facet = lvec_facet[:, num_l_facet // 2]  # [num_model_freqs]
        m0_facet = mvec_facet[:, num_m_facet // 2]  # [num_model_freqs]

        if pad_l:
            # 0 1 2 -> 0 1 2 3
            slice_l = facet_image[:, :1, ...]
            facet_image = np.concatenate([facet_image, np.zeros_like(slice_l)], axis=1)
            # l_pix: num_l_facet // 2 ==> 3 // 2 = 1 -> 4 // 2 = 2 = 3 // 2 + 1
            l0_facet += dls
        if pad_m:
            # 0 1 2 -> 0 1 2 3
            slice_m = facet_image[:, :, :1, ...]
            facet_image = np.concatenate([facet_image, np.zeros_like(slice_m)], axis=2)
            # m_pix: num_m_facet // 2 ==> 3 // 2 = 1 -> 4 // 2 = 2 = 3 // 2 + 1
            m0_facet += dms

        n0_facet = np.sqrt(1 - l0_facet.to('rad').value ** 2 - m0_facet.to('rad').value ** 2) * au.rad
        ra0_facet, dec0_facet = perley_icrs_from_lmn(
            quantity_to_jnp(l0_facet, 'rad'),
            quantity_to_jnp(m0_facet, 'rad'),
            quantity_to_jnp(n0_facet, 'rad'),
            quantity_to_jnp(ras, 'rad'),
            quantity_to_jnp(decs, 'rad')
        )  # [num_model_freqs]

        facet_images.append(facet_image)  # [num_model_freqs, num_l_facet, num_m_facet, [2,2]]
        facet_ras.append(ra0_facet * au.rad)  # [num_model_freqs]
        facet_decs.append(dec0_facet * au.rad)  # [num_model_freqs]
        facet_dls.append(dls)  # [num_model_freqs]
        facet_dms.append(dms)  # [num_model_freqs]
    images = au.Quantity(
        np.stack(facet_images, axis=0)
    )  # [num_facets, num_model_freqs, num_l_facet, num_m_facet, [2,2]]]
    ras = au.Quantity(np.stack(facet_ras, axis=0))  # [num_facets,num_model_freqs]
    decs = au.Quantity(np.stack(facet_decs, axis=0))  # [num_facets,num_model_freqs]
    dls = au.Quantity(np.stack(facet_dls, axis=0))  # [num_facets,num_model_freqs]
    dms = au.Quantity(np.stack(facet_dms, axis=0))  # [num_facets,num_model_freqs]
    return images, ras, decs, dls, dms
