import dataclasses
import pickle
import warnings
from functools import partial
from typing import Tuple, List, Any

import astropy.coordinates as ac
import astropy.units as au
import jax
import numpy as np
import pylab as plt
from astropy import constants
from jax import numpy as jnp, lax

from dsa2000_cal.adapter.utils import broadcast_translate_corrs
from dsa2000_cal.common.array_types import ComplexArray, FloatArray
from dsa2000_cal.common.corr_translation import flatten_coherencies, unflatten_coherencies
from dsa2000_cal.common.ellipse_utils import Gaussian
from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.jvp_linear_op import JVPLinearOp
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.types import VisibilityCoords
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.common.wsclean_util import parse_and_process_wsclean_source_line
from dsa2000_cal.delay_models.base_far_field_delay_engine import BaseFarFieldDelayEngine
from dsa2000_cal.delay_models.base_near_field_delay_engine import BaseNearFieldDelayEngine
from dsa2000_cal.delay_models.uvw_utils import perley_lmn_from_icrs
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel
from dsa2000_cal.visibility_model.source_models.abc import AbstractSourceModel


@dataclasses.dataclass(eq=False)
class BaseGaussianSourceModel(AbstractSourceModel):
    """
    Predict vis for gaussian source.
    """
    model_freqs: FloatArray  # [num_model_freqs] Frequencies
    ra: FloatArray  # [num_sources] ra coordinate of the source
    dec: FloatArray  # [num_sources] dec coordinate of the source
    A: FloatArray  # [num_sources, num_model_freqs,[,2,2]] Flux amplitude of the source
    major_axis: FloatArray  # [num_sources] Major axis of the source in proj.radians
    minor_axis: FloatArray  # [num_sources] Minor axis of the source in proj.radians
    pos_angle: FloatArray  # [num_sources] Position angle of the source in proj.radians

    order_approx: int = 0
    convention: str = 'physical'
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return

        # Check shapes match
        if len(np.shape(self.model_freqs)) != 1:
            raise ValueError("model_freqs must be 1D")
        if len(np.shape(self.ra)) != 1:
            raise ValueError("ra must be 1D")
        if np.shape(self.dec) != np.shape(self.ra):
            raise ValueError(f"ra and dec must have the same shape, got {np.shape(self.ra)} and {np.shape(self.dec)}")
        if np.shape(self.major_axis) != np.shape(self.ra):
            raise ValueError(
                f"major_axis {np.shape(self.major_axis)} must have the same shape as ra {np.shape(self.ra)}")
        if np.shape(self.minor_axis) != np.shape(self.ra):
            raise ValueError(
                "minor_axis must {np.shape(self.minor_axis)} have the same shape as ra {np.shape(self.ra)}")
        if np.shape(self.pos_angle) != np.shape(self.ra):
            raise ValueError("pos_angle must {np.shape(self.pos_angle)} have the same shape as ra {np.shape(self.ra)}")
        if np.shape(self.A)[:2] != (len(self.ra), len(self.model_freqs)):
            raise ValueError(
                f"A must have shape [{np.shape(self.model_freqs)[0]}, {np.shape(self.ra)[0]}[,2,2]], got {np.shape(self.A)}")

    def is_full_stokes(self) -> bool:
        return len(np.shape(self.A)) == 4 and np.shape(self.A)[-2:] == (2, 2)

    def predict(
            self,
            visibility_coords: VisibilityCoords,
            gain_model: GainModel | None,
            near_field_delay_engine: BaseNearFieldDelayEngine,
            far_field_delay_engine: BaseFarFieldDelayEngine,
            geodesic_model: BaseGeodesicModel
    ) -> ComplexArray:
        def body_fn(accumulate, x):
            (ra, dec, major_axis, minor_axis, pos_angle, image) = x

            # vmap over time and freqs
            if self.is_full_stokes():
                out_mapping = '[T,B,C,~P,~Q]'
            else:
                out_mapping = '[T,B,C]'

            @partial(
                multi_vmap,
                in_mapping=f'[C],[T],[T,B,3],[B],[B]',
                out_mapping=out_mapping,
                scan_dims={'C', 'T'},
                verbose=True
            )
            def compute_visibilities_gaussian_single_source(freq, time, uvw, antenna1, antenna2):
                lmn, elevation = geodesic_model.compute_far_field_lmn(ra, dec, time, return_elevation=True)  # [3]
                interp = InterpolatedArray(x=self.model_freqs, values=image, axis=0, regular_grid=True,
                                           check_spacing=False)
                masked_image = jnp.where(elevation <= 0, 0, interp(freq))  # [2, 2]
                lmn_geodesic = geodesic_model.compute_far_field_geodesic(
                    times=time[None],
                    lmn_sources=lmn[None, :],
                    antenna_indices=jnp.stack([antenna1, antenna2])
                )  # [1, num_ant=2, 1, 3]
                if gain_model is not None:
                    # Compute the gains
                    gains = gain_model.compute_gain(
                        freqs=freq[None],
                        times=time[None],
                        lmn_geodesic=lmn_geodesic,
                    )  # [1, num_ant=2, 1, 1,[, 2, 2]]
                    g1 = gains[0, 0, 0, 0, ...]  # [[, 2, 2]]
                    g2 = gains[0, 1, 0, 0, ...]  # [[, 2, 2]]
                else:
                    g1 = g2 = None
                return self._single_compute_visibilty(lmn, uvw, g1, g2, freq, masked_image, major_axis, minor_axis,
                                                      pos_angle)  # [] or [2, 2]

            delta = compute_visibilities_gaussian_single_source(
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
        xs = (self.ra, self.dec, self.major_axis, self.minor_axis, self.pos_angle, self.A)

        # sum over sources
        vis_accumulation, _ = lax.scan(body_fn, init_accumulate, xs)
        return vis_accumulation  # [T, B, C[, 2, 2]]

    def _single_compute_visibilty(self, lmn, uvw, g1, g2, freq, image, major_axis, minor_axis, pos_angle):
        """
        Compute the visibility from a single direction for a single baseline.

        Args:
            lmn: [3]
            uvw: [3]
            g1: [2, 2]
            g2: [2, 2]
            freq: []
            image: [] or [2, 2]
            major_axis: []
            minor_axis: []
            pos_angle: []

        Returns:
            [2, 2] visibility in given direction for given baseline.
        """
        wavelength = quantity_to_jnp(constants.c) / freq

        if self.convention == 'engineering':
            uvw = jnp.negative(uvw)

        uvw /= wavelength

        u, v, w = uvw  # scalar

        l, m, n = lmn  # scalar

        if self.is_full_stokes():
            if g1 is not None and g1 is not None:
                image = kron_product(g1, image, g2.conj().T)

            vis = jax.vmap(
                lambda A: self._single_predict(
                    u, v, w,
                    A=A,
                    l0=l,
                    m0=m,
                    n0=n,
                    major=major_axis,
                    minor=minor_axis,
                    theta=pos_angle
                )
            )(flatten_coherencies(image))  # [4]
            vis = unflatten_coherencies(vis)  # [2,2]
        else:
            if g1 is not None and g2 is not None:
                image = g1 * image * g2.conj()
            vis = self._single_predict(
                u, v, w,
                A=image,
                l0=l,
                m0=m,
                n0=n,
                major=major_axis,
                minor=minor_axis,
                theta=pos_angle
            )  # []
        return vis

    def _single_predict(self, u, v, w,
                        A,
                        l0, m0, n0, major, minor, theta):

        def F_gaussian(u, v):
            gaussian = Gaussian(
                x0=jnp.asarray([l0, m0]),
                major_fwhm=major,
                minor_fwhm=minor,
                pos_angle=theta,
                total_flux=A
            )
            return mp_policy.cast_to_vis(gaussian.fourier(jnp.asarray([u, v])))

        def wkernel(l, m):
            n = jnp.sqrt(1. - l ** 2 - m ** 2)
            phase = -2j * jnp.pi * w * (n - 1.)
            return mp_policy.cast_to_vis(jnp.exp(phase) / n)

        if self.order_approx == 0:
            vis = F_gaussian(u, v) * wkernel(l0, m0)
        elif self.order_approx == 1:

            # Let I(l,m) * W(l,m) ~= I(l,m) * (W(l0, m0) + W_l * (l - l0) + W_m * (m - m0))
            # Where W_l = d/dl W(l0,m0), W_m = d/dm W(l0,m0)
            # F[I(l,m) * W(l,m)] ~= F[I(l,m) * W(l0,m0) + I(l,m) * W_l * (l - l0) + I(l,m) * W_m * (m - m0)]
            #  = (W0 - l0 * W_l - m0 * W_m) * F[I(l,m)] + (d/du * F[I(l,m)] * (W_l) + d/dv * F[I(l,m)] * (W_m)) / (-2 pi i)

            # maybe divide by 2pi
            wkernel_grad = jax.value_and_grad(wkernel, (0, 1), holomorphic=True)

            W0, (Wl, Wm) = wkernel_grad(jnp.asarray(l0, mp_policy.vis_dtype), jnp.asarray(m0, mp_policy.vis_dtype))

            F_jvp = JVPLinearOp(F_gaussian, promote_dtypes=True)
            vec = (
                jnp.asarray(Wl, mp_policy.vis_dtype), jnp.asarray(Wm, mp_policy.vis_dtype)
            )
            # promote_dtypes=True so we don't need to cast the primals here. Otherwise:
            # primals = (u.astype(vec[0].dtypegrad), v.astype(vec[1].dtype))
            primals = (u, v)
            F_jvp = F_jvp(*primals)

            vis = F_gaussian(u, v) * (W0 - l0 * Wl - m0 * Wm) + F_jvp.matvec(*vec) / (-2j * jnp.pi)
        else:
            raise ValueError("order_approx must be 0 or 1")
        return mp_policy.cast_to_vis(vis)

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

        l, m, n = perley_lmn_from_icrs(self.ra, self.dec, phase_center.ra.rad, phase_center.dec.rad)

        lvec = np.linspace(np.min(l), np.max(l), 256)
        mvec = np.linspace(np.min(m), np.max(m), 256)
        L, M = np.meshgrid(lvec, mvec, indexing='ij')

        lm = np.stack([L.flatten(), M.flatten()], axis=-1)

        # Evaluate over LM
        flux_model = np.zeros((lvec.size, mvec.size))

        dl = lvec[1] - lvec[0]
        dm = mvec[1] - mvec[0]

        @jax.jit
        def compute_gaussian_flux(A, l0, m0, major, minor, theta):
            gaussian = Gaussian(
                x0=jnp.stack([l0, m0]),
                major_fwhm=major,
                minor_fwhm=minor,
                pos_angle=theta,
                total_flux=A
            )
            return jax.vmap(gaussian.compute_flux_density)(lm).reshape(L.shape) * dl * dm

        for i, (li, mi, majori, minori, pos_anglei) in enumerate(
                zip(l, m, self.major_axis, self.minor_axis, self.pos_angle)):
            if self.is_full_stokes():
                flux_model += compute_gaussian_flux(
                    self.A[i, 0, 0, 0],
                    li,
                    mi,
                    majori,
                    minori,
                    pos_anglei
                )
            else:
                flux_model += compute_gaussian_flux(
                    self.A[i, 0],
                    li,
                    mi,
                    majori,
                    minori,
                    pos_anglei
                )

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

        im = axs.imshow(
            flux_model.T,
            origin='lower',
            extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]),
            interpolation='nearest'
        )
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
    def flatten(cls, this: "BaseGaussianSourceModel") -> Tuple[List[Any], Tuple[Any, ...]]:
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
                this.ra,
                this.dec,
                this.A,
                this.major_axis,
                this.minor_axis,
                this.pos_angle
            ],
            (
                this.convention,
                this.order_approx
            )
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "BaseGaussianSourceModel":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        (model_freqs, ra, dec, A, major_axis, minor_axis, pos_angle) = children
        (convention, order_approx) = aux_data
        return BaseGaussianSourceModel(
            model_freqs=model_freqs,
            ra=ra,
            dec=dec,
            A=A,
            major_axis=major_axis,
            minor_axis=minor_axis,
            pos_angle=pos_angle,
            convention=convention,
            order_approx=order_approx,
            skip_post_init=True
        )


BaseGaussianSourceModel.register_pytree()


def build_gaussian_source_model(
        model_freqs: au.Quantity,  # [num_model_freq] Frequencies
        ra: au.Quantity,  # [num_sources] l coordinate of the source
        dec: au.Quantity,  # [num_sources] m coordinate of the source
        A: au.Quantity,  # [num_sources, num_freqs[,2,2]] Flux amplitude of the source
        major_axis: au.Quantity,  # [num_sources] Major axis of the source in proj.radians
        minor_axis: au.Quantity,  # [num_sources] Minor axis of the source in proj.radians
        pos_angle: au.Quantity,  # [num_sources] Position angle of the source in proj.radians
        order_approx: int = 0
) -> BaseGaussianSourceModel:
    """
    Build a gaussian source model.

    Args:
        model_freqs: [num_model_freq] Frequencies
        ra: [num_sources] l coordinate of the source
        dec: [num_sources] m coordinate of the source
        A: [num_sources, num_model_freqs,[,2,2]] Flux amplitude of the source
        major_axis: [num_sources] Major axis of the source in proj.radians
        minor_axis: [num_sources] Minor axis of the source in proj.radians
        pos_angle: [num_sources] Position angle of the source in proj.radians

    Returns:
        the GaussianSourceModel
    """
    A = quantity_to_jnp(A, 'Jy')
    model_freqs = quantity_to_jnp(model_freqs, 'Hz')
    ra = quantity_to_jnp(ra, 'rad')
    dec = quantity_to_jnp(dec, 'rad')
    major_axis = quantity_to_jnp(major_axis, 'rad')
    minor_axis = quantity_to_jnp(minor_axis, 'rad')
    pos_angle = quantity_to_jnp(pos_angle, 'rad')
    return BaseGaussianSourceModel(
        model_freqs=model_freqs,
        ra=ra,
        dec=dec,
        A=A,
        major_axis=major_axis,
        minor_axis=minor_axis,
        pos_angle=pos_angle,
        order_approx=order_approx
    )


def build_gaussian_source_model_from_wsclean_components(
        wsclean_clean_component_file: str,
        model_freqs: au.Quantity,
        full_stokes: bool = True
) -> BaseGaussianSourceModel:
    """
    Create a GaussianSourceModel from a wsclean model file.

    Args:
        wsclean_clean_component_file: the wsclean model file
        model_freqs: the frequencies of the model
        full_stokes: whether the model is full stokes

    Returns:
        the GaussianSourceModel
    """
    # Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency='125584411.621094', MajorAxis, MinorAxis, Orientation
    # Example: s0c0,POINT,08:28:05.152,39.35.08.511,0.000748810650400475,[-0.00695379313004673,-0.0849693907803257],false,125584411.621094,,,
    # RA and dec are the central coordinates of the component, in notation of "hh:mm:ss.sss" and "dd.mm.ss.sss".
    # The MajorAxis, MinorAxis and Orientation columns define the shape of the Gaussian.
    # The axes are given in units of arcseconds, and orientation is in degrees.

    source_directions = []
    spectrum = []
    major_axis = []
    minor_axis = []
    pos_angle = []
    with open(wsclean_clean_component_file, 'r') as fp:
        for line in fp:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('#'):
                continue
            if line.startswith('Format'):
                continue
            parsed_results = parse_and_process_wsclean_source_line(line, model_freqs)
            if parsed_results is None:
                continue
            if parsed_results.type_ != 'GAUSSIAN':
                continue
            if parsed_results.major is None or parsed_results.minor is None or parsed_results.theta is None:
                raise ValueError("Major, minor, and theta must be provided for Gaussian sources")
            source_directions.append(parsed_results.direction)
            spectrum.append(parsed_results.spectrum)
            major_axis.append(parsed_results.major)
            minor_axis.append(parsed_results.minor)
            pos_angle.append(parsed_results.theta)

    source_directions = ac.concatenate(source_directions).transform_to(ac.ICRS)

    ra = source_directions.ra
    dec = source_directions.dec
    A = np.stack(spectrum, axis=0)  # [num_sources, num_freqs]

    major_axis = au.Quantity(major_axis)
    minor_axis = au.Quantity(minor_axis)
    pos_angle = au.Quantity(pos_angle)

    if full_stokes:
        A = np.asarray(
            broadcast_translate_corrs(
                quantity_to_jnp(A[..., None], 'Jy'),
                ('I',),
                (('XX', 'XY'), ('YX', 'YY'))
            )
        ) * au.Jy  # [num_sources, num_freqs, 2, 2]

    return build_gaussian_source_model(
        model_freqs=model_freqs,
        ra=ra,
        dec=dec,
        A=A,
        major_axis=major_axis,
        minor_axis=minor_axis,
        pos_angle=pos_angle
    )
