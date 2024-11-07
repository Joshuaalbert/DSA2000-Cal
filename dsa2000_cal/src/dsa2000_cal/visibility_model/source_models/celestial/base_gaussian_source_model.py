import dataclasses
from functools import partial
from typing import NamedTuple

import astropy.coordinates as ac
import astropy.units as au
import jax
import numpy as np
import pylab as plt
from astropy import constants
from jax import numpy as jnp, lax

from dsa2000_cal.common.array_types import ComplexArray, FloatArray
from dsa2000_cal.common.corr_translation import stokes_I_to_linear, flatten_coherencies, unflatten_coherencies
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


class GaussianModelData(NamedTuple):
    """
    Data for predict.
    """
    image: FloatArray  # [source,[2,2]] in [[xx, xy], [yx, yy]] format or stokes I
    lmn: FloatArray  # [source, 3]
    major_axis: FloatArray  # [source]
    minor_axis: FloatArray  # [source]
    pos_angle: FloatArray  # [source]


@dataclasses.dataclass(eq=False)
class BaseGaussianSourceModel(AbstractSourceModel[GaussianModelData]):
    """
    Predict vis for point source.
    """
    model_freqs: FloatArray  # [num_model_freqs] Frequencies
    ra: FloatArray  # [num_sources] ra coordinate of the source
    dec: FloatArray  # [num_sources] dec coordinate of the source
    A: FloatArray  # [num_model_freqs,num_sources,[,2,2]] Flux amplitude of the source
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
        if np.shape(self.A)[:2] != (len(self.model_freqs), len(self.ra)):
            raise ValueError(
                f"A must have shape [{np.shape(self.model_freqs)[0]}, {np.shape(self.ra)[0]}[,2,2]], got {np.shape(self.A)}")

    def is_full_stokes(self) -> bool:
        return len(np.shape(self.A)) == 4 and np.shape(self.A)[-2:] == (2, 2)

    def get_model_slice(self, freq: FloatArray, time: FloatArray,
                        geodesic_model: BaseGeodesicModel) -> GaussianModelData:
        lmn, elevation = geodesic_model.compute_far_field_lmn(self.ra, self.dec, time, return_elevation=True)
        interp = InterpolatedArray(x=self.model_freqs, values=self.A, axis=0)
        image = interp(freq)  # [num_sources[, 2, 2]]
        elevation_mask = elevation <= 0  # [num_sources]
        if self.is_full_stokes():
            image = jnp.where(elevation_mask[:, None, None], jnp.zeros_like(image), image)  # [num_sources, 2, 2]
        else:
            image = jnp.where(elevation_mask, jnp.zeros_like(image), image)  # [num_sources]

        return GaussianModelData(
            image=mp_policy.cast_to_image(image),
            lmn=mp_policy.cast_to_angle(lmn),
            major_axis=mp_policy.cast_to_angle(self.major_axis),
            minor_axis=mp_policy.cast_to_angle(self.minor_axis),
            pos_angle=mp_policy.cast_to_angle(self.pos_angle)
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
        def compute_baseline_visibilities_gaussian(uvw, freq, time):
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
            )  # [num_sources, 2, 2]

            if gain_model is not None:
                lmn_phased = geodesic_model.compute_far_field_lmn(self.ra, self.dec)  # [num_sources, 3]
                lmn_geodesic = geodesic_model.compute_far_field_geodesic(
                    times=time[None],
                    lmn_sources=lmn_phased
                )  # [1, num_ant, num_sources, 3]
                # Compute the gains
                gains = gain_model.compute_gain(
                    freqs=freq[None],
                    times=time[None],
                    lmn_geodesic=lmn_geodesic,
                )  # [1, num_ant, 1, num_sources,[, 2, 2]]
                g1 = gains[0, visibility_coords.antenna_1, 0, :, ...]  # [B, num_sources[, 2, 2]]
                g2 = gains[0, visibility_coords.antenna_2, 0, :, ...]  # [B, num_sources[, 2, 2]]
            else:
                g1 = g2 = None

            if self.is_full_stokes():
                image_mapping = "[S,2,2]"
                gain_mapping = "[B,S,2,2]"
                out_mapping = "[B,~P,~Q]"
            else:
                image_mapping = "[S]"
                gain_mapping = "[B,S]"
                out_mapping = "[B]"

            @partial(
                multi_vmap,
                in_mapping=f"[S,3],[B,3],{gain_mapping},{gain_mapping},{image_mapping},[S],[S],[S]",
                out_mapping=out_mapping,
                scan_dims={'S'},
                verbose=True
            )
            def compute_visibilities_gaussian_over_sources(lmn, uvw, g1, g2, image,
                                                           major_axis, minor_axis, pos_angle):
                """
                Compute visibilities for a single direction, accumulating over sources.

                Args:
                    lmn: [3]
                    uvw: [3]
                    g1: [S[, 2, 2]]
                    g2: [S[, 2, 2]]
                    image: [S[, 2, 2]]
                    major_axis: [S]
                    minor_axis: [S]
                    pos_angle: [S]

                Returns:
                    vis_accumulation: [B[, 2, 2]] visibility for given baseline, accumulated over all provided directions.
                """

                def body_fn(accumulate, x):
                    (lmn, g1, g2, image, major_axis, minor_axis, pos_angle) = x
                    delta = self._single_compute_visibilty(lmn, uvw, g1, g2, freq, image,
                                                           major_axis, minor_axis, pos_angle)  # [] or [2, 2]
                    accumulate += mp_policy.cast_to_vis(delta)
                    return accumulate, ()

                xs = (lmn, g1, g2, image, major_axis, minor_axis, pos_angle)
                init_accumulate = jnp.zeros((2, 2) if self.is_full_stokes() else (), dtype=mp_policy.vis_dtype)
                vis_accumulation, _ = lax.scan(body_fn, init_accumulate, xs, unroll=4)
                return vis_accumulation  # [] or [2, 2]

            return compute_visibilities_gaussian_over_sources(model_data.lmn, uvw, g1, g2, model_data.image,
                                                              model_data.major_axis, model_data.minor_axis,
                                                              model_data.pos_angle)

        visibilities = compute_baseline_visibilities_gaussian(
            visibility_coords.uvw,
            visibility_coords.freqs,
            visibility_coords.times
        )  # [num_times, num_baselines, num_freqs[,2, 2]]
        return visibilities

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

        l, m, n = perley_lmn_from_icrs(self.ra, self.dec, phase_tracking.ra.rad, phase_tracking.dec.rad)

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
                    self.A[0, i, 0, 0],
                    li,
                    mi,
                    majori,
                    minori,
                    pos_anglei
                )
            else:
                flux_model += compute_gaussian_flux(
                    self.A[0, i],
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


# register pytree

def base_gaussian_source_model_flatten(model: BaseGaussianSourceModel):
    return (
        [
            model.model_freqs,
            model.ra,
            model.dec,
            model.A,
            model.major_axis,
            model.minor_axis,
            model.pos_angle
        ],
        (
            model.convention,
            model.order_approx
        )
    )


def base_gaussian_source_model_unflatten(aux_data, children):
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


jax.tree_util.register_pytree_node(
    BaseGaussianSourceModel,
    base_gaussian_source_model_flatten,
    base_gaussian_source_model_unflatten
)


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
    Build a point source model.

    Args:
        model_freqs: [num_model_freq] Frequencies
        ra: [num_sources] l coordinate of the source
        dec: [num_sources] m coordinate of the source
        A: [num_model_freqs, num_sources,[,2,2]] Flux amplitude of the source
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
    A = np.stack(spectrum, axis=1) * au.Jy  # [num_freqs, num_sources]

    major_axis = au.Quantity(major_axis)
    minor_axis = au.Quantity(minor_axis)
    pos_angle = au.Quantity(pos_angle)

    if full_stokes:
        A = np.asarray(stokes_I_image_to_linear(quantity_to_jnp(A, 'Jy'), flat_output=False)) * au.Jy

    return build_gaussian_source_model(
        model_freqs=model_freqs,
        ra=ra,
        dec=dec,
        A=A,
        major_axis=major_axis,
        minor_axis=minor_axis,
        pos_angle=pos_angle
    )


@partial(jax.jit, static_argnames=['flat_output'])
def stokes_I_image_to_linear(image: FloatArray, flat_output: bool) -> FloatArray:
    """
    Convert a Stokes I image to linear.

    Args:
        image: [Nl, Nm]

    Returns:
        linear: [Nl, Nm, ...]
    """

    @partial(multi_vmap, in_mapping="[f,s]", out_mapping="[f,s,...]")
    def convert_stokes_I_to_linear(coh):
        return stokes_I_to_linear(coh, flat_output)

    return convert_stokes_I_to_linear(image)
