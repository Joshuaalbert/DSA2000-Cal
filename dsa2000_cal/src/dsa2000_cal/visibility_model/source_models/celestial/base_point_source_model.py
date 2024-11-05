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
from dsa2000_cal.common.corr_translation import stokes_I_to_linear
from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.jax_utils import multi_vmap
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


class PointModelData(NamedTuple):
    """
    Data for predict.
    """
    freqs: FloatArray  # [chan]
    times: FloatArray  # [num_time]
    image: FloatArray  # [num_time, chan, source,[2,2]] in [[xx, xy], [yx, yy]] format or stokes I
    lmn: FloatArray  # [source, 3]


@dataclasses.dataclass(eq=False)
class BasePointSourceModel(AbstractSourceModel[PointModelData]):
    """
    Predict vis for point source.
    """
    model_freqs: FloatArray  # [num_model_freqs] Frequencies
    ra: FloatArray  # [num_sources] ra coordinate of the source
    dec: FloatArray  # [num_sources] dec coordinate of the source
    A: FloatArray  # [num_model_freqs,num_sources,[,2,2]] Flux amplitude of the source

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
            raise ValueError("ra and dec must have the same shape")
        if np.shape(self.A)[:2] != (len(self.model_freqs), len(self.ra)):
            raise ValueError(
                f"A must have shape [{np.shape(self.model_freqs)[0]}, {np.shape(self.ra)[0]}[,2,2]], got {np.shape(self.A)}")

    def is_full_stokes(self) -> bool:
        return len(np.shape(self.A)) == 4 and np.shape(self.A)[-2:] == (2, 2)

    def get_model_data(self, freqs: FloatArray, times: FloatArray, geodesic_model: BaseGeodesicModel) -> PointModelData:
        mean_time = jnp.mean(times)
        lmn, elevation = geodesic_model.compute_far_field_lmn(self.ra, self.dec, mean_time, return_elevation=True)
        interp = InterpolatedArray(x=self.model_freqs, values=self.A, axis=0)
        image = interp(freqs)  # [num_freqs, num_sources[, 2, 2]]
        elevation_mask = elevation <= 0  # [num_sources]
        if self.is_full_stokes():
            image = jnp.where(elevation_mask[None, :, None, None], jnp.zeros_like(image),
                              image)  # [num_freqs, num_sources, 2, 2]
        else:
            image = jnp.where(elevation_mask[None, :], jnp.zeros_like(image), image)  # [num_freqs, num_sources]
        # Add a time dimension
        image = image[None]  # [1, num_freqs, num_sources[, 2, 2]]

        return PointModelData(
            freqs=mp_policy.cast_to_freq(freqs),
            times=mp_policy.cast_to_time(mean_time[None]),
            image=mp_policy.cast_to_image(image),
            lmn=mp_policy.cast_to_angle(lmn)
        )

    # def get_flux_weighted_center_lmn(self, model_data: PointModelData):
    #     # Get flux weighted lmn
    #     A_sum = jnp.sum(model_data.image)  # []
    #     l_avg = jnp.sum(model_data.lmn[None, None, :, 0, None, None] * model_data.image) / jnp.sum(A_sum)  # [num_source]
    #     m_avg = jnp.sum(model_data.lmn[None, None, :, 1, None, None] * model_data.image,
    #                     axis=(0, 1, 3, 4)) / jnp.sum(A_sum)  # [num_source]
    #     n_avg = jnp.sum(model_data.lmn[None, None, :, 2, None, None] * model_data.image,
    #                     axis=(0, 1, 3, 4)) / jnp.sum(A_sum)  # [num_source]
    #     lmn_avg = jnp.stack([l_avg, m_avg, n_avg], axis=-1)  # [num_source, 3]
    #     lmn_avg = lmn_avg / jnp.linalg.norm(lmn_avg, axis=-1, keepdims=True)  # [num_source, 3]

    def predict(
            self,
            model_data: PointModelData,
            visibility_coords: VisibilityCoords,
            gain_model: GainModel | None,
            near_field_delay_engine: BaseNearFieldDelayEngine,
            far_field_delay_engine: BaseFarFieldDelayEngine,
            geodesic_model: BaseGeodesicModel
    ) -> ComplexArray:
        # freqs: jax.Array  # [chan]
        # image: jax.Array  # [num_time/1, chan, source[,2,2]] in [[xx, xy], [yx, yy]] format or stokes I
        # lmn: jax.Array  # [source, 3]
        if gain_model is not None:
            # normalise again
            lmn_geodesic = geodesic_model.compute_far_field_geodesic(
                times=model_data.times,
                lmn_sources=model_data.lmn
            )  # [num_time, num_ant, num_sources, 3]
            # Compute the gains
            gains = gain_model.compute_gain(
                freqs=model_data.freqs,
                times=model_data.times,
                lmn_geodesic=lmn_geodesic,
            )  # [num_time, num_ant, num_freq, num_sources,[, 2, 2]]
        else:
            gains = None

        _t = visibility_coords.time_idx
        _a1 = visibility_coords.antenna_1
        _a2 = visibility_coords.antenna_2

        is_gains = gains is not None
        tile_times = np.shape(model_data.image)[0] == 1
        if tile_times:
            image = model_data.image[0]
            if self.is_full_stokes():
                image_mapping = "[c,s,2,2]"
            else:
                image_mapping = "[c,s]"
        else:
            image = model_data.image[_t]
            if self.is_full_stokes():
                image_mapping = "[r,c,s,2,2]"
            else:
                image_mapping = "[r,c,s]"

        if is_gains:
            if self.is_full_stokes():
                g1 = gains[_t, _a1, :, :, :, :]
                g2 = gains[_t, _a1, :, :, :, :]
                g_mapping = "[r,c,s,2,2]"
            else:
                g1 = gains[_t, _a1, :, :]
                g2 = gains[_t, _a1, :, :]
                g_mapping = "[r,c,s]"
        else:
            g1 = None
            g2 = None
            g_mapping = "[]"

        if self.is_full_stokes():
            out_mapping = "[r,c,~p,~q]"
        else:
            out_mapping = "[r,c]"

        # Data will be sharded over frequency so don't reduce over these dimensions, or else communication happens.
        # We want the outer broadcast to be over chan, so we'll do this order.

        # lmn: [source, 3]
        # uvw: [rows, 3]
        # g1, g2: g_mapping
        # freq: [chan]
        # image: [source, chan, 2, 2]
        @partial(
            multi_vmap,
            in_mapping=f"[s,3],[r,3],{g_mapping},{g_mapping},[c],{image_mapping}",
            out_mapping=out_mapping,
            scan_dims={'s'},
            verbose=True
        )
        def compute_visibilities_point_sources(lmn, uvw, g1, g2, freq, image):
            """
            Compute visibilities for a single row, channel, accumulating over sources.

            Args:
                lmn: [source, 3]
                uvw: [3]
                g1: [source, 2, 2] or None
                g2: [source, 2, 2] or None
                freq: []
                image: [source, 2, 2]

            Returns:
                vis_accumulation: [2, 2] visibility for given baseline, accumulated over all provided directions.
            """

            def body_fn(accumulate, x):
                (lmn, g1, g2, image) = x
                delta = self._single_compute_visibilty(lmn, uvw, g1, g2, freq, image)  # [] or [2, 2]
                accumulate += mp_policy.cast_to_vis(delta)
                return accumulate, ()

            xs = (lmn, g1, g2, image)
            init_accumulate = jnp.zeros((2, 2) if self.is_full_stokes() else (), dtype=mp_policy.vis_dtype)
            vis_accumulation, _ = lax.scan(body_fn, init_accumulate, xs, unroll=4)
            return vis_accumulation  # [] or [2, 2]

        visibilities = compute_visibilities_point_sources(
            model_data.lmn,
            visibility_coords.uvw,
            g1,
            g2,
            model_data.freqs,
            image
        )  # [row, chan[, 2, 2]]
        return visibilities

    def _single_compute_visibilty(self, lmn, uvw, g1, g2, freq, image):
        """
        Compute the visibility from a single direction for a single baseline.

        Args:
            lmn: [3]
            uvw: [3]
            g1: [2, 2]
            g2: [2, 2]
            freq: []
            image: [] or [2, 2]

        Returns:
            [2, 2] visibility in given direction for given baseline.
        """
        wavelength = quantity_to_jnp(constants.c) / freq

        if self.convention == 'engineering':
            uvw = jnp.negative(uvw)

        uvw /= wavelength

        u, v, w = uvw  # scalar

        l, m, n = lmn  # scalar

        # -2*pi*freq/c*(l*u + m*v + (n-1)*w)
        delay = l * u + m * v + (n - 1.) * w  # scalar

        phi = (-2 * np.pi) * delay  # scalar
        fringe = jax.lax.complex(jnp.cos(phi), jnp.sin(phi)) / n  # scalar

        if self.is_full_stokes():
            if g1 is None or g1 is None:
                return mp_policy.cast_to_vis(fringe * image)  # [2, 2]
            return mp_policy.cast_to_vis(fringe * kron_product(g1, image, g2.conj().T))
        else:
            if g1 is None or g1 is None:
                return mp_policy.cast_to_vis(fringe * image)
            return mp_policy.cast_to_vis(fringe * (g1 * g2.conj() * image))

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

        lvec = np.linspace(np.min(l) - 0.01, np.max(l) + 0.01, 256)
        mvec = np.linspace(np.min(m) - 0.01, np.max(m) + 0.01, 256)

        # Evaluate over LM
        flux_model = np.zeros((lvec.size, mvec.size))

        dl = lvec[1] - lvec[0]
        dm = mvec[1] - mvec[0]

        for i, (li, mi) in enumerate(zip(l, m)):
            l_idx = int((li - lvec[0]) / dl)
            m_idx = int((mi - mvec[0]) / dm)
            if l_idx >= 0 and l_idx < lvec.size and m_idx >= 0 and m_idx < mvec.size:
                if self.is_full_stokes():
                    flux_model[l_idx, m_idx] += self.A[0, i, 0, 0]
                else:
                    flux_model[l_idx, m_idx] += self.A[0, i]

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

        im = axs.imshow(flux_model.T, origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]))
        # colorbar
        plt.colorbar(im, ax=axs)
        axs.set_xlabel('l [proj.rad]')
        axs.set_ylabel('m [proj.rad]')
        if save_file is not None:
            plt.savefig(save_file)
        plt.show()


def base_point_source_model_flatten(model: BasePointSourceModel):
    return (
        [
            model.model_freqs,
            model.ra,
            model.dec,
            model.A
        ],
        (
            model.convention,
        )
    )

# register pytree

def base_point_source_model_unflatten(aux_data, children):
    (model_freqs, ra, dec, A) = children
    (convention,) = aux_data
    return BasePointSourceModel(
        model_freqs=model_freqs,
        ra=ra,
        dec=dec,
        A=A,
        convention=convention,
        skip_post_init=True
    )

jax.tree_util.register_pytree_node(
    BasePointSourceModel,
    base_point_source_model_flatten,
    base_point_source_model_unflatten
)


def build_point_source_model(
        model_freqs: au.Quantity,  # [num_model_freq] Frequencies
        ra: au.Quantity,  # [num_sources] l coordinate of the source
        dec: au.Quantity,  # [num_sources] m coordinate of the source
        A: au.Quantity,  # [num_sources, num_freqs[,2,2]] Flux amplitude of the source
) -> BasePointSourceModel:
    """
    Build a point source model.

    Args:
        model_freqs: [num_model_freq] Frequencies
        ra: [num_sources] l coordinate of the source
        dec: [num_sources] m coordinate of the source
        A: [num_model_freqs, num_sources,[,2,2]] Flux amplitude of the source

    Returns:
        the PointSourceModel
    """
    A = quantity_to_jnp(A, 'Jy')
    model_freqs = quantity_to_jnp(model_freqs, 'Hz')
    ra = quantity_to_jnp(ra, 'rad')
    dec = quantity_to_jnp(dec, 'rad')
    return BasePointSourceModel(
        model_freqs=model_freqs,
        ra=ra,
        dec=dec,
        A=A
    )




def build_point_source_model_from_wsclean_components(
        wsclean_clean_component_file: str,
        model_freqs: au.Quantity,
        full_stokes: bool = True
) -> BasePointSourceModel:
    """
    Create a GaussianSourceModel from a wsclean model file.

    Args:
        wsclean_clean_component_file: the wsclean model file
        model_freqs: the frequencies of the model
        full_stokes: whether the model is full stokes

    Returns:
        the PointSourceModel
    """
    # Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency='125584411.621094', MajorAxis, MinorAxis, Orientation
    # Example: s0c0,POINT,08:28:05.152,39.35.08.511,0.000748810650400475,[-0.00695379313004673,-0.0849693907803257],false,125584411.621094,,,
    # RA and dec are the central coordinates of the component, in notation of "hh:mm:ss.sss" and "dd.mm.ss.sss".
    # The MajorAxis, MinorAxis and Orientation columns define the shape of the Gaussian.
    # The axes are given in units of arcseconds, and orientation is in degrees.

    source_directions = []
    spectrum = []
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
            if parsed_results.type_ != 'POINT':
                continue
            source_directions.append(parsed_results.direction)
            spectrum.append(parsed_results.spectrum)

    source_directions = ac.concatenate(source_directions).transform_to(ac.ICRS)

    ra = source_directions.ra
    dec = source_directions.dec
    A = np.stack(spectrum, axis=1) * au.Jy  # [num_freqs, num_sources]

    if full_stokes:
        A = np.asarray(stokes_I_image_to_linear(quantity_to_jnp(A, 'Jy'), flat_output=False)) * au.Jy

    return build_point_source_model(
        model_freqs=model_freqs,
        ra=ra,
        dec=dec,
        A=A
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
