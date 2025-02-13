import dataclasses
import pickle
import warnings
from functools import partial
from typing import Tuple, List, Any

import jax
import numpy as np
import pylab as plt
from astropy import constants, units as au, coordinates as ac
from jax import numpy as jnp, lax

from dsa2000_cal.common.corr_utils import broadcast_translate_corrs
from dsa2000_cal.common.array_types import ComplexArray, FloatArray
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


@dataclasses.dataclass(eq=False)
class BasePointSourceModel(AbstractSourceModel):
    """
    Predict vis for point source.
    """
    model_freqs: FloatArray  # [num_model_freqs] Frequencies
    ra: FloatArray  # [num_sources] ra coordinate of the source
    dec: FloatArray  # [num_sources] dec coordinate of the source
    A: FloatArray  # [num_sources, num_model_freqs,[,2,2]] Flux amplitude of the source

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
        if np.shape(self.A)[:2] != (len(self.ra), len(self.model_freqs)):
            raise ValueError(
                f"A must have shape [{len(self.ra)}, {len(self.model_freqs)}[,2,2]], got {np.shape(self.A)}")

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
            (ra, dec, image) = x

            def compute_lmn_geodesic(time):
                lmn, elevation = geodesic_model.compute_far_field_lmn(ra, dec, time, return_elevation=True)  # [3], []
                lmn_geodesic = geodesic_model.compute_far_field_geodesic(
                    times=time[None],
                    lmn_sources=lmn[None, :]
                )  # [t=1, A, s=1, 3]
                return lmn, lmn_geodesic[0], elevation  # [3], [A, s=1, 3], []

            lmn, lmn_geodesic, elevation = jax.vmap(compute_lmn_geodesic)(
                visibility_coords.times)  # [T,3], [T, A, s=1, 3], [T]

            if gain_model is not None:
                # Compute the gains
                gains = gain_model.compute_gain(
                    freqs=visibility_coords.freqs,
                    times=visibility_coords.times,
                    lmn_geodesic=lmn_geodesic
                )  # [T, A, C, 1,[, 2, 2]]
                g1 = gains[:, :, :, 0][:, visibility_coords.antenna1, ...]  # [T, B, C, [, 2, 2]]
                g2 = gains[:, :, :, 0][:, visibility_coords.antenna2, ...]  # [T, B, C, [, 2, 2]]
                gains_mapping = '[T,B,C,...]'
            else:
                gains_mapping = '[]'
                g1 = g2 = None

            # vmap over time and freqs
            if self.is_full_stokes():
                out_mapping = '[T,B,C,~P,~Q]'
            else:
                out_mapping = '[T,B,C]'

            @partial(
                multi_vmap,
                in_mapping=f'[C],[T,B,3],{gains_mapping},{gains_mapping},[T,3],[T]',
                out_mapping=out_mapping,
                scan_dims={'C', 'T'},
                verbose=True
            )
            def compute_visibilities_point_single_source(freq, uvw, g1, g2, lmn, elevation):
                interp = InterpolatedArray(x=self.model_freqs, values=image, axis=0, regular_grid=True,
                                           check_spacing=False)
                masked_image = jnp.where(elevation <= 0, 0, interp(freq))  # [2, 2]

                vis = self._single_compute_visibilty(lmn, uvw, g1, g2, freq, masked_image)  # [] or [2, 2]

                return vis

            # g1, g2 = _print((g1, g2))

            delta = compute_visibilities_point_single_source(
                visibility_coords.freqs,
                visibility_coords.uvw,
                g1,
                g2,
                lmn,
                elevation
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
        xs = (self.ra, self.dec, self.A)

        # sum over sources
        vis_accumulation, _ = lax.scan(body_fn, init_accumulate, xs)
        return vis_accumulation  # [T, B, C[, 2, 2]]

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

        # Evaluate over LM
        flux_model = np.zeros((lvec.size, mvec.size))

        dl = lvec[1] - lvec[0]
        dm = mvec[1] - mvec[0]

        for i, (li, mi) in enumerate(zip(l, m)):
            l_idx = int((li - lvec[0]) / dl)
            m_idx = int((mi - mvec[0]) / dm)
            if l_idx >= 0 and l_idx < lvec.size and m_idx >= 0 and m_idx < mvec.size:
                if self.is_full_stokes():
                    flux_model[l_idx, m_idx] += self.A[i, 0, 0, 0]
                else:
                    flux_model[l_idx, m_idx] += self.A[i, 0]

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
    def flatten(cls, this: "BasePointSourceModel") -> Tuple[List[Any], Tuple[Any, ...]]:
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
                this.A
            ],
            (
                this.convention,
            )
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "BasePointSourceModel":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
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


BasePointSourceModel.register_pytree()


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
        A: [num_sources, num_model_freqs,[,2,2]] Flux amplitude of the source

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
    A = np.stack(spectrum, axis=0)  # [num_sources, num_freqs]

    if full_stokes:
        A = np.asarray(
            broadcast_translate_corrs(
                quantity_to_jnp(A[..., None], 'Jy'),
                ('I',),
                (('XX', 'XY'), ('YX', 'YY'))
            )
        ) * au.Jy  # [num_sources, num_freqs, 2, 2]

    return build_point_source_model(
        model_freqs=model_freqs,
        ra=ra,
        dec=dec,
        A=A
    )


def build_calibration_point_source_models_from_wsclean(
        wsclean_component_file: str,
        model_freqs: au.Quantity,
        pointing: ac.ICRS,
        fov_fwhm: au.Quantity,
        full_stokes: bool = True,
):
    """
    Build a calibration source model from wsclean components.

    Args:
        wsclean_component_file: the wsclean component file
        model_freqs: the model frequencies
        fov_fwhm: the fov fwhm
        full_stokes: whether the model should be full stokes

    Returns:
        the calibration source model
    """
    sky_model = build_point_source_model_from_wsclean_components(
        wsclean_clean_component_file=wsclean_component_file,
        model_freqs=model_freqs,
        full_stokes=full_stokes
    )
    #     model_freqs: FloatArray  # [num_model_freqs] Frequencies
    #     ra: FloatArray  # [num_sources] ra coordinate of the source
    #     dec: FloatArray  # [num_sources] dec coordinate of the source
    #     A: FloatArray  # [num_sources, num_model_freqs,[,2,2]] Flux amplitude of the source
    num_facets = np.shape(sky_model.A)[0]
    # Turn each facet into a 1 facet sky model
    sky_models = []
    for facet_idx in range(num_facets):
        A = sky_model.A[facet_idx:facet_idx + 1]  # [facet=1,num_model_freqs, [2,2]]
        ra = sky_model.ra[facet_idx:facet_idx + 1]  # [facet=1]
        dec = sky_model.dec[facet_idx:facet_idx + 1]  # [facet=1]
        # Get haversine distance from pointing,only keep sources within fov_fwhm
        haversine_distance = ac.SkyCoord(ra=ra * au.rad, dec=dec * au.rad).separation(pointing).to('rad')
        if haversine_distance > 0.5 * fov_fwhm:
            continue
        sky_models.append(
            BasePointSourceModel(
                model_freqs=sky_model.model_freqs,
                A=A,
                ra=ra,
                dec=dec,
                convention=sky_model.convention
            )
        )
    return sky_models


