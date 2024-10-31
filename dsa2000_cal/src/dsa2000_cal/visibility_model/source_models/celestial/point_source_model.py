import dataclasses
from functools import partial
from typing import NamedTuple, Tuple

import astropy.coordinates as ac
import astropy.units as au
import jax
import numpy as np
import pylab as plt
from astropy import constants
from jax import numpy as jnp, lax

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.corr_translation import stokes_I_to_linear
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.common.wsclean_util import parse_and_process_wsclean_source_line
from dsa2000_cal.delay_models.far_field import VisibilityCoords


@partial(jax.jit, static_argnames=['flat_output'])
def stokes_I_image_to_linear(image: jax.Array, flat_output: bool) -> jax.Array:
    """
    Convert a Stokes I image to linear.

    Args:
        image: [Nl, Nm]

    Returns:
        linear: [Nl, Nm, ...]
    """

    @partial(multi_vmap, in_mapping="[s,f]", out_mapping="[s,f,...]")
    def f(coh):
        return stokes_I_to_linear(coh, flat_output)

    return f(image)


class PointSourceModelParams(SerialisableBaseModel):
    freqs: au.Quantity  # [num_freqs] Frequencies
    l0: au.Quantity  # [num_sources] l coordinate of the source
    m0: au.Quantity  # [num_sources] m coordinate of the source
    A: au.Quantity  # [num_sources, num_freqs[,2,2]] Flex amplitude of the source

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(PointSourceModelParams, self).__init__(**data)
        _check_point_source_model_params(self)


class PointModelData(NamedTuple):
    """
    Data for predict.
    """
    freqs: jax.Array  # [chan]
    image: jax.Array  # [source, chan[, 2, 2]] in [[xx, xy], [yx, yy]] format or stokes
    gains: jax.Array | None  # [[source,] time, ant, chan[, 2, 2]]
    lmn: jax.Array  # [source, 3]


def _check_point_source_model_params(params: PointSourceModelParams):
    if not params.freqs.unit.is_equivalent(au.Hz):
        raise ValueError(f"Expected freqs to be in Hz, got {params.freqs.unit}")
    if not params.l0.unit.is_equivalent(au.dimensionless_unscaled):
        raise ValueError(f"Expected l0 to be dimensionless, got {params.l0.unit}")
    if not params.m0.unit.is_equivalent(au.dimensionless_unscaled):
        raise ValueError(f"Expected m0 to be dimensionless, got {params.m0.unit}")
    if not params.A.unit.is_equivalent(au.Jy):
        raise ValueError(f"Expected A to be in Jy, got {params.A.unit}")

    # Ensure all are 1D vectors
    if params.freqs.isscalar:
        params.freqs = params.freqs.reshape((-1,))
    if params.l0.isscalar:
        params.l0 = params.l0.reshape((-1,))
    if params.m0.isscalar:
        params.m0 = params.m0.reshape((-1,))
    if params.A.isscalar:
        params.A = params.A.reshape((-1, 1))

    num_sources = params.l0.shape[0]
    num_freqs = params.freqs.shape[0]

    if not (params.A.shape == (num_sources, num_freqs) or params.A.shape == (num_sources, num_freqs, 2, 2)):
        raise ValueError(f"A must have shape ({num_sources},{num_freqs}[2,2]) got {params.A.shape}")

    if not all([x.shape == (num_sources,) for x in [params.l0, params.m0]]):
        raise ValueError("All inputs must have the same shape")


@dataclasses.dataclass(eq=False)
class PointSourceModel(AbstractSourceModel):
    """
    Predict vis for point source.
    """
    freqs: au.Quantity  # [num_freqs] Frequencies
    l0: au.Quantity  # [num_sources] l coordinate of the source
    m0: au.Quantity  # [num_sources] m coordinate of the source
    A: au.Quantity  # [num_sources, num_freqs[,2,2]] Flex amplitude of the source

    def __getitem__(self, item):
        return PointSourceModel(
            freqs=self.freqs,
            l0=self.l0[item],
            m0=self.m0[item],
            A=self.A[item]
        )

    def __add__(self, other: 'PointSourceModel') -> 'PointSourceModel':
        # ensure freqs same
        if not np.all(self.freqs == other.freqs):
            raise ValueError("Frequencies must match")
        # Ensure both same is_stokes
        if self.is_full_stokes() != other.is_full_stokes():
            raise ValueError("Both must be full stokes or not")
        # concat
        return PointSourceModel(
            freqs=self.freqs,
            l0=au.Quantity(np.concatenate([self.l0, other.l0])),
            m0=au.Quantity(np.concatenate([self.m0, other.m0])),
            A=au.Quantity(np.concatenate([self.A, other.A], axis=0))
        )

    def is_full_stokes(self) -> bool:
        return len(self.A.shape) == 4 and self.A.shape[-2:] == (2, 2)

    def get_lmn_sources(self) -> jax.Array:
        """
        Get the lmn coordinates of the sources.

        Returns:
            lmn: [num_sources, 3] l, m, n coordinates of the sources
        """
        n0 = np.sqrt(1. - self.l0 ** 2 - self.m0 ** 2)
        return jnp.stack(
            [
                quantity_to_jnp(self.l0),
                quantity_to_jnp(self.m0),
                quantity_to_jnp(n0)
            ],
            axis=-1
        )  # [num_sources, 3]

    def get_model_data(self, gains: jax.Array | None = None) -> PointModelData:
        """
        Get the model data for the Gaussian source model.

        Args:
            gains: [[source,] time, ant, chan[, 2, 2]] the gains to apply

        Returns:
            model_data: the model data
        """
        lmn = jnp.stack(
            [
                quantity_to_jnp(self.l0), quantity_to_jnp(self.m0), jnp.sqrt(1 - self.l0 ** 2 - self.m0 ** 2)
            ], axis=-1
        )
        return PointModelData(
            freqs=mp_policy.cast_to_freq(quantity_to_jnp(self.freqs)),
            image=mp_policy.cast_to_image(quantity_to_jnp(self.A, 'Jy')),
            lmn=mp_policy.cast_to_angle(lmn),
            gains=mp_policy.cast_to_gain(gains)
        )

    @staticmethod
    def from_point_source_params(params: PointSourceModelParams, **kwargs) -> 'PointSourceModel':
        return PointSourceModel(**params.dict(), **kwargs)

    def __post_init__(self):
        _check_point_source_model_params(self)

        self.num_sources = self.l0.shape[0]
        self.num_freqs = self.freqs.shape[0]

        self.n0 = np.sqrt(1 - self.l0 ** 2 - self.m0 ** 2)  # [num_sources]
        self.wavelengths = constants.c / self.freqs  # [num_freqs]

    def total_flux(self) -> au.Quantity:
        return au.Quantity(jnp.sum(self.A, axis=0))

    def flux_weighted_lmn(self) -> au.Quantity:
        A_avg = np.mean(self.A, axis=1)  # [num_sources]
        m_avg = np.sum(self.m0 * A_avg) / np.sum(A_avg)
        l_avg = np.sum(self.l0 * A_avg) / np.sum(A_avg)
        lmn = np.stack([l_avg, m_avg, np.sqrt(1 - l_avg ** 2 - m_avg ** 2)]) * au.dimensionless_unscaled
        return lmn

    @staticmethod
    def from_wsclean_model(wsclean_clean_component_file: str, phase_tracking: ac.ICRS,
                           freqs: au.Quantity, full_stokes: bool = True) -> 'PointSourceModel':
        """
        Create a GaussianSourceModel from a wsclean model file.

        Args:
            wsclean_clean_component_file: the wsclean model file
            phase_tracking: the phase tracking center
            freqs: the frequencies to use
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
                parsed_results = parse_and_process_wsclean_source_line(line, freqs)
                if parsed_results is None:
                    continue
                if parsed_results.type_ != 'POINT':
                    continue
                source_directions.append(parsed_results.direction)
                spectrum.append(parsed_results.spectrum)

        source_directions = ac.concatenate(source_directions).transform_to(ac.ICRS)
        lmn0 = icrs_to_lmn(source_directions, phase_tracking)
        l0 = lmn0[:, 0]
        m0 = lmn0[:, 1]
        n0 = lmn0[:, 2]

        A = jnp.stack(spectrum, axis=0) * au.Jy  # [num_sources, num_freqs]

        if full_stokes:
            A = np.asarray(stokes_I_image_to_linear(quantity_to_jnp(A, 'Jy'), flat_output=False)) * au.Jy

        return PointSourceModel(
            freqs=freqs,
            l0=l0,
            m0=m0,
            A=A,
        )

    def get_flux_model(self, lvec=None, mvec=None):

        # Use imshow to plot the sky model evaluated over a LM grid

        if lvec is None or mvec is None:
            # Use imshow to plot the sky model evaluated over a LM grid
            l_min = np.min(self.l0) - 0.01
            m_min = np.min(self.m0) - 0.01
            l_max = np.max(self.l0) + 0.01
            m_max = np.max(self.m0) + 0.01
            lvec = np.linspace(l_min.value, l_max.value, 256)
            mvec = np.linspace(m_min.value, m_max.value, 256)

        # Evaluate over LM
        flux_model = np.zeros((mvec.size, lvec.size)) * au.Jy

        dl = lvec[1] - lvec[0]
        dm = mvec[1] - mvec[0]

        for i in range(self.num_sources):
            l_idx = int((self.l0[i] - lvec[0]) / dl)
            m_idx = int((self.m0[i] - mvec[0]) / dm)
            if l_idx >= 0 and l_idx < lvec.size and m_idx >= 0 and m_idx < mvec.size:
                if self.is_full_stokes():
                    flux_model[m_idx, l_idx] += self.A[i, 0, 0, 0]
                else:
                    flux_model[m_idx, l_idx] += self.A[i, 0]
        return lvec, mvec, flux_model

    def plot(self, save_file: str = None):
        lvec, mvec, flux_model = self.get_flux_model()
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

        im = axs.imshow(flux_model.to('Jy').value, origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]))
        # colorbar
        plt.colorbar(im, ax=axs)
        axs.set_xlabel('l')
        axs.set_ylabel('m')
        if save_file is not None:
            plt.savefig(save_file)
        plt.show()


@dataclasses.dataclass(eq=False)
class PointPredict:
    convention: str = 'physical'

    def check_predict_inputs(self, model_data: PointModelData
                             ) -> Tuple[bool, bool, bool]:
        """
        Check the inputs for predict.

        Args:
            model_data: data, see above for shape info.

        Returns:
            direction_dependent_gains: bool
            full_stokes: bool
            is_gains: bool
        """
        full_stokes = len(np.shape(model_data.image)) == 4 and np.shape(model_data.image)[-2:] == (2, 2)
        is_gains = model_data.gains is not None

        if len(np.shape(model_data.lmn)) != 2 or np.shape(model_data.lmn)[1] != 3:
            raise ValueError(f"Expected lmn to have shape [source, 3], got {np.shape(model_data.lmn)}")

        if len(np.shape(model_data.freqs)) != 1:
            raise ValueError(f"Expected freqs to have shape [chan], got {np.shape(model_data.freqs)}")

        num_sources = np.shape(model_data.lmn)[0]
        num_chan = np.shape(model_data.freqs)[0]

        if full_stokes:
            if np.shape(model_data.image) != (num_sources, num_chan, 2, 2):
                raise ValueError(f"Expected image to have shape [source, chan, 2, 2], got {np.shape(model_data.image)}")
        else:
            if np.shape(model_data.image) != (num_sources, num_chan):
                raise ValueError(f"Expected image to have shape [source, chan], got {np.shape(model_data.image)}")

        if is_gains:
            if full_stokes:
                if len(np.shape(model_data.gains)) == 5:  # [time, ant, chan, 2, 2]
                    time, ant, _, _, _ = np.shape(model_data.gains)
                    direction_dependent_gains = False
                    if np.shape(model_data.gains) != (time, ant, num_chan, 2, 2):
                        raise ValueError(
                            f"Expected gains to have shape [time, ant, chan, 2, 2], got {np.shape(model_data.gains)}."
                        )
                elif len(np.shape(model_data.gains)) == 6:  # [source, time, ant, chan, 2, 2]
                    _, time, ant, _, _, _ = np.shape(model_data.gains)
                    direction_dependent_gains = True
                    if np.shape(model_data.gains) != (num_sources, time, ant, num_chan, 2, 2):
                        raise ValueError(
                            f"Expected gains to have shape [source, time, ant, chan, 2, 2], "
                            f"got {np.shape(model_data.gains)}."
                        )
                else:
                    raise ValueError(
                        f"Expected gains to have shape [source, time, ant, chan, 2, 2] or [time, ant, chan, 2, 2], "
                        f"got {np.shape(model_data.gains)}."
                    )
            else:
                if len(np.shape(model_data.gains)) == 3:  # [time, ant, chan]
                    time, ant, _ = np.shape(model_data.gains)
                    direction_dependent_gains = False
                    if np.shape(model_data.gains) != (time, ant, num_chan):
                        raise ValueError(
                            f"Expected gains to have shape [time, ant, chan], got {np.shape(model_data.gains)}."
                        )
                elif len(np.shape(model_data.gains)) == 4:  # [source, time, ant, chan]
                    _, time, ant, _ = np.shape(model_data.gains)
                    direction_dependent_gains = True
                    if np.shape(model_data.gains) != (num_sources, time, ant, num_chan):
                        raise ValueError(
                            f"Expected gains to have shape [source, time, ant, chan], got {np.shape(model_data.gains)}."
                        )
                else:
                    raise ValueError(
                        f"Expected gains to have shape [source, time, ant, chan] or [time, ant, chan], "
                        f"got {np.shape(model_data.gains)}."
                    )
        else:
            direction_dependent_gains = False

        if is_gains:
            if direction_dependent_gains:
                print(f"Point prediction with unique gains per source.")
            else:
                print(f"Point prediction with shared gains across sources.")

        return direction_dependent_gains, full_stokes, is_gains

    def predict(self, model_data: PointModelData, visibility_coords: VisibilityCoords) -> jax.Array:
        """
        Predict visibilities from DFT model data.

        Args:
            model_data: data, see above for shape info.
            visibility_coords: visibility coordinates.

        Returns:
            visibilities: [row, chan[, 2, 2]] in linear correlation basis.
        """

        direction_dependent_gains, full_stokes, is_gains = self.check_predict_inputs(
            model_data=model_data
        )

        if is_gains:

            _t = visibility_coords.time_idx
            _a1 = visibility_coords.antenna_1
            _a2 = visibility_coords.antenna_2
            if direction_dependent_gains:
                if full_stokes:
                    g1 = model_data.gains[:, _t, _a1, :, :, :]
                    g2 = model_data.gains[:, _t, _a2, :, :, :]
                    g_mapping = "[s,r,c,2,2]"
                else:
                    g1 = model_data.gains[:, _t, _a1, :]
                    g2 = model_data.gains[:, _t, _a2, :]
                    g_mapping = "[s,r,c]"
            else:
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

        # Data will be sharded over frequency so don't reduce over these dimensions, or else communication happens.
        # We want the outer broadcast to be over chan, so we'll do this order.

        # lmn: [source, 3]
        # uvw: [rows, 3]
        # g1, g2: g_mapping
        # freq: [chan]
        # image: [source, chan, 2, 2]
        @partial(
            multi_vmap,
            in_mapping=f"[s,3],[r,3],{g_mapping},{g_mapping},[c],[s,c,2,2]",
            out_mapping="[r,c,...]",
            scan_dims={'s'},
            verbose=True
        )
        def compute_visibility(lmn, uvw, g1, g2, freq, image):
            """
            Compute visibilities for a single row, channel, accumulating over sources.

            Args:
                lmn: [source, 3]
                uvw: [3]
                g1: [[source,] 2, 2] or None
                g2: [[source,] 2, 2] or None
                freq: []
                image: [source, 2, 2]

            Returns:
                vis_accumulation: [2, 2] visibility for given baseline, accumulated over all provided directions.
            """

            if direction_dependent_gains:
                def body_fn(accumulate, x):
                    (lmn, g1, g2, image) = x
                    delta = self._single_compute_visibilty(lmn, uvw, g1, g2, freq, image)  # [] or [2, 2]
                    accumulate += mp_policy.cast_to_vis(delta)
                    return accumulate, ()

                xs = (lmn, g1, g2, image)
            else:
                def body_fn(accumulate, x):
                    (lmn, image) = x
                    delta = self._single_compute_visibilty(lmn, uvw, g1, g2, freq, image)  # [] or [2, 2]
                    accumulate += mp_policy.cast_to_vis(delta)
                    return accumulate, ()

                xs = (lmn, image)
            init_accumulate = jnp.zeros((2, 2) if full_stokes else (), dtype=mp_policy.vis_dtype)
            vis_accumulation, _ = lax.scan(body_fn, init_accumulate, xs, unroll=4)
            return vis_accumulation  # [] or [2, 2]

        visibilities = compute_visibility(
            model_data.lmn,
            visibility_coords.uvw,
            g1,
            g2,
            model_data.freqs,
            model_data.image
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

        phi = (-2j * np.pi) * delay  # scalar
        fringe = jnp.exp(phi) / n

        if np.shape(image) == (2, 2):  # full stokes
            if g1 is None or g1 is None:
                return mp_policy.cast_to_vis(fringe * image)
            return mp_policy.cast_to_vis(fringe) * mp_policy.cast_to_vis(kron_product(g1, image, g2.conj().T))
        elif np.shape(image) == ():
            if g1 is None or g1 is None:
                return mp_policy.cast_to_vis(fringe * image)
            return mp_policy.cast_to_vis(fringe * (g1 * g2.conj() * image))
