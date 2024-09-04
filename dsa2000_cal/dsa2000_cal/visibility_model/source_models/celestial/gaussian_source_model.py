import dataclasses
from functools import partial
from typing import Tuple, NamedTuple

import astropy.coordinates as ac
import astropy.units as au
import jax
import numpy as np
import pylab as plt
import sympy as sp
from astropy import constants
from astropy.coordinates import offset_by
from jax import numpy as jnp, lax
from jax._src.typing import SupportsDType

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.corr_translation import flatten_coherencies, unflatten_coherencies, stokes_I_to_linear
from dsa2000_cal.common.ellipse_utils import ellipse_eval, Gaussian
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.jvp_linear_op import JVPLinearOp
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.common.vec_utils import kron_product
from dsa2000_cal.common.wsclean_util import parse_and_process_wsclean_source_line
from dsa2000_cal.delay_models.far_field import VisibilityCoords


def linear_term_derivation():
    """
    Derives the linear approximation term for w-correction.
    """
    # Define symbols
    l, m, l0, m0, w = sp.symbols('l m l0 m0 w')
    n0 = sp.sqrt(1 - l0 ** 2 - m0 ** 2)
    n = sp.sqrt(1 - l ** 2 - m ** 2)

    # Define the expression inside the brackets of our RIME equation
    zeroth_term = sp.exp(-2 * sp.pi * sp.I * (n0 - 1) * w) / n0

    expression = sp.exp(-2 * sp.pi * sp.I * (n - 1) * w) / n - zeroth_term

    # Compute the first-order Taylor expansion around l0, m0 to second order
    taylor_expansion = expression.subs({l: l0, m: m0}) + (sp.Matrix([expression]).jacobian([l, m])).subs(
        {l: l0, m: m0}).dot(sp.Matrix([l - l0, m - m0])).simplify()

    # pretty print
    sp.pprint(taylor_expansion)

    # (l0*(l - l0) + m0*(m - m0))*(1 + 2*I*pi*w*n0)*exp(-2*I*pi*w*(n0 - 1))/n0**3
    correct = (l0 * (l - l0) + m0 * (m - m0)) * (1 + 2 * sp.I * sp.pi * w * n0) * sp.exp(
        -2 * sp.I * sp.pi * w * (n0 - 1)) / n0 ** 3

    assert taylor_expansion.equals(correct)


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


class GaussianSourceModelParams(SerialisableBaseModel):
    freqs: au.Quantity  # [num_freqs] Frequencies
    l0: au.Quantity  # [num_sources] l coordinate of the source
    m0: au.Quantity  # [num_sources] m coordinate of the source
    A: au.Quantity  # [num_sources, num_freqs] Flux amplitude of the source
    major: au.Quantity  # [num_sources] Major axis of the source
    minor: au.Quantity  # [num_sources] Minor axis of the source
    theta: au.Quantity  # [num_sources] Position angle of the source

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(GaussianSourceModelParams, self).__init__(**data)
        _check_gaussian_source_model_params(self)


def _check_gaussian_source_model_params(params: GaussianSourceModelParams):
    if not params.freqs.unit.is_equivalent(au.Hz):
        raise ValueError(f"Expected freqs to be in Hz, got {params.freqs.unit}")
    if not params.l0.unit.is_equivalent(au.dimensionless_unscaled):
        raise ValueError(f"Expected l0 to be dimensionless, got {params.l0.unit}")
    if not params.m0.unit.is_equivalent(au.dimensionless_unscaled):
        raise ValueError(f"Expected m0 to be dimensionless, got {params.m0.unit}")
    if not params.A.unit.is_equivalent(au.Jy):
        raise ValueError(f"Expected A to be in Jy, got {params.A.unit}")
    if not params.major.unit.is_equivalent(au.dimensionless_unscaled):
        raise ValueError(f"Expected major to be dimensionless, got {params.major.unit}")
    if not params.minor.unit.is_equivalent(au.dimensionless_unscaled):
        raise ValueError(f"Expected minor to be dimensionless, got {params.minor.unit}")
    if not params.theta.unit.is_equivalent(au.rad):
        raise ValueError(f"Expected theta to be in radians, got {params.theta.unit}")

    # Ensure all are 1D vectors
    if params.freqs.isscalar:
        params.freqs = params.freqs.reshape((-1,))
    if params.l0.isscalar:
        params.l0 = params.l0.reshape((-1,))
    if params.m0.isscalar:
        params.m0 = params.m0.reshape((-1,))
    if params.A.isscalar:
        params.A = params.A.reshape((-1, 1))
    if params.major.isscalar:
        params.major = params.major.reshape((-1,))
    if params.minor.isscalar:
        params.minor = params.minor.reshape((-1,))
    if params.theta.isscalar:
        params.theta = params.theta.reshape((-1,))

    num_sources = params.l0.shape[0]
    num_freqs = params.freqs.shape[0]

    if not (params.A.shape == (num_sources, num_freqs) or params.A.shape == (num_sources, num_freqs, 2, 2)):
        raise ValueError(f"A must have shape ({num_sources},{num_freqs},[2,2]) got {params.A.shape}")

    if not all([x.shape == (num_sources,) for x in [params.l0, params.m0, params.major, params.minor, params.theta]]):
        raise ValueError("All inputs must have the same shape")


class GaussianModelData(NamedTuple):
    """
    Data for predict.
    """
    freqs: jax.Array  # [chan]
    image: jax.Array  # [source, chan[, 2, 2]] in [[xx, xy], [yx, yy]] format or stokes
    gains: jax.Array | None  # [[source,] time, ant, chan[, 2, 2]]
    lmn: jax.Array  # [source, 3]
    ellipse_params: jax.Array  # [source, 3] (major, minor, theta)


def transform_ellipsoidal_params_to_plane_of_sky(major: au.Quantity, minor: au.Quantity, theta: au.Quantity,
                                                 source_directions: ac.ICRS, phase_tracking: ac.ICRS,
                                                 lmn_transform_params: bool = True) -> Tuple[
    au.Quantity, au.Quantity, au.Quantity, au.Quantity, au.Quantity]:
    """
    Transform the ellipsoidal parameters to the plane of the sky.

    Args:
        major: [sources] the major axis of the sources
        minor: [sources] the minor axis of the sources
        theta: [sources] the position angle of the sources
        source_directions: [sources] the directions of the sources
        phase_tracking: the phase tracking center
        lmn_transform_params: whether to transform the ellipsoidal parameters to the plane of the sky

    Returns:
        l0: [sources] the l coordinate of the sources in the plane of the sky
        m0: [sources] the m coordinate of the sources in the plane of the sky
        major_tangent: [sources] the major axis of the sources in the plane of the sky
        minor_tangent: [sources] the minor axis of the sources in the plane of the sky
        theta_tangent: [sources] the position angle of the sources in the plane of the sky
    """

    if not major.unit.is_equivalent(au.rad):
        raise ValueError(f"Expected major to be in radians, got {major.unit}")
    if not minor.unit.is_equivalent(au.rad):
        raise ValueError(f"Expected minor to be in radians, got {minor.unit}")
    if not theta.unit.is_equivalent(au.rad):
        raise ValueError(f"Expected theta to be in radians, got {theta.unit}")

    lmn0 = icrs_to_lmn(source_directions, phase_tracking)
    l0 = lmn0[:, 0]
    m0 = lmn0[:, 1]
    n0 = lmn0[:, 2]

    # If you truely treat as ellipsoids on the sphere you get something like this:
    if lmn_transform_params:
        def get_constraint_points(posang, distance):
            s_ra, s_dec = offset_by(
                lon=source_directions.ra, lat=source_directions.dec,
                posang=posang, distance=distance
            )
            s = ac.ICRS(s_ra, s_dec)
            lmn = icrs_to_lmn(s, phase_tracking)
            return lmn[:, 0], lmn[:, 1]

        # Offset by theta and a distance of half-major axis ==> half-major in tangent
        l1, m1 = get_constraint_points(theta, major / 2.)
        # Offset by theta + 90 and a distance of half-minor axis ==> half-minor in tangent
        l2, m2 = get_constraint_points(theta + au.Quantity(90, 'deg'), minor / 2.)

        major_tangent = 2. * np.sqrt((l1 - l0) ** 2 + (m1 - m0) ** 2)
        minor_tangent = 2. * np.sqrt((l2 - l0) ** 2 + (m2 - m0) ** 2)
        theta_tangent = theta
    else:
        major_tangent = major.to(au.rad).value * au.dimensionless_unscaled
        minor_tangent = minor.to(au.rad).value * au.dimensionless_unscaled
        theta_tangent = theta
    return l0, m0, major_tangent, minor_tangent, theta_tangent


@dataclasses.dataclass(eq=False)
class GaussianSourceModel(AbstractSourceModel):
    """
    Predict vis for Gaussian source, with optional first-order approximation.

    Zeroth-order approximation: Constant n=n0.
    First-order approximation: expand linearly around l0, m0, and use polynomial and shift Fourier rules.

    V_ij = V_ij(n=n0) + (l - l0) * dV_ij(l=l0,m=m0)/dl + (m - m0) * dV_ij(l=l0,m=m0)/dm

    ==>

    V_ij = F[I(l,m) * (C + (l - l0) * A + (m - m0) * B)]

    where:

        A = l0 * (1 + 2j * pi * w * n0) * exp(-2j * pi * w * (n0 - 1)) / n0**3
        B = m0 * (1 + 2j * pi * w * n0) * exp(-2j * pi * w * (n0 - 1)) / n0**3
        C = exp(-2j * pi * w * (n0 - 1)) / n0
    """
    freqs: au.Quantity  # [num_freqs] Frequencies
    l0: au.Quantity  # [num_sources] l coordinate of the source
    m0: au.Quantity  # [num_sources] m coordinate of the source
    A: au.Quantity  # [num_sources, num_freqs[,2,2]] Flux amplitude of the source
    major: au.Quantity  # [num_sources] Major axis of the source
    minor: au.Quantity  # [num_sources] Minor axis of the source
    theta: au.Quantity  # [num_sources] Position angle of the source

    def __getitem__(self, item):
        return GaussianSourceModel(
            freqs=self.freqs,
            l0=self.l0[item],
            m0=self.m0[item],
            A=self.A[item],
            major=self.major[item],
            minor=self.minor[item],
            theta=self.theta[item]
        )

    def __add__(self, other: 'GaussianSourceModel') -> 'GaussianSourceModel':
        # ensure freqs same
        if not np.all(self.freqs == other.freqs):
            raise ValueError("Frequencies must match")
        # Ensure both same is_stokes
        if self.is_full_stokes() != other.is_full_stokes():
            raise ValueError("Both must be full stokes or not")
        # concat
        return GaussianSourceModel(
            freqs=self.freqs,
            l0=au.Quantity(np.concatenate([self.l0, other.l0])),
            m0=au.Quantity(np.concatenate([self.m0, other.m0])),
            A=au.Quantity(np.concatenate([self.A, other.A], axis=0)),
            major=au.Quantity(np.concatenate([self.major, other.major])),
            minor=au.Quantity(np.concatenate([self.minor, other.minor])),
            theta=au.Quantity(np.concatenate([self.theta, other.theta])),

        )

    def is_full_stokes(self) -> bool:
        return len(self.A.shape) == 4 and self.A.shape[-2:] == (2, 2)

    def get_lmn_sources(self) -> jax.Array:
        """
        Get the lmn coordinates of the sources.

        Returns:
            lmn: [num_sources, 3] l, m, n coordinates of the sources
        """
        n0 = np.sqrt(1 - self.l0 ** 2 - self.m0 ** 2)
        return jnp.stack(
            [
                quantity_to_jnp(self.l0),
                quantity_to_jnp(self.m0),
                quantity_to_jnp(n0)
            ],
            axis=-1
        )

    def get_model_data(self, gains: jax.Array | None = None) -> GaussianModelData:
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
        ellipse_params = jnp.stack(
            [
                quantity_to_jnp(self.major),
                quantity_to_jnp(self.minor),
                quantity_to_jnp(self.theta)
            ],
            axis=-1
        )
        return GaussianModelData(
            freqs=quantity_to_jnp(self.freqs),
            image=quantity_to_jnp(self.A, 'Jy'),
            lmn=lmn,
            gains=gains,
            ellipse_params=ellipse_params
        )

    def __post_init__(self):
        _check_gaussian_source_model_params(self)

        self.num_sources = self.l0.shape[0]
        self.num_freqs = self.freqs.shape[0]

        self.n0 = np.sqrt(1 - self.l0 ** 2 - self.m0 ** 2)  # [num_sources]
        self.wavelengths = constants.c / self.freqs  # [num_freqs]

    @staticmethod
    def from_gaussian_source_params(params: GaussianSourceModelParams, **kwargs) -> 'GaussianSourceModel':
        return GaussianSourceModel(**params.dict(), **kwargs)

    def total_flux(self) -> au.Quantity:
        return au.Quantity(np.sum(self.A, axis=0))

    def flux_weighted_lmn(self) -> au.Quantity:
        A_avg = np.mean(self.A, axis=1)  # [num_sources]
        m_avg = np.sum(self.m0 * A_avg) / np.sum(A_avg)
        l_avg = np.sum(self.l0 * A_avg) / np.sum(A_avg)
        lmn = np.stack([l_avg, m_avg, np.sqrt(1 - l_avg ** 2 - m_avg ** 2)]) * au.dimensionless_unscaled
        return lmn

    @staticmethod
    def from_wsclean_model(wsclean_clean_component_file: str, phase_tracking: ac.ICRS,
                           freqs: au.Quantity, lmn_transform_params: bool = True,
                           full_stokes: bool = True) -> 'GaussianSourceModel':
        """
        Create a GaussianSourceModel from a wsclean model file.

        Args:
            wsclean_clean_component_file: the wsclean model file
            time: the time of the observation
            phase_tracking: the phase tracking center
            freqs: the frequencies to use
            lmn_transform_params: whether to transform the ellipsoidal parameters to the plane of the sky
            full_stokes: whether the model is full stokes

        Returns:
            GaussianSourceModel: the Gaussian source model
        """
        # Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency='125584411.621094', MajorAxis, MinorAxis, Orientation
        # Example: s0c0,POINT,08:28:05.152,39.35.08.511,0.000748810650400475,[-0.00695379313004673,-0.0849693907803257],false,125584411.621094,,,
        # RA and dec are the central coordinates of the component, in notation of "hh:mm:ss.sss" and "dd.mm.ss.sss".
        # The MajorAxis, MinorAxis and Orientation columns define the shape of the Gaussian.
        # The axes are given in units of arcseconds, and orientation is in degrees.

        source_directions = []
        spectrum = []
        major = []
        minor = []
        theta = []
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
                if parsed_results.type_ != 'GAUSSIAN':
                    continue
                if parsed_results.major is None or parsed_results.minor is None or parsed_results.theta is None:
                    raise ValueError("Major, minor, and theta must be provided for Gaussian sources")
                source_directions.append(parsed_results.direction)
                spectrum.append(parsed_results.spectrum)
                major.append(parsed_results.major)
                minor.append(parsed_results.minor)
                theta.append(parsed_results.theta)

        source_directions = ac.concatenate(source_directions).transform_to(ac.ICRS)

        A = jnp.stack(spectrum, axis=0) * au.Jy  # [num_sources, num_freqs]
        # The ellipsoidal parameters are on the sphere. Let's do a transform to plane of sky.
        major = au.Quantity(major)
        minor = au.Quantity(minor)
        theta = au.Quantity(theta)

        l0, m0, major_tangent, minor_tangent, theta_tangent = transform_ellipsoidal_params_to_plane_of_sky(
            major=major,
            minor=minor,
            theta=theta,
            source_directions=source_directions,
            phase_tracking=phase_tracking,
            lmn_transform_params=lmn_transform_params
        )

        if full_stokes:
            A = np.asarray(stokes_I_image_to_linear(quantity_to_jnp(A, 'Jy'), flat_output=False)) * au.Jy
        return GaussianSourceModel(
            freqs=freqs,
            l0=l0,
            m0=m0,
            A=A,
            major=major_tangent,
            minor=minor_tangent,
            theta=theta_tangent
        )

    def get_flux_model(self, lvec=None, mvec=None):
        if self.is_full_stokes():
            raise ValueError("Cannot plot full stokes.")

        # Use imshow to plot the sky model evaluated over a LM grid
        if lvec is None or mvec is None:
            l_min = np.min(self.l0 - self.major) - 0.01
            l_max = np.max(self.l0 + self.major) + 0.01
            m_min = np.min(self.m0 - self.major) - 0.01
            m_max = np.max(self.m0 + self.major) + 0.01
            lvec = np.linspace(l_min.value, l_max.value, 256)
            mvec = np.linspace(m_min.value, m_max.value, 256)

        dl = lvec[1] - lvec[0]
        dm = mvec[1] - mvec[0]

        M, L = np.meshgrid(mvec, lvec, indexing='ij')
        # Evaluate over LM
        flux_density = np.zeros_like(L) * au.Jy

        @jax.jit
        def _gaussian_flux(A, l0, m0, major, minor, theta):
            return jax.vmap(
                lambda l, m: ellipse_eval(A, major, minor, theta, l, m, l0, m0)
            )(jnp.asarray(L).flatten(), jnp.asarray(M).flatten()).reshape(L.shape)

        pixel_area = (lvec[1] - lvec[0]) * (mvec[1] - mvec[0])

        for i in range(self.num_sources):
            args = (
                quantity_to_jnp(self.A[i, 0], 'Jy'),
                quantity_to_jnp(self.l0[i]),
                quantity_to_jnp(self.m0[i]),
                quantity_to_jnp(self.major[i]),
                quantity_to_jnp(self.minor[i]),
                quantity_to_jnp(self.theta[i]))
            flux_density += np.asarray(_gaussian_flux(*args)) * au.Jy
        return lvec, mvec, flux_density * pixel_area

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


def derive_transform():
    # u_prime = u * minor * jnp.cos(theta) - v * minor * jnp.sin(theta)
    # v_prime = u * major * jnp.sin(theta) + v * major * jnp.cos(theta)
    u_prime, v_prime, u, v, theta, major, minor = sp.symbols('uprime vprime u v theta major minor')
    # Solve for u, v
    solution = sp.solve([u_prime - u * minor * sp.cos(theta) + v * minor * sp.sin(theta),
                         v_prime - u * major * sp.sin(theta) - v * major * sp.cos(theta)], (u, v))
    print(solution[u].simplify())
    print(solution[v].simplify())


@dataclasses.dataclass(eq=False)
class GaussianPredict:
    order_approx: int = 0
    convention: str = 'physical'
    dtype: SupportsDType = jnp.complex64

    def check_predict_inputs(self, model_data: GaussianModelData
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

        if np.shape(model_data.lmn) != np.shape(model_data.ellipse_params):
            raise ValueError(
                f"Expected lmn and ellipse_params to have the same shape, "
                f"got {np.shape(model_data.lmn)} and {np.shape(model_data.ellipse_params)}"
            )

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

    def predict(self, model_data: GaussianModelData, visibility_coords: VisibilityCoords) -> jax.Array:
        """
        Predict visibilities from Gaussian model data.

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

        # lmn: [source, 3]
        # uvw: [rows, 3]
        # g1, g2: [[source,] row, chan, 2, 2]
        # freq: [chan]
        # image: [source, chan, 2, 2]
        # ellipse_params: [source, 3]
        @partial(
            multi_vmap,
            in_mapping=f"[s,3],[r,3],{g_mapping},{g_mapping},[c],[s,c,2,2],[s,3]",
            out_mapping="[r,c,...]",
            scan_dims={'s'},
            verbose=True
        )
        def compute_visibility(lmn, uvw, g1, g2, freq, image, ellipse_params):

            if direction_dependent_gains:
                def body_fn(accumulate, x):
                    (lmn, g1, g2, image, ellipse_params) = x
                    delta = self._single_compute_visibilty(lmn, uvw, g1, g2, freq, image,
                                                           ellipse_params)  # [] or [2, 2]
                    accumulate += delta
                    return accumulate, ()

                xs = (lmn, g1, g2, image, ellipse_params)
            else:
                def body_fn(accumulate, x):
                    (lmn, image, ellipse_params) = x
                    delta = self._single_compute_visibilty(lmn, uvw, g1, g2, freq, image,
                                                           ellipse_params)  # [] or [2, 2]
                    accumulate += delta
                    return accumulate, ()

                xs = (lmn, image, ellipse_params)

            init_accumulate = jnp.zeros((2, 2) if full_stokes else (), dtype=self.dtype)
            vis_accumulation, _ = lax.scan(body_fn, init_accumulate, xs, unroll=4)
            return vis_accumulation  # [] or [2, 2]

        visibilities = compute_visibility(
            model_data.lmn,
            visibility_coords.uvw,
            g1,
            g2,
            model_data.freqs,
            model_data.image,
            model_data.ellipse_params
        )  # [row, chan[, 2, 2]]
        return visibilities

    def _single_compute_visibilty(self, lmn, uvw, g1, g2, freq, image, ellipse_params):
        """
        Compute the visibility from a single direction for a single baseline.

        Args:
            lmn: [3]
            uvw: [3]
            g1: [] or [2, 2]
            g2: [] or [2, 2]
            freq: []
            image: [] or [2, 2]

        Returns:
            [2, 2] visibility in given direction for given baseline.
        """
        wavelength = quantity_to_jnp(constants.c) / freq

        if self.convention == 'casa':
            uvw = jnp.negative(uvw)

        uvw /= wavelength

        u, v, w = uvw  # scalar

        l0, m0, n0 = lmn  # scalar
        major, minor, theta = ellipse_params

        if np.shape(image) == (2, 2):  # full stokes
            if g1 is not None and g2 is not None:
                image = flatten_coherencies(kron_product(g1, image, g2.conj().T))  # [4]
            vis = jax.vmap(
                lambda A: self._single_predict(
                    u, v, w,
                    A=A,
                    l0=l0,
                    m0=m0,
                    n0=n0,
                    major=major,
                    minor=minor,
                    theta=theta
                )
            )(image)  # [4]
            vis = unflatten_coherencies(vis)  # [2,2]
        else:
            if g1 is not None and g2 is not None:
                image = g1 * image * g2.conj()
            vis = self._single_predict(
                u, v, w,
                A=image,
                l0=l0,
                m0=m0,
                n0=n0,
                major=major,
                minor=minor,
                theta=theta
            )  # []
        return vis

    def _gaussian_fourier(self, u, v, A, l0, m0, major, minor, theta):
        """
        Computes the Fourier transform of the Gaussian source, over given u, v coordinates.

        Args:
            u: scalar
            v: scalar
            A: scalar
            l0: scalar
            m0: scalar
            major: scalar
            minor: scalar
            theta: scalar

        Returns:
            Fourier transformed Gaussian source evaluated at uvw
        """
        gaussian = Gaussian(
            x0=jnp.asarray([l0, m0]),
            major_fwhm=major,
            minor_fwhm=minor,
            pos_angle=theta,
            total_flux=A
        )
        return gaussian.fourier(jnp.asarray([u, v]))

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
            return gaussian.fourier(jnp.asarray([u, v]))

        def wkernel(l, m):
            n = jnp.sqrt(1. - l ** 2 - m ** 2)
            return jnp.exp(-2j * jnp.pi * w * (n - 1.)) / n

        if self.order_approx == 0:
            vis = F_gaussian(u, v) * wkernel(l0, m0)
        elif self.order_approx == 1:

            # Let I(l,m) * W(l,m) ~= I(l,m) * (W(l0, m0) + W_l * (l - l0) + W_m * (m - m0))
            # Where W_l = d/dl W(l0,m0), W_m = d/dm W(l0,m0)
            # F[I(l,m) * W(l,m)] ~= F[I(l,m) * W(l0,m0) + I(l,m) * W_l * (l - l0) + I(l,m) * W_m * (m - m0)]
            #  = (W0 - l0 * W_l - m0 * W_m) * F[I(l,m)] + (d/du * F[I(l,m)] * (W_l) + d/dv * F[I(l,m)] * (W_m)) / (-2 pi i)

            # maybe divide by 2pi
            wkernel_grad = jax.value_and_grad(wkernel, (0, 1), holomorphic=True)

            W0, (Wl, Wm) = wkernel_grad(jnp.asarray(l0, self.dtype), jnp.asarray(m0, self.dtype))

            F_jvp = JVPLinearOp(F_gaussian, promote_dtypes=True)
            vec = (
                jnp.asarray(Wl, self.dtype), jnp.asarray(Wm, self.dtype)
            )
            # promote_dtypes=True so we don't need to cast the primals here. Otherwise:
            # primals = (u.astype(vec[0].dtypegrad), v.astype(vec[1].dtype))
            primals = (u, v)
            F_jvp = F_jvp(*primals)

            vis = F_gaussian(u, v) * (W0 - l0 * Wl - m0 * Wm) + F_jvp.matvec(*vec) / (-2j * jnp.pi)
        else:
            raise ValueError("order_approx must be 0 or 1")
        return vis
