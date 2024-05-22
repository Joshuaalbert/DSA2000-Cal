import dataclasses
from typing import Tuple

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pylab as plt
import sympy as sp
from astropy import constants
from astropy.coordinates import offset_by
from jax import numpy as jnp
from jax._src.typing import SupportsDType

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.ellipse_utils import ellipse_eval
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.source_models.wsclean_util import parse_and_process_wsclean_source_line


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

    if params.A.shape != (num_sources, num_freqs):
        raise ValueError(f"A must have shape ({num_sources},{num_freqs}) got {params.A.shape}")

    if not all([x.shape == (num_sources,) for x in [params.l0, params.m0, params.major, params.minor, params.theta]]):
        raise ValueError("All inputs must have the same shape")


def transform_ellipsoidal_params_to_plane_of_sky(
        major: au.Quantity,
        minor: au.Quantity,
        theta: au.Quantity,
        source_directions: ac.ICRS,
        phase_tracking: ac.ICRS,
        obs_time: at.Time,
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
        obs_time: the time of the observation
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

    lmn0 = icrs_to_lmn(source_directions, obs_time, phase_tracking)
    l0 = lmn0[:, 0]
    m0 = lmn0[:, 1]

    # If you truely treat as ellipsoids on the sphere you get something like this:
    if lmn_transform_params:
        def get_constraint_points(posang, distance):
            s_ra, s_dec = offset_by(
                lon=source_directions.ra, lat=source_directions.dec,
                posang=posang, distance=distance
            )
            s = ac.ICRS(s_ra, s_dec)
            lmn = icrs_to_lmn(s, obs_time, phase_tracking)
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
    A: au.Quantity  # [num_sources, num_freqs] Flux amplitude of the source
    major: au.Quantity  # [num_sources] Major axis of the source
    minor: au.Quantity  # [num_sources] Minor axis of the source
    theta: au.Quantity  # [num_sources] Position angle of the source
    order_approx: int = 0

    dtype: SupportsDType = jnp.complex64

    def __post_init__(self):
        _check_gaussian_source_model_params(self)

        self.num_sources = self.l0.shape[0]
        self.num_freqs = self.freqs.shape[0]

        self.n0 = np.sqrt(1 - self.l0 ** 2 - self.m0 ** 2)  # [num_sources]
        self.wavelengths = constants.c / self.freqs  # [num_freqs]

    @staticmethod
    def from_gaussian_source_params(params: GaussianSourceModelParams, **kwargs) -> 'GaussianSourceModel':
        return GaussianSourceModel(**params.dict(), **kwargs)

    def flux_weighted_lmn(self) -> au.Quantity:
        A_avg = np.mean(self.A, axis=1)  # [num_sources]
        m_avg = np.sum(self.m0 * A_avg) / np.sum(A_avg)
        l_avg = np.sum(self.l0 * A_avg) / np.sum(A_avg)
        lmn = np.stack([l_avg, m_avg, np.sqrt(1 - l_avg ** 2 - m_avg ** 2)]) * au.dimensionless_unscaled
        return lmn

    @staticmethod
    def from_wsclean_model(wsclean_clean_component_file: str, time: at.Time, phase_tracking: ac.ICRS,
                           freqs: au.Quantity, lmn_transform_params: bool = True, **kwargs) -> 'GaussianSourceModel':
        """
        Create a GaussianSourceModel from a wsclean model file.

        Args:
            wsclean_clean_component_file: the wsclean model file
            time: the time of the observation
            phase_tracking: the phase tracking center
            freqs: the frequencies to use
            lmn_transform_params: whether to transform the ellipsoidal parameters to the plane of the sky
            **kwargs:

        Returns:

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

        A = jnp.stack(spectrum, axis=0) * au.Jy
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
            obs_time=time,
            lmn_transform_params=lmn_transform_params
        )

        return GaussianSourceModel(
            freqs=freqs,
            l0=l0,
            m0=m0,
            A=A,
            major=major_tangent,
            minor=minor_tangent,
            theta=theta_tangent,
            **kwargs
        )

    def get_flux_model(self, lvec=None, mvec=None):
        # Use imshow to plot the sky model evaluated over a LM grid
        if lvec is None or mvec is None:
            l_min = np.min(self.l0 - self.major) - 0.01
            l_max = np.max(self.l0 + self.major) + 0.01
            m_min = np.min(self.m0 - self.major) - 0.01
            m_max = np.max(self.m0 + self.major) + 0.01
            lvec = np.linspace(l_min.value, l_max.value, 256)
            mvec = np.linspace(m_min.value, m_max.value, 256)
        M, L = np.meshgrid(mvec, lvec, indexing='ij')
        # Evaluate over LM
        flux_model = np.zeros_like(L) * au.Jy

        @jax.jit
        def _gaussian_flux(A, l0, m0, major, minor, theta):
            return jax.vmap(
                lambda l, m: ellipse_eval(A, major, minor, theta, l, m, l0, m0)
            )(jnp.asarray(L).flatten(), jnp.asarray(M).flatten()).reshape(L.shape)

        pixel_area = (lvec[1] - lvec[0]) * (mvec[1] - mvec[0])
        print(quantity_to_jnp(self.A[0, :], 'Jy'))

        for i in range(self.num_sources):
            args = (
                quantity_to_jnp(self.A[i, 0], 'Jy'),
                quantity_to_jnp(self.l0[i]),
                quantity_to_jnp(self.m0[i]),
                quantity_to_jnp(self.major[i]),
                quantity_to_jnp(self.minor[i]),
                quantity_to_jnp(self.theta[i]))
            beam_area = (np.pi / (2. * np.log(2.))) * quantity_to_jnp(self.major[i]) * quantity_to_jnp(self.minor[i])
            flux_model += np.asarray(_gaussian_flux(*args)) * pixel_area / beam_area * au.Jy
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

    def __add__(self, other: 'GaussianSourceModel') -> 'GaussianSourceModel':
        if not np.all(self.freqs == other.freqs):
            raise ValueError("Frequency mismatch")
        if not np.all(self.order_approx == other.order_approx):
            raise ValueError("Order approximation mismatch")
        return GaussianSourceModel(
            freqs=self.freqs,
            l0=au.Quantity(np.concatenate([self.l0, other.l0])),
            m0=au.Quantity(np.concatenate([self.m0, other.m0])),
            A=au.Quantity(np.concatenate([self.A, other.A], axis=0)),
            major=au.Quantity(np.concatenate([self.major, other.major])),
            minor=au.Quantity(np.concatenate([self.minor, other.minor])),
            theta=au.Quantity(np.concatenate([self.theta, other.theta])),
            order_approx=self.order_approx,
            dtype=self.dtype
        )


def derive_transform():
    # u_prime = u * minor * jnp.cos(theta) - v * minor * jnp.sin(theta)
    # v_prime = u * major * jnp.sin(theta) + v * major * jnp.cos(theta)
    u_prime, v_prime, u, v, theta, major, minor = sp.symbols('uprime vprime u v theta major minor')
    # Solve for u, v
    solution = sp.solve([u_prime - u * minor * sp.cos(theta) + v * minor * sp.sin(theta),
                         v_prime - u * major * sp.sin(theta) - v * major * sp.cos(theta)], (u, v))
    print(solution[u].simplify())
    print(solution[v].simplify())
