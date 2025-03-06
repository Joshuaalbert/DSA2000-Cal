import dataclasses
import pickle
import warnings
from abc import abstractmethod, ABC
from functools import partial
from typing import Tuple, List, Any

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.interp_utils import InterpolatedArray
from dsa2000_common.common.jax_utils import multi_vmap
from dsa2000_common.common.quantity_utils import time_to_jnp

tfpd = tfp.distributions


def compute_ionosphere_intersection(
        x_gcrs, k_gcrs,
        x0_gcrs,
        bottom: FloatArray, width: FloatArray
):
    """
    Compute the intersection of a geodesic with the ionosphere layer, which is seen as a spherical shell around Earth.

    Args:
        x_gcrs: [3] the origin of the geodesic
        k_gcrs: [3] the direction of the geodesic
        x0_gcrs: [3] the reference point where ionosphere is measured from.
        bottom: the bottom of the ionosphere layer
        width: the width of the ionosphere layer

    Returns:
        the intersection points along the geodesic
    """
    # |x(smin)| = |x + smin * k| = |x0| + bottom = v
    # ==> |x + smin * k|^2 = (|x0| + bottom)^2 = v^2
    # ==> x^2 + 2 smin x . k + smin^2 - v^2 = 0
    # ==> smin = - (x . k) +- sqrt((x . k)^2 +(x^2 - v^2))
    # choose the positive root
    xk = x_gcrs @ k_gcrs
    xx = x_gcrs @ x_gcrs
    x0_norm = jnp.linalg.norm(x0_gcrs)
    vmin = x0_norm + bottom
    vmax = vmin + width

    smin = - xk + jnp.sqrt(xk ** 2 - (xx - vmin ** 2))
    smax = - xk + jnp.sqrt(xk ** 2 - (xx - vmax ** 2))
    return smin, smax


def calc_intersections(x, s, x0_radius, bottom, width, s_normed: bool = False):
    """
    Compute the intersection of geodesic with top and bottom of layer.

    Args:
        x: [3] the antenna position in GCRS
        s: [3] the direction in GCRS (may be non-normed)

    Returns:
        the bottom and top
    """
    # Let a=x, c=u*s, b=a+c, we want to find u such that |b|^2 = (x0_radius + dr)^2
    # Cosine rule: u^2 s^2 = c^2 = a^2 + b^2 - 2 a.b
    # = a^2 + b^2 - 2 a.(a + u*s) = b^2 - a^2 - 2 u a.s
    # Quadratic formula in u gives: u = 1/s^2 (-a.s +- sqrt((a.s)^2 - a^2 + b^2))
    # Now use b^2 = (x0_radius + dr)^2 = x0_radius^2 + 2 x0_radius dr + dr^2
    # ==> u = 1/s^2 (-a.s +- sqrt((a.s)^2 + x0_radius^2 - a^2 + 2 x0_radius dr + dr^2))
    if not s_normed:
        # cheapest is just to norm s
        s = s / jnp.linalg.norm(s)
    x2 = jnp.sum(x ** 2)
    x_norm = jnp.sqrt(x2)
    two_x_norm = 2 * x_norm
    xs = jnp.sum(x * s)
    xs2 = xs ** 2
    x02_minus_x2 = x0_radius ** 2 - x2
    _tmp = xs2 + x02_minus_x2
    dr = bottom
    s_bottom = -xs + jnp.sqrt(_tmp + two_x_norm * dr + dr ** 2)
    dr = bottom + width
    s_top = -xs + jnp.sqrt(_tmp + two_x_norm * dr + dr ** 2)
    return s_bottom, s_top


@dataclasses.dataclass
class GaussianLineIntegral:
    mu: FloatArray | None
    Sigma: FloatArray
    Sigma_inv: FloatArray = None
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return
        # self.D = np.shape(self.mu)[-1]
        if self.Sigma_inv is None:
            self.Sigma_inv = jnp.linalg.inv(self.Sigma)
        # self.norm_factor = 1 / ((2 * np.pi) ** (self.D / 2) * jnp.sqrt(jnp.linalg.det(self.Sigma)))

    def gaussian(self, x):
        if self.mu is not None:
            dx = x - self.mu
        else:
            dx = x
        return jnp.exp(-0.5 * dx.T @ self.Sigma_inv @ dx)

    def finite_integral(self, a, u, t1, t2):
        """Closed-form solution for line integral through 3D Gaussian"""

        if self.mu is not None:
            d = a - self.mu
        else:
            d = a

        # A = u.T @ self.Sigma_inv @ u
        # B = d.T @ self.Sigma_inv @ u
        # C = d.T @ self.Sigma_inv @ d
        u_prime = self.Sigma_inv @ u
        d_prime = self.Sigma_inv @ d
        A = u.T @ u_prime
        B = d.T @ u_prime
        C = d.T @ d_prime

        B_over_A = B / A

        exponent = -0.5 * (C - B ** 2 / A)
        erf_arg = jnp.sqrt(A / 2)
        s1 = t1 + B_over_A
        s2 = t2 + B_over_A

        integral = (jnp.exp(exponent) * jnp.sqrt((0.5 * np.pi) / A)
                    * (jax.lax.erf(erf_arg * s2) - jax.lax.erf(erf_arg * s1)))
        integral = jnp.where(A < 1e-12, 0., integral)
        return integral

    def infinite_integral(self, a, u):
        if self.mu is not None:
            d = a - self.mu
        else:
            d = a
        # A = u.T @ self.Sigma_inv @ u
        # B = d.T @ self.Sigma_inv @ u
        # C = d.T @ self.Sigma_inv @ d
        u_prime = self.Sigma_inv @ u
        d_prime = self.Sigma_inv @ d
        A = u.T @ u_prime
        B = d.T @ u_prime
        C = d.T @ d_prime

        exponent = -0.5 * (C - B ** 2 / A)

        integral = jnp.exp(exponent) * jnp.sqrt((2 * np.pi) / A)
        integral = jnp.where(A < 1e-12, 0., integral)
        return integral


class AbstractIonosphereLayer(ABC):

    @abstractmethod
    def compute_kernel(self, x1, s1, t1, x2, s2, t2, resolution: int, s_normed: bool = False):
        """
        Compute the covariance.

        Args:
            x1: [D, T, A] GCRS position of antenna 1
            s1: [D, T, A] the GCRS direction of geodesic 1
            t1: [D, T, A] the TT earth time at antenna 1
            x2: [D', T', A'] GCRS position of antenna 2
            s2: [D', T', A'] the GCRS direction of geodesic 2
            t2: [D', T', A'] the TT earth time at antenna 2
            resolution: the number of points to use for the integral
            s_normed: whether direction is normed

        Returns:
            the covariance between both geodesics.
        """
        ...

    @abstractmethod
    def compute_mean(self, x, s, t, s_normed: bool = False):
        """
        Compute the mean.

        Args:
            x: [D, T, A] the GCRS position of antenna
            s: [D, T, A] the GCRS direction of geodesic
            t: [D, T, A] the times
            s_normed: whether the directions are normed

        Returns:
            the mean
        """
        ...

    def compute_geodesic_coords(
            self,
            antennas_gcrs: InterpolatedArray,
            times: FloatArray,
            directions_gcrs: InterpolatedArray
    ):
        """
        Compute the covariance and mean of generative marginal process.

        Args:
            antennas_gcrs: interp (t) -> [A, 3]
            times: [T] times
            directions_gcrs: interp (t) -> [D, 3]

        Returns:
            [D,T,A] x, shat, t
        """

        @partial(
            multi_vmap,
            in_mapping="[T]",
            out_mapping="[T,...],[T,...],[T]"
        )
        def get_coords(time):
            x = antennas_gcrs(time)  # [A, 3]
            s = directions_gcrs(time)  # [D, 3]
            shat = s / jnp.linalg.norm(s, axis=-1, keepdims=True)
            x, shat = jnp.broadcast_arrays(x[None, :, :], shat[:, None, :])
            t = jnp.broadcast_to(time[None, None], np.shape(x)[:-1])
            return x, shat, t

        x, shat, t = get_coords(times)  # [T, D, A, 3], [T, D, A, 3], [T, D, A]
        x = jax.lax.transpose(x, [1, 0, 2, 3])
        shat = jax.lax.transpose(shat, [1, 0, 2, 3])
        t = jax.lax.transpose(t, [1, 0, 2])
        D, T, A = jnp.shape(t)
        return x, shat, t

    def compute_tec_process_params(self, antennas_gcrs: InterpolatedArray,
                                   times: FloatArray, directions_gcrs: InterpolatedArray, resolution: int):
        x1, s1, t1 = x2, s2, t2 = self.compute_geodesic_coords(antennas_gcrs, times, directions_gcrs)
        K = self.compute_kernel(
            x1, s1, t1, x2, s2, t2, s_normed=True, resolution=resolution
        )
        mean = self.compute_mean(x1, s1, t1, s_normed=True)
        return K, mean

    def compute_dtec_process_params(self,
                                    reference_antenna_gcrs: InterpolatedArray,
                                    antennas_gcrs: InterpolatedArray,
                                    times: FloatArray,
                                    directions_gcrs: InterpolatedArray,
                                    resolution: int = 27):
        x1, s1, t1 = x2, s2, t2 = self.compute_geodesic_coords(antennas_gcrs, times, directions_gcrs)
        K11 = self.compute_kernel(
            x1, s1, t1, x2, s2, t2, s_normed=True, resolution=resolution
        )
        mean1 = self.compute_mean(x1, s1, t1, s_normed=True)

        x0, s0, t0 = self.compute_geodesic_coords(reference_antenna_gcrs, times, directions_gcrs)
        K01 = self.compute_kernel(
            x0, s0, t0, x2, s2, t2, s_normed=True, resolution=resolution
        )
        K10 = self.compute_kernel(
            x1, s1, t1, x0, s0, t0, s_normed=True, resolution=resolution
        )
        K00 = self.compute_kernel(
            x0, s0, t0, x0, s0, t0, s_normed=True, resolution=resolution
        )
        mean0 = self.compute_mean(x0, s0, t0, s_normed=True)

        K = K11 + K00 - (K01 + K10)
        mean = mean1 - mean0
        return K, mean

    def sample_tec(self, key, antennas_gcrs: InterpolatedArray,
                   times: FloatArray,
                   directions_gcrs: InterpolatedArray,
                   jitter_mtec=0.5, resolution: int = 27):
        """
        Sample ionosphere TEC.

        Args:
            key: PRNGKey
            antennas_gcrs: [A] antennas (time) -> [A, 3]
            times: [T] times
            directions_gcrs: [D] directions (time) -> [D, 3]
            jitter_mtec: how much diagonal jitter, equivalent to adding white noise.
            resolution: how many resolution elements to use, default tuned to DSA2000

        Returns:
            [D, T, A] shaped array of DTEC or TEC
        """
        K, mean = self.compute_tec_process_params(antennas_gcrs, times, directions_gcrs, resolution)
        D, T, A = np.shape(mean)

        K = jax.lax.reshape(K, (D * T * A, D * T * A))
        mean = jax.lax.reshape(mean, [D * T * A])

        # Sample now

        # Efficient add to diagonal
        diag_idxs = jnp.diag_indices(K.shape[0])
        K = K.at[diag_idxs].add(jitter_mtec ** 2)

        sample = tfpd.MultivariateNormalTriL(
            loc=mean,
            scale_tril=jnp.linalg.cholesky(K)
        ).sample(
            seed=key
        )

        sample = jax.lax.reshape(sample, [D, T, A])
        return sample

    def sample_dtec(self, key,
                    reference_antenna_gcrs: InterpolatedArray,
                    antennas_gcrs: InterpolatedArray,
                    times: FloatArray,
                    directions_gcrs: InterpolatedArray,
                    jitter_mtec=0.5, resolution: int = 27):
        """
        Sample ionosphere DTEC.

        Args:
            key: PRNGKey
            reference_antenna_gcrs: [1] atennas (time) -> [1, 3]
            antennas_gcrs: [A] antennas (time) -> [A, 3]
            times: [T] times
            directions_gcrs: [D] directions (time) -> [D, 3]
            jitter_mtec: how much diagonal jitter, equivalent to adding white noise.
            resolution: how many resolution elements to use, default tuned to DSA2000

        Returns:
            [D, T, A] shaped array of DTEC or TEC
        """
        K, mean = self.compute_dtec_process_params(reference_antenna_gcrs, antennas_gcrs, times, directions_gcrs,
                                                   resolution)

        D, T, A = np.shape(mean)

        K = jax.lax.reshape(K, (D * T * A, D * T * A))
        mean = jax.lax.reshape(mean, [D * T * A])

        # Sample now

        # Efficient add to diagonal
        diag_idxs = jnp.diag_indices(K.shape[0])
        K = K.at[diag_idxs].add(jitter_mtec ** 2)

        sample = tfpd.MultivariateNormalTriL(
            loc=mean,
            scale_tril=jnp.linalg.cholesky(K)
        ).sample(
            seed=key
        )

        sample = jax.lax.reshape(sample, [D, T, A])
        return sample


@dataclasses.dataclass(eq=False)
class IonosphereLayer(AbstractIonosphereLayer):
    """
    An ionosphere layer with Gaussian radial basis function parametrisation. This is equivalent to traditional RBF,
    or exponentiated quadratic kernel.

    Units:
    - Distances in [km]
    - Densities in [10^10 e/m^3]
    - TEC in [mTECU]
    """
    length_scale: FloatArray  # [km]
    longitude_pole: FloatArray  # [rad]
    latitude_pole: FloatArray  # [rad]
    bottom_velocity: FloatArray  # [km/s]
    radial_velocity: FloatArray  # [km/s]
    x0_radius: FloatArray  # [km]
    bottom: FloatArray  # [km]
    width: FloatArray  # [km]
    fed_mu: FloatArray  # [1e10 e-/m^3]
    fed_sigma: FloatArray  # [1e10 e-/m^3]
    method: str = 'semi_analytic'
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return

    def double_tomographic_integral(self, x1, x2, s1, s2, s1m, s1p, s2m, s2p, resolution: int):
        """
        Compute the double integral of the Gaussian kernel over two lines, using a semi-analytic solution.

        Args:
            x1: [3] array of the first position
            x2: [3] array of the second position
            s1: [3] array of the first direction
            s2: [3] array of the second direction
            s1m: [3] array of the first intersection for the first line
            s1p: the second intersection for the first line
            s2m: the first intersection for the second line
            s2p: the second intersection for the second line
            resolution: the number of points to use for the integral

        Returns:
            the integral
        """

        eye = jnp.eye(3)
        Sigma = eye * self.length_scale ** 2
        Sigma_inv = eye / self.length_scale ** 2
        gaussian_line_integral = GaussianLineIntegral(None, Sigma, Sigma_inv=Sigma_inv)

        def integrand(t2):
            a = x1 - x2 - t2 * s2
            u = s1
            return gaussian_line_integral.finite_integral(a, u, s1m, s1p)

        ds2 = (s2p - s2m) / resolution
        t2_array = jnp.linspace(s2m, s2p, resolution + 1)
        summnd = jax.vmap(integrand)(t2_array)
        integral = 0.5 * jnp.sum(summnd[:-1] + summnd[1:]) * ds2
        return self.fed_sigma ** 2 * integral

    def numerical_double_integral(self, x1, x2, s1, s2, s1m, s1p, s2m, s2p, resolution: int):
        eye = jnp.eye(3)
        Sigma = eye * self.length_scale ** 2
        Sigma_inv = eye / self.length_scale ** 2
        gaussian_line_integral = GaussianLineIntegral(None, Sigma, Sigma_inv=Sigma_inv)

        def integrand(t1, t2):
            x = x1 + t1 * s1 - x2 - t2 * s2
            return gaussian_line_integral.gaussian(x)

        t1_array = jnp.linspace(s1m, s1p, resolution + 1)
        t2_array = jnp.linspace(s2m, s2p, resolution + 1)
        ds1 = (s1p - s1m) / resolution
        ds2 = (s2p - s2m) / resolution
        integral = self.fed_sigma ** 2 * jnp.sum(
            jax.vmap(lambda s1: jax.vmap(lambda s2: integrand(s1, s2))(t2_array))(t1_array)) * ds1 * ds2
        return integral

    def project_geodesic(self, x, s, s_normed: bool = False):
        """
        Project a geodesic in GCRS to the bottom and top of layer.

        Args:
            x: [3] GCRS position of antenna
            s: [3] direction of geodesic

        Returns:
            the GCRS position of top and bottom of layer at intersection.
        """
        s_bottom, s_top = calc_intersections(x, s, self.x0_radius, self.bottom, self.width, s_normed=s_normed)
        x_bottom = x + s_bottom * s
        x_top = x + s_top * s
        return x_bottom, x_top

    def apply_frozen_flow(self, x_proj, t):
        """
        Apply the frozen flow to a projected point.

        Args:
            x_proj: [3] the projected point on layer in GCRS
            t: the time of flow since reference point

        Returns:
            [3] the point within the reference field
        """
        bottom_radius = self.x0_radius + self.bottom
        # circumference
        # C = 2 * np.pi * bottom_radius # [km]
        # time to go around
        # T = C  / self.bottom_velocity # [s]
        omega = self.bottom_velocity / bottom_radius  # [rad / s]
        return efficient_rodriges_rotation(
            x_proj=x_proj,
            rdot=self.radial_velocity,
            omega=omega,
            dt=-t,
            alpha_pole=self.longitude_pole,
            delta_pole=self.latitude_pole
        )

    def _compute_kernel(self, x1, s1, t1, x2, s2, t2, resolution: int, s_normed: bool = False):
        """
        Compute the kernel.

        Args:
            x1: [3] GCRS position of antenna 1
            s1: [3] the GCRS direction of geodesic 1
            t1: the TT earth time at antenna 1
            x2: [3] GCRS position of antenna 2
            s2: [3] the GCRS direction of geodesic 2
            t2: the TT earth time at antenna 2
            resolution: how many resolution elements to use, default tuned to DSA2000

        Returns:
            the TEC covariance between both geodesics.
        """
        # project both to intersections with layer
        x1_bottom, x1_top = self.project_geodesic(x1, s1, s_normed=s_normed)
        x2_bottom, x2_top = self.project_geodesic(x2, s2, s_normed=s_normed)
        # Apply frozen flow, to find point in reference field
        x1_bottom = self.apply_frozen_flow(x1_bottom, t1)
        x1_top = self.apply_frozen_flow(x1_top, t1)
        x2_bottom = self.apply_frozen_flow(x2_bottom, t2)
        x2_top = self.apply_frozen_flow(x2_top, t2)
        # Define new vectors
        s1 = x1_top - x1_bottom
        s2 = x2_top - x2_bottom
        one = jnp.ones(())
        zero = jnp.zeros(())
        if self.method == 'semi_analytic':
            return self.double_tomographic_integral(
                x1_bottom, x2_bottom, s1, s2,
                zero, one, zero, one, resolution=resolution
            )
        else:
            return self.numerical_double_integral(
                x1_bottom, x2_bottom, s1, s2,
                zero, one, zero, one, resolution=resolution
            )

    def _compute_mean(self, x, s, t, s_normed: bool = False):
        """
        Compute the mean.

        Args:
            x: [3] the GCRS position of antenna
            s: [3] the GCRS direction of geodesic
            t: the time
            s_normed: whether direction is normed

        Returns:
            the mean TEC
        """
        x_bottom, x_top = self.project_geodesic(x, s, s_normed=s_normed)
        # Apply frozen flow, to find point in reference field
        x_bottom = self.apply_frozen_flow(x_bottom, t)
        x_top = self.apply_frozen_flow(x_top, t)
        s = x_top - x_bottom
        intersection_length = jnp.sqrt(jnp.sum(jnp.square(s)))
        return intersection_length * self.fed_mu

    def compute_kernel(self, x1, s1, t1, x2, s2, t2, resolution: int, s_normed: bool = False):
        @partial(
            multi_vmap,
            in_mapping="[D,T,A,3],[D,T,A,3],[D,T,A],[D',T',A',3],[D',T',A',3],[D',T',A']",
            out_mapping="[D,T,A,D',T',A']"
        )
        def get_kernel(x1, s1_hat, t1, x2, s2_hat, t2):
            return self._compute_kernel(x1, s1_hat, t1, x2, s2_hat, t2, s_normed=s_normed, resolution=resolution)

        K = get_kernel(x1, s1, t1, x2, s2, t2)  # [D,T,A,D',T',A']
        return K

    def compute_mean(self, x, s, t, s_normed: bool = False):
        @partial(
            multi_vmap,
            in_mapping="[D,T,A,3],[D,T,A,3],[D,T,A]",
            out_mapping="[D,T,A]"
        )
        def get_mean(x, s_hat, t):
            return self._compute_mean(x, s_hat, t, s_normed=s_normed)

        mean = get_mean(x, s, t)  # [D, T, A]
        return mean

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
    def flatten(cls, this: "IonosphereLayer") -> Tuple[List[Any], Tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        return [
            this.length_scale, this.longitude_pole, this.latitude_pole, this.bottom_velocity, this.radial_velocity,
            this.x0_radius, this.bottom, this.width, this.fed_mu, this.fed_sigma
        ], (
            this.method,
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "IonosphereLayer":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        length_scale, longitude_pole, latitude_pole, bottom_velocity, radial_velocity, x0_radius, bottom, width, fed_mu, fed_sigma = children
        (method,) = aux_data
        return IonosphereLayer(
            length_scale=length_scale, longitude_pole=longitude_pole, latitude_pole=latitude_pole,
            bottom_velocity=bottom_velocity, radial_velocity=radial_velocity, x0_radius=x0_radius,
            bottom=bottom, width=width, fed_mu=fed_mu, fed_sigma=fed_sigma, method=method,
            skip_post_init=True
        )


IonosphereLayer.register_pytree()


@dataclasses.dataclass(eq=False)
class IonosphereMultiLayer(AbstractIonosphereLayer):
    layers: List[IonosphereLayer]

    def compute_mean(self, x, s, t, s_normed: bool = False):
        means = []
        for layer in self.layers:
            means.append(layer.compute_mean(x, s, t, s_normed))
        return sum(means[1:], start=means[0])

    def compute_kernel(self, x1, s1, t1, x2, s2, t2, resolution: int, s_normed: bool = False):
        kernels = []
        for layer in self.layers:
            kernels.append(layer.compute_kernel(x1, s1, t1, x2, s2, t2, resolution, s_normed))
        return sum(kernels[1:], start=kernels[0])

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
    def flatten(cls, this: "IonosphereMultiLayer") -> Tuple[List[Any], Tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        return [
            this.layers
        ], (

        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]) -> "IonosphereMultiLayer":
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        [layers] = children
        return IonosphereMultiLayer(
            layers=layers
        )


IonosphereMultiLayer.register_pytree()


def calibrate_resolution(layer: IonosphereLayer, max_sep, max_angle, target_rtol=0.01):
    """
    Get a resolution value that bounds error.

    Args:
        max_sep: the max sep
        max_angle: the max angular sep
        target_rtol: the target rtol

    Returns:
        a resolution by brute force search
    """
    u_sep = jnp.linspace(0., max_sep, 10)
    u_angle = jnp.linspace(0., max_angle, 10)
    U_sep, U_angle = jnp.meshgrid(u_sep, u_angle, indexing='ij')

    def eval_baseline(resolution):
        def inner_eval(u_sep, u_angle):
            x1 = jnp.array([0., 0., 0.])
            x2 = jnp.array([u_sep, 0., 0.])
            s1 = jnp.array([0., 0., 1.])
            s2 = jnp.array([0., jnp.sin(u_angle), jnp.cos(u_angle)])

            integral = layer.numerical_double_integral(x1, x2, s1, s2,
                                                       0., layer.width, 0., layer.width,
                                                       resolution=resolution)
            return integral

        return jax.vmap(inner_eval)(U_sep.flatten(), U_angle.flatten())

    def eval(baseline, resolution):
        def inner_eval(u_sep, u_angle):
            x1 = jnp.array([0., 0., 0.])
            x2 = jnp.array([u_sep, 0., 0.])
            s1 = jnp.array([0., 0., 1.])
            s2 = jnp.array([0., jnp.sin(u_angle), jnp.cos(u_angle)])
            integral1 = layer.double_tomographic_integral(x1, x2, s1, s2,
                                                          0., layer.width, 0., layer.width,
                                                          resolution=resolution)
            return integral1

        error = (jax.vmap(inner_eval)(U_sep.flatten(), U_angle.flatten()) - baseline) / baseline
        return jnp.max(jnp.abs(error))
        # return jnp.mean(error) + jnp.sqrt(jnp.mean(error ** 2))  # 1sigma conf

    eval_jit = jax.jit(eval, static_argnames=['resolution'])

    baseline = eval_baseline(2000)

    lower_res, upper_res = [1, 1000]
    lower_bound = eval_jit(baseline, lower_res)
    upper_bound = eval_jit(baseline, upper_res)
    print(f"Lower bound: {lower_res} -> {lower_bound}, Upper bound: {upper_res} -> {upper_bound}")
    # bisect discretely until we find the resolution
    _res = [lower_res, upper_res]
    _evals = [lower_bound, upper_bound]
    while upper_res - lower_res > 1:
        resolution = (upper_res + lower_res) // 2
        rel_error = eval_jit(baseline, resolution)
        _res.append(resolution)
        _evals.append(rel_error)
        if rel_error > target_rtol:  # replace lower bound
            lower_res = resolution
            lower_bound = rel_error
        else:
            upper_res = resolution
            upper_bound = rel_error
        print(f"Lower bound: {lower_res} -> {lower_bound}, Upper bound: {upper_res} -> {upper_bound}")

    plt.scatter(_res, _evals)
    plt.xlabel('Resolution')
    plt.ylabel('Log10 Relative Error')
    plt.title(f"Rel error for max_sep={max_sep}, max_angle={max_angle}")
    plt.show()
    return upper_res


def construct_eval_interp_struct(antennas: ac.EarthLocation, ref_location: ac.EarthLocation, times: at.Time,
                                 ref_time: at.Time, directions: ac.ICRS, dt=1 * au.min):
    x0_radius = np.linalg.norm(ref_location.get_gcrs(ref_time).cartesian.xyz.to('km')).value
    T = int((times.max() - times.min()) / dt) + 1
    model_times = times.min() + np.arange(0., T) * au.min
    model_times_jax = time_to_jnp(model_times, ref_time)
    times_jax = time_to_jnp(times, ref_time)
    antennas_gcrs = InterpolatedArray(
        model_times_jax,
        antennas.get_gcrs(model_times[:, None]).cartesian.xyz.to('km').value.transpose((1, 2, 0)),
        axis=0,
        regular_grid=True,
        check_spacing=True
    )
    directions_gcrs = InterpolatedArray(
        model_times_jax,
        directions.transform_to(ref_location.get_gcrs(model_times[:, None])).cartesian.xyz.value.transpose(
            (1, 2, 0)),
        axis=0,
        regular_grid=True,
        check_spacing=True
    )
    return x0_radius, times_jax, antennas_gcrs, directions_gcrs


def evolve_antennas(x0_gcrs, dt):
    """
    Evolve antennas in GCRS by a time step.

    Args:
        x0_gcrs: [3] the initial position in GCRS
        dt: the amount of time

    Returns:
        [3] the evolved position in GCRS
    """
    omega = 7.292115315411851e-05  # ~ 2 * np.pi / ((23 + 56 / 60) * 3600)
    alpha_pole = 0.015949670685007602
    delta_pole = 1.5683471107500062
    return efficient_rodriges_rotation(
        x_proj=x0_gcrs,
        rdot=0.,
        omega=omega,
        dt=dt,
        alpha_pole=alpha_pole,
        delta_pole=delta_pole
    )


def efficient_rodriges_rotation(x_proj, rdot, omega, dt, alpha_pole, delta_pole):
    """
    Efficiently rotate a 3D vector using Rodriges' rotation formula about a pole.

    Args:
        x_proj: [3] the 3D vector to rotate
        rdot: the radial velocity
        omega: the angular velocity in rad/s
        dt: the time step
        alpha_pole: the longitude of the pole
        delta_pole: the latitude of the pole

    Returns:
        [3] the x_proj vector rotated
    """
    if np.shape(x_proj) != (3,):
        raise ValueError(f"Expected 3D vector input, got {np.shape(x_proj)}.")
    r_init = jnp.linalg.norm(x_proj)
    x_proj /= r_init
    r_final = r_init + rdot * dt

    x, y, z = x_proj[0], x_proj[1], x_proj[2]
    x0 = dt * omega
    x1 = jnp.cos(x0)
    x2 = -x1
    x3 = jnp.cos(alpha_pole)
    x4 = jnp.cos(delta_pole)
    x5 = x4 ** 2
    x6 = x1 - 1
    x7 = x5 * x6
    x8 = jnp.sin(alpha_pole)
    x9 = jnp.sin(x0)
    x10 = x8 * x9
    x11 = jnp.sin(delta_pole)
    x12 = x11 * x6
    x13 = x12 * x3
    x14 = x4 * z
    x15 = x11 * x9
    x16 = x3 * x7 * x8
    x17 = x3 * x9
    x18 = x12 * x8
    x_out = -x * (x2 + x3 ** 2 * x7) - x14 * (-x10 + x13) - y * (x15 + x16)
    y_out = -x * (-x15 + x16) - x14 * (x17 + x18) - y * (x2 + x7 * x8 ** 2)
    z_out = -x * x4 * (x10 + x13) - x4 * y * (-x17 + x18) + z * (x1 * x5 - x5 + 1)
    return r_final * jnp.stack([x_out, y_out, z_out])
