import dataclasses
import pickle
import time
import warnings
from abc import abstractmethod, ABC
from collections import deque
from functools import partial
from typing import Tuple, List, Any

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import matplotlib.dates as mdates
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from matplotlib.widgets import Slider
from typing_extensions import NamedTuple

from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.astropy_utils import create_spherical_earth_grid, mean_itrs
from dsa2000_common.common.interp_utils import InterpolatedArray
from dsa2000_common.common.jax_utils import multi_vmap
from dsa2000_common.common.quantity_utils import time_to_jnp, quantity_to_jnp
from dsa2000_common.common.serialise_utils import SerialisableBaseModel
from dsa2000_common.delay_models.uvw_utils import perley_lmn_from_icrs
from dsa2000_common.gain_models.base_spherical_interpolator import phi_theta_from_lmn, build_spherical_interpolator, \
    BaseSphericalInterpolatorGainModel

tfpd = tfp.distributions

TEC_CONV = -8.4479745 * au.rad * au.MHz  # rad * MHz / mTECU


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
    x2 = jnp.sum(x * x)
    x_norm = jnp.sqrt(x2)
    two_x_norm = x_norm + x_norm
    xs = jnp.sum(x * s)
    xs2 = xs * xs
    x02_minus_x2 = x0_radius * x0_radius - x2
    _tmp = xs2 + x02_minus_x2
    dr = bottom
    s_bottom = -xs + jnp.sqrt(_tmp + two_x_norm * dr + dr * dr)
    dr = bottom + width
    s_top = -xs + jnp.sqrt(_tmp + two_x_norm * dr + dr * dr)
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


class ConditionalCache(NamedTuple):
    """
    A cache for kernel calculations, which can be reused if system considered not to change much.
    """
    K_xx: FloatArray | None  # [N, N]
    mean_x: FloatArray | None  # [N]
    K_ss: FloatArray | None  # [M, M]
    mean_s: FloatArray | None  # [M]
    K_sx: FloatArray | None  # [M, N]


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

    def compute_conditional_tec_kernel(self, antennas_gcrs: InterpolatedArray,
                                       times: FloatArray, directions_gcrs: InterpolatedArray,
                                       antennas_gcrs_other: InterpolatedArray,
                                       times_other: FloatArray, directions_gcrs_other: InterpolatedArray,
                                       resolution: int):
        x1, s1, t1 = self.compute_geodesic_coords(antennas_gcrs, times, directions_gcrs)
        x2, s2, t2 = self.compute_geodesic_coords(antennas_gcrs_other, times_other, directions_gcrs_other)
        K = self.compute_kernel(
            x1, s1, t1, x2, s2, t2, s_normed=True, resolution=resolution
        )
        return K

    def compute_conditional_dtec_kernel(self,
                                        reference_antenna_gcrs: InterpolatedArray,
                                        antennas_gcrs: InterpolatedArray,
                                        times: FloatArray,
                                        directions_gcrs: InterpolatedArray,
                                        antennas_gcrs_other: InterpolatedArray,
                                        times_other: FloatArray, directions_gcrs_other: InterpolatedArray,
                                        resolution: int = 27):
        x1, s1, t1 = self.compute_geodesic_coords(antennas_gcrs, times, directions_gcrs)
        x2, s2, t2 = self.compute_geodesic_coords(antennas_gcrs_other, times_other, directions_gcrs_other)
        K11 = self.compute_kernel(
            x1, s1, t1, x2, s2, t2, s_normed=True, resolution=resolution
        )
        x0, s0, t0 = self.compute_geodesic_coords(reference_antenna_gcrs, times, directions_gcrs)
        x0_, s0_, t0_ = self.compute_geodesic_coords(reference_antenna_gcrs, times_other, directions_gcrs_other)
        K01 = self.compute_kernel(
            x0, s0, t0, x2, s2, t2, s_normed=True, resolution=resolution
        )
        K10 = self.compute_kernel(
            x1, s1, t1, x0_, s0_, t0_, s_normed=True, resolution=resolution
        )
        K00 = self.compute_kernel(
            x0, s0, t0, x0_, s0_, t0_, s_normed=True, resolution=resolution
        )
        K = K11 + K00 - (K01 + K10)
        return K

    def _sample_mvn(self, key, mean, K, jitter_mtec=0.5):
        # Efficient add to diagonal
        diag_idxs = jnp.diag_indices(K.shape[0])
        K = K.at[diag_idxs].add(jitter_mtec ** 2)
        sample = tfpd.MultivariateNormalTriL(
            loc=mean,
            scale_tril=jnp.linalg.cholesky(K)
        ).sample(
            seed=key
        )
        return sample

    def _marginal_sample(self, key, K, mean, jitter_mtec=0.5):
        D, T, A = np.shape(mean)
        K = jax.lax.reshape(K, (D * T * A, D * T * A))
        mean = jax.lax.reshape(mean, [D * T * A])
        sample = self._sample_mvn(key, mean, K, jitter_mtec)
        sample = jax.lax.reshape(sample, [D, T, A])
        return sample

    def _conditional_sample(self, key, K_ss, mean_s, K_xx, mean_x, K_sx, y_other, jitter_mtec=0.5):
        # reshape
        D, T, A = np.shape(mean_s)
        D_, T_, A_ = np.shape(mean_x)

        K_ss = jax.lax.reshape(K_ss, (D * T * A, D * T * A))
        mean_s = jax.lax.reshape(mean_s, [D * T * A])
        K_xx = jax.lax.reshape(K_xx, (D_ * T_ * A_, D_ * T_ * A_))
        mean_x = jax.lax.reshape(mean_x, [D_ * T_ * A_])
        K_sx = jax.lax.reshape(K_sx, (D * T * A, D_ * T_ * A_))
        y_other = jax.lax.reshape(y_other, (D_ * T_ * A_,))

        # more efficiently
        diag_idxs = jnp.diag_indices(K_xx.shape[0])
        K_xx = K_xx.at[diag_idxs].add(jitter_mtec ** 2)

        # Compute the Cholesky factorization: K_xx = L L^T.
        L = jnp.linalg.cholesky(K_xx)

        # Solve for Z = L^{-1} K_sx.T using a triangular solve.
        Z = jax.scipy.linalg.solve_triangular(L, K_sx.T, lower=True)

        # K_ss - K_sx K_xx^{-1} K_sx.T = K_ss - Z^T Z
        K = K_ss - Z.T @ Z

        # For the  mean: m_s + K_sx K_xx^{-1} (y_x - m_x) = m_s + Z^T L^{-1} (y_x - m_x)
        v = jax.scipy.linalg.solve_triangular(L, y_other - mean_x, lower=True)
        mean = mean_s + Z.T @ v

        sample = self._sample_mvn(key, mean, K, jitter_mtec)
        sample = jax.lax.reshape(sample, [D, T, A])
        return sample

    def _conditional_predict(self, K_ss, mean_s, K_xx, mean_x, K_sx, y_other, jitter_mtec=0.5):
        # reshape
        D, T, A = np.shape(mean_s)
        D_, T_, A_ = np.shape(mean_x)

        K_ss = jax.lax.reshape(K_ss, (D * T * A, D * T * A))
        mean_s = jax.lax.reshape(mean_s, [D * T * A])
        K_xx = jax.lax.reshape(K_xx, (D_ * T_ * A_, D_ * T_ * A_))
        mean_x = jax.lax.reshape(mean_x, [D_ * T_ * A_])
        K_sx = jax.lax.reshape(K_sx, (D * T * A, D_ * T_ * A_))
        y_other = jax.lax.reshape(y_other, (D_ * T_ * A_,))

        # more efficiently
        diag_idxs = jnp.diag_indices(K_xx.shape[0])
        K_xx = K_xx.at[diag_idxs].add(jitter_mtec ** 2)

        # Compute the Cholesky factorization: K_xx = L L^T.
        L = jnp.linalg.cholesky(K_xx)

        # Solve for Z = L^{-1} K_sx.T using a triangular solve.
        Z = jax.scipy.linalg.solve_triangular(L, K_sx.T, lower=True)

        # Predictive variance:
        # K_ss - K_sx @ K_xx^{-1} @ K_sx.T = K_ss - Z^T Z; extract the diagonal as sum of squares.
        pred_var = jnp.diag(K_ss) - jnp.sum(Z ** 2, axis=0)

        # For the predictive mean
        v = jax.scipy.linalg.solve_triangular(L, y_other - mean_x, lower=True)
        mean = mean_s + Z.T @ v

        mean = jax.lax.reshape(mean, [D, T, A])
        pred_var = jax.lax.reshape(pred_var, [D, T, A])
        return mean, pred_var

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
        return self._marginal_sample(key, K, mean, jitter_mtec)

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
        return self._marginal_sample(key, K, mean, jitter_mtec)

    def sample_conditional_tec(
            self,
            key,
            antennas_gcrs: InterpolatedArray,
            times: FloatArray,
            directions_gcrs: InterpolatedArray,
            antennas_gcrs_other: InterpolatedArray,
            times_other: FloatArray,
            directions_gcrs_other: InterpolatedArray,
            tec_other: FloatArray,
            jitter_mtec=0.5, resolution: int = 27,
            cache: ConditionalCache | None = None
    ):
        """
        Sample ionosphere TEC.

        Args:
            key: PRNGKey
            antennas_gcrs: [A] antennas (time) -> [A, 3]
            times: [T] times
            directions_gcrs: [D] directions (time) -> [D, 3]
            times_other: [T'] times
            directions_gcrs_other: [D'] directions (time) -> [D', 3]
            tec_other: [D', T', A'] TEC
            jitter_mtec: how much diagonal jitter, equivalent to adding white noise.
            resolution: how many resolution elements to use, default tuned to DSA2000

        Returns:
            [D, T, A] shaped array of DTEC or TEC
        """
        K_xx, mean_x = self.compute_tec_process_params(
            antennas_gcrs_other, times_other, directions_gcrs_other, resolution
        )
        K_ss, mean_s = self.compute_tec_process_params(antennas_gcrs, times, directions_gcrs, resolution)
        K_sx = self.compute_conditional_tec_kernel(
            antennas_gcrs, times, directions_gcrs, antennas_gcrs_other, times_other, directions_gcrs_other,
            resolution
        )
        if cache is None:
            cache = ConditionalCache(K_xx=None, mean_x=None, K_ss=None, mean_s=None, K_sx=None)
        K_xx = cache.K_xx if cache.K_xx is not None and np.shape(cache.K_xx) == np.shape(K_xx) else K_xx
        mean_x = cache.mean_x if cache.mean_x is not None and np.shape(cache.mean_x) == np.shape(mean_x) else mean_x
        K_ss = cache.K_ss if cache.K_ss is not None and np.shape(cache.K_ss) == np.shape(K_ss) else K_ss
        mean_s = cache.mean_s if cache.mean_s is not None and np.shape(cache.mean_s) == np.shape(mean_s) else mean_s
        K_sx = cache.K_sx if cache.K_sx is not None and np.shape(cache.K_sx) == np.shape(K_sx) else K_sx
        cache = ConditionalCache(
            K_xx=K_xx,
            mean_x=mean_x,
            K_ss=K_ss,
            mean_s=mean_s,
            K_sx=K_sx
        )
        return self._conditional_sample(key, K_ss, mean_s, K_xx, mean_x, K_sx, tec_other, jitter_mtec), cache

    def sample_conditional_dtec(self, key,
                                reference_antenna_gcrs: InterpolatedArray,
                                antennas_gcrs: InterpolatedArray,
                                times: FloatArray,
                                directions_gcrs: InterpolatedArray,
                                antennas_gcrs_other: InterpolatedArray,
                                times_other: FloatArray,
                                directions_gcrs_other: InterpolatedArray,
                                dtec_other: FloatArray,
                                jitter_mtec=0.5, resolution: int = 27,
                                cache: ConditionalCache | None = None):
        """
        Sample ionosphere DTEC.

        Args:
            key: PRNGKey
            reference_antenna_gcrs: the reference antenna (time) -> [1, 3]
            antennas_gcrs: [A] antennas (time) -> [A, 3]
            times: [T] times
            directions_gcrs: [D] directions (time) -> [D, 3]
            antennas_gcrs_other: [A'] antennas (time) -> [A', 3]
            times_other: [T'] times
            directions_gcrs_other: [D'] directions (time) -> [D', 3]
            dtec_other: [D', T', A'] TEC
            jitter_mtec: how much diagonal jitter, equivalent to adding white noise.
            resolution: how many resolution elements to use, default tuned to DSA2000

        Returns:
            [D, T, A] shaped mean and var
        """

        K_xx, mean_x = self.compute_dtec_process_params(
            reference_antenna_gcrs, antennas_gcrs_other, times_other, directions_gcrs_other, resolution
        )
        K_ss, mean_s = self.compute_dtec_process_params(
            reference_antenna_gcrs, antennas_gcrs, times, directions_gcrs, resolution
        )
        K_sx = self.compute_conditional_dtec_kernel(
            reference_antenna_gcrs, antennas_gcrs, times, directions_gcrs, antennas_gcrs_other, times_other,
            directions_gcrs_other, resolution
        )
        if cache is None:
            cache = ConditionalCache(K_xx=None, mean_x=None, K_ss=None, mean_s=None, K_sx=None)
        K_xx = cache.K_xx if cache.K_xx is not None and np.shape(cache.K_xx) == np.shape(K_xx) else K_xx
        mean_x = cache.mean_x if cache.mean_x is not None and np.shape(cache.mean_x) == np.shape(mean_x) else mean_x
        K_ss = cache.K_ss if cache.K_ss is not None and np.shape(cache.K_ss) == np.shape(K_ss) else K_ss
        mean_s = cache.mean_s if cache.mean_s is not None and np.shape(cache.mean_s) == np.shape(mean_s) else mean_s
        K_sx = cache.K_sx if cache.K_sx is not None and np.shape(cache.K_sx) == np.shape(K_sx) else K_sx
        cache = ConditionalCache(
            K_xx=K_xx,
            mean_x=mean_x,
            K_ss=K_ss,
            mean_s=mean_s,
            K_sx=K_sx
        )
        return self._conditional_sample(key, K_ss, mean_s, K_xx, mean_x, K_sx, dtec_other, jitter_mtec), cache

    def predict_conditional_tec(
            self,
            antennas_gcrs: InterpolatedArray,
            times: FloatArray,
            directions_gcrs: InterpolatedArray,
            antennas_gcrs_other: InterpolatedArray,
            times_other: FloatArray,
            directions_gcrs_other: InterpolatedArray,
            tec_other: FloatArray,
            jitter_mtec=0.5, resolution: int = 27,
            cache: ConditionalCache | None = None):
        """
        Predict ionosphere TEC.

        Args:
            antennas_gcrs: [A] antennas (time) -> [A, 3]
            times: [T] times
            directions_gcrs: [D] directions (time) -> [D, 3]
            antennas_gcrs_other: [A'] antennas (time) -> [A', 3]
            times_other: [T'] times
            directions_gcrs_other: [D'] directions (time) -> [D', 3]
            tec_other: [D', T', A'] TEC
            jitter_mtec: how much diagonal jitter, equivalent to adding white noise.
            resolution: how many resolution elements to use, default tuned to DSA2000

        Returns:
            [D, T, A] shaped mean and var
        """
        K_xx, mean_x = self.compute_tec_process_params(
            antennas_gcrs_other, times_other, directions_gcrs_other, resolution
        )
        K_ss, mean_s = self.compute_tec_process_params(antennas_gcrs, times, directions_gcrs, resolution)
        K_sx = self.compute_conditional_tec_kernel(
            antennas_gcrs, times, directions_gcrs, antennas_gcrs_other, times_other, directions_gcrs_other,
            resolution
        )
        if cache is None:
            cache = ConditionalCache(K_xx=None, mean_x=None, K_ss=None, mean_s=None, K_sx=None)
        K_xx = cache.K_xx if cache.K_xx is not None and np.shape(cache.K_xx) == np.shape(K_xx) else K_xx
        mean_x = cache.mean_x if cache.mean_x is not None and np.shape(cache.mean_x) == np.shape(mean_x) else mean_x
        K_ss = cache.K_ss if cache.K_ss is not None and np.shape(cache.K_ss) == np.shape(K_ss) else K_ss
        mean_s = cache.mean_s if cache.mean_s is not None and np.shape(cache.mean_s) == np.shape(mean_s) else mean_s
        K_sx = cache.K_sx if cache.K_sx is not None and np.shape(cache.K_sx) == np.shape(K_sx) else K_sx
        cache = ConditionalCache(
            K_xx=K_xx,
            mean_x=mean_x,
            K_ss=K_ss,
            mean_s=mean_s,
            K_sx=K_sx
        )
        return self._conditional_predict(K_ss, mean_s, K_xx, mean_x, K_sx, tec_other, jitter_mtec), cache

    def predict_conditional_dtec(self,
                                 reference_antenna_gcrs: InterpolatedArray,
                                 antennas_gcrs: InterpolatedArray,
                                 times: FloatArray,
                                 directions_gcrs: InterpolatedArray,
                                 antennas_gcrs_other: InterpolatedArray,
                                 times_other: FloatArray,
                                 directions_gcrs_other: InterpolatedArray,
                                 dtec_other: FloatArray,
                                 jitter_mtec=0.5, resolution: int = 27,
                                 cache: ConditionalCache | None = None):
        """
        Predict ionosphere DTEC.

        Args:
            reference_antenna_gcrs: the reference antenna (time) -> [1, 3]
            antennas_gcrs: [A] antennas (time) -> [A, 3]
            times: [T] times
            directions_gcrs: [D] directions (time) -> [D, 3]
            antennas_gcrs_other: [A'] antennas (time) -> [A', 3]
            times_other: [T'] times
            directions_gcrs_other: [D'] directions (time) -> [D', 3]
            dtec_other: [D', T', A'] TEC
            jitter_mtec: how much diagonal jitter, equivalent to adding white noise.
            resolution: how many resolution elements to use, default tuned to DSA2000

        Returns:
            [D, T, A] shaped mean and var
        """
        K_xx, mean_x = self.compute_dtec_process_params(
            reference_antenna_gcrs, antennas_gcrs_other, times_other, directions_gcrs_other, resolution
        )
        K_ss, mean_s = self.compute_dtec_process_params(
            reference_antenna_gcrs, antennas_gcrs, times, directions_gcrs, resolution
        )
        K_sx = self.compute_conditional_dtec_kernel(
            reference_antenna_gcrs, antennas_gcrs, times, directions_gcrs, antennas_gcrs_other, times_other,
            directions_gcrs_other, resolution
        )
        if cache is None:
            cache = ConditionalCache(K_xx=None, mean_x=None, K_ss=None, mean_s=None, K_sx=None)
        K_xx = cache.K_xx if cache.K_xx is not None and np.shape(cache.K_xx) == np.shape(K_xx) else K_xx
        mean_x = cache.mean_x if cache.mean_x is not None and np.shape(cache.mean_x) == np.shape(mean_x) else mean_x
        K_ss = cache.K_ss if cache.K_ss is not None and np.shape(cache.K_ss) == np.shape(K_ss) else K_ss
        mean_s = cache.mean_s if cache.mean_s is not None and np.shape(cache.mean_s) == np.shape(mean_s) else mean_s
        K_sx = cache.K_sx if cache.K_sx is not None and np.shape(cache.K_sx) == np.shape(K_sx) else K_sx
        cache = ConditionalCache(
            K_xx=K_xx,
            mean_x=mean_x,
            K_ss=K_ss,
            mean_s=mean_s,
            K_sx=K_sx
        )
        return self._conditional_predict(K_ss, mean_s, K_xx, mean_x, K_sx, dtec_other, jitter_mtec), cache


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

    def __post_init__(self):
        if len(self.layers) == 0:
            raise ValueError('Got no layers.')

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


def compute_x0_radius(ref_location: ac.EarthLocation, ref_time: at.Time):
    """
    Computes the radius of a reference location in GCRS. This is the location which the ionosphere height is relative to.

    Args:
        ref_location: the location
        ref_time: a specific time

    Returns:
        a radius in km
    """
    return np.linalg.norm(ref_location.get_gcrs(ref_time).cartesian.xyz.to('km')).value


def construct_eval_interp_struct(antennas: ac.EarthLocation, ref_location: ac.EarthLocation, times: at.Time,
                                 ref_time: at.Time, directions: ac.ICRS, model_times: at.Time):
    x0_radius = compute_x0_radius(ref_location, ref_time)
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


def create_model_antennas(antennas: ac.EarthLocation, spatial_resolution: au.Quantity) -> ac.EarthLocation:
    """
    Create model antennas at a given spatial resolution.

    Args:
        antennas: [n] the input antennas
        spatial_resolution: a spatial resolution

    Returns:
        [m] model antennas, ideally m<n
    """
    antennas_itrs = antennas.get_itrs()
    antennas_itrs_xyz = antennas_itrs.cartesian.xyz.T
    array_center = mean_itrs(antennas_itrs).earth_location
    radius = np.max(np.linalg.norm(
        antennas_itrs_xyz - array_center.get_itrs().cartesian.xyz,
        axis=-1
    ))
    print(f"Array radius: {radius}")

    model_antennas = create_spherical_earth_grid(
        center=array_center,
        radius=radius,
        dr=spatial_resolution
    )

    # filter out model antennas that are too far from any actual antenna
    def keep(model_antenna: ac.EarthLocation):
        dist = np.linalg.norm(
            model_antenna.get_itrs().cartesian.xyz - antennas_itrs_xyz,
            axis=-1
        )
        return np.any(dist < spatial_resolution)

    # List of EarthLocation
    model_antennas = list(filter(keep, model_antennas))
    # Via ITRS then back to EarthLocation
    model_antennas: ac.EarthLocation = ac.concatenate(list(map(lambda x: x.get_itrs(), model_antennas))).earth_location
    if len(model_antennas) >= len(antennas):
        print(f"Spatial resolution: {spatial_resolution} ==> more model antennas than actual antennas.")
    return model_antennas


@partial(jax.jit, static_argnames=['do_tec'])
def sample_dtec(key, ionosphere: IonosphereMultiLayer, reference_antenna_gcrs, antennas_gcrs,
                directions_gcrs, times, jitter, do_tec: bool = False):
    if do_tec:
        return ionosphere.sample_tec(
            key=key,
            antennas_gcrs=antennas_gcrs,
            directions_gcrs=directions_gcrs,
            times=times,
            jitter_mtec=jitter,
        )  # [D, T, A]
    else:
        return ionosphere.sample_dtec(
            key=key,
            reference_antenna_gcrs=reference_antenna_gcrs,
            antennas_gcrs=antennas_gcrs,
            directions_gcrs=directions_gcrs,
            times=times,
            jitter_mtec=jitter,
        )  # [D, T, A]


@partial(jax.jit, static_argnames=['do_tec', 'clear_s'])
def sample_conditional_dtec(key, ionosphere: IonosphereMultiLayer, reference_antenna_gcrs, antennas_gcrs,
                            directions_gcrs, times, antennas_gcrs_other,
                            directions_gcrs_other, times_other, dtec_other, jitter, cache=None,
                            do_tec: bool = False, clear_s: bool = False):
    if do_tec:
        samples, cache = ionosphere.sample_conditional_tec(
            key=key,
            antennas_gcrs=antennas_gcrs,
            times=times,
            directions_gcrs=directions_gcrs,
            antennas_gcrs_other=antennas_gcrs_other,
            times_other=times_other,
            directions_gcrs_other=directions_gcrs_other,
            tec_other=dtec_other,
            jitter_mtec=jitter,
            cache=cache
        )  # [D, T, A]
    else:
        samples, cache = ionosphere.sample_conditional_dtec(
            key=key,
            reference_antenna_gcrs=reference_antenna_gcrs,
            antennas_gcrs=antennas_gcrs,
            times=times,
            directions_gcrs=directions_gcrs,
            antennas_gcrs_other=antennas_gcrs_other,
            times_other=times_other,
            directions_gcrs_other=directions_gcrs_other,
            dtec_other=dtec_other,
            jitter_mtec=jitter,
            cache=cache
        )  # [D, T, A]
    if clear_s:
        cache = cache._replace(K_ss=None, mean_s=None, K_sx=None)
    return samples, cache


@partial(jax.jit, static_argnames=['do_tec', 'clear_s'])
def predict_conditional_dtec(
        ionosphere: IonosphereMultiLayer, reference_antenna_gcrs, antennas_gcrs,
        directions_gcrs, times, model_antennas_gcrs, model_directions_gcrs, model_times, dtec_other, cache=None,
        do_tec: bool = False, clear_s: bool = False
):
    if do_tec:
        (pred_mean, _), cache = ionosphere.predict_conditional_tec(
            antennas_gcrs=antennas_gcrs,
            times=times,
            directions_gcrs=directions_gcrs,
            antennas_gcrs_other=model_antennas_gcrs,
            times_other=model_times,
            directions_gcrs_other=model_directions_gcrs,
            tec_other=dtec_other,
            cache=cache
        )  # [D, T, A]
    else:

        (pred_mean, _), cache = ionosphere.predict_conditional_dtec(
            reference_antenna_gcrs=reference_antenna_gcrs,
            antennas_gcrs=antennas_gcrs,
            times=times,
            directions_gcrs=directions_gcrs,
            antennas_gcrs_other=model_antennas_gcrs,
            times_other=model_times,
            directions_gcrs_other=model_directions_gcrs,
            dtec_other=dtec_other,
            cache=cache
        )  # [D, T, A]
    if clear_s:
        cache = cache._replace(K_ss=None, mean_s=None, K_sx=None)
    return pred_mean, cache


def build_ionosphere_gain_model(
        key,
        ionosphere: AbstractIonosphereLayer,
        model_freqs: au.Quantity,
        antennas: ac.EarthLocation,
        ref_location: ac.EarthLocation,
        times: at.Time,
        ref_time: at.Time,
        directions: ac.ICRS,
        phase_centre: ac.ICRS,
        full_stokes: bool = True,
        spatial_resolution: au.Quantity = 2 * au.km,
        predict_batch_size: int = 1,
        resolution: int = 257,
        save_file: str | None = None
) -> BaseSphericalInterpolatorGainModel:
    """
    Simulate ionosphere gain model.

    Args:
        model_freqs: the frequencies to evaluate at.
        antennas: [A] the antennas
        ref_location: the reference location for simulation (need not be an antenna)
        times: [T] the times
        ref_time: the ref time things are relative to
        directions: [D] the directions
        phase_centre: the pointing and phase tracking center
        save_file: save dtec to file to explore later

    Returns:
        a gain model (note directions far from a `direction` are ill-defined).
    """

    dtec_samples = simulate_ionosphere(key, ionosphere, ref_location, antennas, directions, times, ref_time,
                                       spatial_resolution, predict_batch_size, save_file)
    # explore_dtec(dtec_samples, antennas, directions, times)

    phase_factor = quantity_to_jnp(TEC_CONV / model_freqs, 'rad')  # [F]
    phase = dtec_samples[..., None] * phase_factor  # [T, D, A, F]
    model_gains = jax.lax.complex(jnp.cos(phase), jnp.sin(phase))  # [T, D, A, F]
    if full_stokes:
        model_gains = model_gains[..., None, None] * jnp.eye(2)  # [T, D, A, F, 2, 2]
    model_gains = model_gains * au.dimensionless_unscaled

    # The samples are now in past_sample
    # Get gains into [num_model_times, num_model_dir, num_ant, num_model_freqs, 2, 2]
    # Will need to get the lmn of the directions

    model_phi, model_theta = phi_theta_from_lmn(
        *perley_lmn_from_icrs(
            alpha=directions.ra.rad,
            dec=directions.dec.rad,
            alpha0=phase_centre.ra.rad,
            dec0=phase_centre.dec.rad
        )
    )  # [num_model_dir, 3]
    if np.any(~np.isfinite(model_phi)) or np.any(~np.isfinite(model_theta)):
        select = np.where(np.isnan(model_phi) | np.any(np.isnan(model_theta)))
        directions_bad = directions[select]
        model_phi_bad = model_phi[select]
        model_theta_bad = model_theta[select]
        l, m, n = perley_lmn_from_icrs(
            alpha=directions.ra.rad,
            dec=directions.dec.rad,
            alpha0=phase_centre.ra.rad,
            dec0=phase_centre.dec.rad
        )
        l_bad = l[select]
        m_bad = m[select]
        n_bad = n[select]
        joint_bad = list(
            zip(directions_bad.ra.rad, directions_bad.dec.rad, model_phi_bad, model_theta_bad, l_bad, m_bad, n_bad))

        raise ValueError(
            f"Got nans in model phi / theta for phase center {phase_centre} where (ra,dec,phi,theta,l,m,n)=\n{joint_bad}")

    model_phi = model_phi * au.rad
    model_theta = model_theta * au.rad

    return build_spherical_interpolator(
        antennas=antennas,
        model_times=times,
        ref_time=ref_time,
        model_phi=model_phi,
        model_theta=model_theta,
        model_freqs=model_freqs,
        model_gains=model_gains,
        tile_antennas=False,
        regrid_num_neighbours=1,
        resolution=resolution
    )


def simulate_ionosphere(key, ionosphere, ref_location, antennas, directions, times, ref_time, spatial_resolution,
                        predict_batch_size, save_file, do_tec: bool = False):
    """
    Simulate (D)TEC and save to file.

    Args:
        key:
        ionosphere:
        ref_location:
        antennas:
        directions:
        times:
        ref_time:
        spatial_resolution:
        predict_batch_size:
        save_file:

    Returns:

    """
    # These model times are just for interpolators, not for actual simulation.
    # TODO: once use the evolve_gcrs we can remove these.
    model_dt = au.Quantity(6, 's')
    T = int((times.max() - times.min()) / model_dt) + 1
    model_times = times.min() + np.arange(0., T) * model_dt
    # Get lower resolution grid to sample over.
    if spatial_resolution == 0 * au.km:
        model_antennas = antennas
        do_interp = False
    else:
        model_antennas = create_model_antennas(antennas, spatial_resolution=spatial_resolution)
        do_interp = True
    model_antennas_gcrs = InterpolatedArray(
        time_to_jnp(model_times, ref_time),
        model_antennas.get_gcrs(model_times[:, None]).cartesian.xyz.to('km').value.transpose((1, 2, 0)),
        axis=0,
        regular_grid=True,
        check_spacing=True
    )
    model_directions = directions
    model_directions_gcrs = InterpolatedArray(
        time_to_jnp(model_times, ref_time),
        model_directions.transform_to(ref_location.get_gcrs(model_times[:, None])).cartesian.xyz.value.transpose(
            (1, 2, 0)),
        axis=0,
        regular_grid=True,
        check_spacing=True
    )
    # construct the evaluation and interpolation structure
    x0_radius, times_jax, antennas_gcrs, directions_gcrs = construct_eval_interp_struct(
        antennas, ref_location, times, ref_time, directions, model_times
    )
    reference_antenna_gcrs = antennas_gcrs[0:1]
    past_sample = deque(maxlen=1)
    samples = []
    flow_cache = None
    for t in range(len(times_jax)):
        sample_key, key = jax.random.split(key)
        t0 = time.time()
        if len(past_sample) == 0:
            sample = jax.block_until_ready(
                sample_dtec(
                    key=sample_key,
                    ionosphere=ionosphere,
                    reference_antenna_gcrs=reference_antenna_gcrs,
                    antennas_gcrs=model_antennas_gcrs,
                    directions_gcrs=model_directions_gcrs,
                    times=times_jax[t:t + 1],
                    jitter=0.5,
                    do_tec=do_tec,
                )
            )  # [D, 1, A']
            if np.any(np.isnan(sample)):
                print(f"Got nans in sampling {t}... try again wiith bigger jitter")
                sample = jax.block_until_ready(
                    sample_dtec(
                        key=sample_key,
                        ionosphere=ionosphere,
                        reference_antenna_gcrs=reference_antenna_gcrs,
                        antennas_gcrs=model_antennas_gcrs,
                        directions_gcrs=model_directions_gcrs,
                        times=times_jax[t:t + 1],
                        jitter=1.,
                        do_tec=do_tec,
                    )
                )
                if np.any(np.isnan(sample)):
                    raise ValueError("Getting nans in ionosphere model")
        else:
            n_past = len(past_sample)
            sample, flow_cache = jax.block_until_ready(
                sample_conditional_dtec(
                    key=sample_key,
                    ionosphere=ionosphere,
                    reference_antenna_gcrs=reference_antenna_gcrs,
                    antennas_gcrs=model_antennas_gcrs,
                    times=times_jax[t:t + 1],
                    directions_gcrs=model_directions_gcrs,
                    antennas_gcrs_other=model_antennas_gcrs,
                    directions_gcrs_other=model_directions_gcrs,
                    times_other=times_jax[t - n_past:t],
                    dtec_other=jnp.concatenate(past_sample, axis=1),
                    jitter=0.5,
                    cache=flow_cache,
                    do_tec=do_tec,
                )
            )
            if np.any(np.isnan(sample)):
                print(f"Got nans in sampling {t}... try again wiith bigger jitter")
                sample, flow_cache = jax.block_until_ready(
                    sample_conditional_dtec(
                        key=sample_key,
                        ionosphere=ionosphere,
                        reference_antenna_gcrs=reference_antenna_gcrs,
                        antennas_gcrs=model_antennas_gcrs,
                        times=times_jax[t:t + 1],
                        directions_gcrs=model_directions_gcrs,
                        antennas_gcrs_other=model_antennas_gcrs,
                        directions_gcrs_other=model_directions_gcrs,
                        times_other=times_jax[t - n_past:t],
                        dtec_other=jnp.concatenate(past_sample, axis=1),
                        jitter=1.,
                        cache=flow_cache,
                        do_tec=do_tec,
                    )
                )
                if np.any(np.isnan(sample)):
                    raise ValueError("Getting nans in ionosphere model")
        past_sample.append(sample)
        samples.append(sample)
        t1 = time.time()
        print(f"Conditional sample iteration took {t1 - t0:.2f} seconds")
    if do_interp:
        interp_cache = None
        for t, _ in enumerate(samples):
            t0 = time.time()
            _interp_samples = []
            # predict in batches
            for start_ant_idx in range(0, len(antennas), predict_batch_size):
                # automatically handles remainders
                stop_ant_idx = min(start_ant_idx + predict_batch_size, len(antennas))
                _interp_sample, interp_cache = jax.block_until_ready(
                    predict_conditional_dtec(
                        ionosphere=ionosphere,
                        reference_antenna_gcrs=reference_antenna_gcrs,
                        antennas_gcrs=antennas_gcrs[start_ant_idx:stop_ant_idx],
                        directions_gcrs=directions_gcrs,
                        times=times_jax[t:t + 1],
                        model_antennas_gcrs=model_antennas_gcrs,
                        model_directions_gcrs=model_directions_gcrs,
                        model_times=times_jax[t:t + 1],
                        dtec_other=samples[t],
                        cache=interp_cache,
                        do_tec=do_tec,
                        clear_s=True
                    )
                )  # [D, 1, batch_size]
                _interp_samples.append(_interp_sample)
            samples[t] = jnp.concatenate(_interp_samples, axis=2)  # [D, T, A]
            t1 = time.time()
            print(f"Interp step took {t1 - t0:.2f}s")
    samples = jnp.concatenate(samples, axis=1)  # [D, T, A]
    dtec_samples = samples.transpose((1, 0, 2))  # [T, D, A]
    if save_file is not None:
        result = SimulationResult(
            is_tec=do_tec,
            times=times,
            directions=directions,
            antennas=antennas,
            dtec=np.asarray(dtec_samples)
        )
        with open(save_file, 'w') as f:
            f.write(result.json(indent=2))
        print(f"Saved to {save_file}.")
    return dtec_samples


class SimulationResult(SerialisableBaseModel):
    is_tec: bool
    times: at.Time
    directions: ac.ICRS
    antennas: ac.EarthLocation
    dtec: np.ndarray  # [T, D, A]

    def interactive_explore(self):
        explore_dtec(self.dtec, self.antennas, self.directions, self.times, self.is_tec)


def explore_dtec(dtec: np.ndarray, antennas: ac.EarthLocation, directions: ac.ICRS, times: at.Time, is_tec: bool):
    """
    Plot an interactive plot of dtec.

    Args:
        dtec: [T, D, A] dtec or tec.
        antennas: [A] antennas.
        directions: [D] the directions.
        times: [T] times.
    """
    T, D, A = np.shape(dtec)
    if np.shape(dtec) != (len(times), len(directions), len(antennas)):
        raise ValueError(f"dtec shape {np.shape(dtec)} doesn't match [T, D, A].")

    if is_tec:
        vmin = np.min(dtec)
        vmax = np.max(dtec)
    else:
        vmin = -50
        vmax = 50
    # Convert times to matplotlib date format.
    times_dt = times.to_datetime()  # array of datetime objects
    times_plot = mdates.date2num(times_dt)  # convert to float numbers

    fig = plt.figure(figsize=(10, 12))

    # === Define axes positions manually ===
    # Coordinates: [left, bottom, width, height] in figure fraction.
    #
    # Axis 1: Time series plot (no colorbar)
    ax1 = fig.add_axes([0.1, 0.7, 0.65, 0.2])
    slider_ax_t = fig.add_axes([0.78, 0.7, 0.03, 0.2])  # vertical slider for d_idx

    # Axis 2: Directions scatter (with colorbar)
    ax2 = fig.add_axes([0.1, 0.4, 0.65, 0.2])
    slider_ax_d = fig.add_axes([0.78, 0.4, 0.03, 0.2])  # vertical slider for t_idx
    ax2_colorbar = fig.add_axes([0.82, 0.4, 0.03, 0.2])

    # Axis 3: Antenna scatter (with colorbar)
    ax3 = fig.add_axes([0.1, 0.1, 0.65, 0.2])
    slider_ax_a = fig.add_axes([0.78, 0.1, 0.03, 0.2])  # vertical slider for a_idx
    ax3_colorbar = fig.add_axes([0.82, 0.1, 0.03, 0.2])

    # --- Initialize default indices ---
    t_idx_init = 0
    d_idx_init = 0
    a_idx_init = 0

    # --- Plot for Axis 1: Time Series ---
    # Plot: dtec[:, d_idx, a_idx] vs. times.
    line1, = ax1.plot(times_plot, dtec[:, d_idx_init, a_idx_init])
    ax1.set_ylim(vmin, vmax)
    ax1.set_title("Time Slice[:, d, a]")
    ax1.set_xlabel("Time")
    if is_tec:
        ax1.set_ylabel(r"$\mathrm{TEC}$ (mTECU)")
    else:
        ax1.set_ylabel(r"$\Delta\mathrm{TEC}$ (mTECU)")
    ax1.xaxis_date()
    fig.autofmt_xdate()

    # --- Plot for Axis 2: Directions Scatter ---
    # Plot: dtec[t_idx, :, a_idx] vs. (directions.ra.deg, directions.dec.deg)
    sc2 = ax2.scatter(directions.ra.deg, directions.dec.deg,
                      c=dtec[t_idx_init, :, a_idx_init], cmap='jet', vmin=vmin, vmax=vmax)
    ax2.set_title("Direction Slice[t, :, a]")
    ax2.set_xlabel("RA (deg)")
    ax2.set_ylabel("Dec (deg)")
    if is_tec:
        cb2 = plt.colorbar(sc2, cax=ax2_colorbar, label=r"$\mathrm{TEC}$ (mTECU)")
    else:
        cb2 = plt.colorbar(sc2, cax=ax2_colorbar, label=r"$\Delta\mathrm{TEC}$ (mTECU)")

    # --- Plot for Axis 3: Antenna Scatter ---
    # Plot: dtec[t_idx, d_idx, :] vs. (antennas.lon.deg, antennas.lat.deg)
    sc3 = ax3.scatter(antennas.lon.deg, antennas.lat.deg,
                      c=dtec[t_idx_init, d_idx_init, :], cmap='jet', vmin=vmin, vmax=vmax)
    ax3.set_title("Antenna Slice[t, d, :]")
    ax3.set_xlabel("Longitude (deg)")
    ax3.set_ylabel("Latitude (deg)")
    if is_tec:
        cb3 = plt.colorbar(sc3, cax=ax3_colorbar, label=r"$\mathrm{TEC}$ (mTECU)")
    else:
        cb3 = plt.colorbar(sc3, cax=ax3_colorbar, label=r"$\Delta\mathrm{TEC}$ (mTECU)")

    # --- Create vertical sliders ---
    slider_t = Slider(slider_ax_t, 't_idx', 0, T - 1, valinit=t_idx_init,
                      valstep=1, orientation='vertical')
    slider_d = Slider(slider_ax_d, 'd_idx', 0, D - 1, valinit=d_idx_init,
                      valstep=1, orientation='vertical')
    slider_a = Slider(slider_ax_a, 'a_idx', 0, A - 1, valinit=a_idx_init,
                      valstep=1, orientation='vertical')

    # --- Update function ---
    def update(val):
        # Get current slider values as integers.
        t_idx = int(slider_t.val)
        d_idx = int(slider_d.val)
        a_idx = int(slider_a.val)

        # Axis 1: Time series slice (depends on d_idx and a_idx)
        line1.set_ydata(dtec[:, d_idx, a_idx])

        # Axis 2: Directions scatter (depends on t_idx and a_idx)
        sc2.set_array(dtec[t_idx, :, a_idx])

        # Axis 3: Antenna scatter (depends on t_idx and d_idx)
        sc3.set_array(dtec[t_idx, d_idx, :])

        fig.canvas.draw_idle()

    slider_t.on_changed(update)
    slider_d.on_changed(update)
    slider_a.on_changed(update)

    plt.show()


# simulate ionosphere over a give resolution

def evolve_gcrs(gcrs, dt):
    """
    Evolve GCRS coordinates in time via rotation around Earth's rotational pole.

    Args:
        gcrs: [3] the initial GCRS coordinate.
        dt: the amount of time, positive or negative.

    Returns:
        [3] the evolved position in GCRS
    """
    # TODO: could replace InterpolatedArrays with application of this about a reference time.
    omega = 7.292115315411851e-05  # ~ 2 * np.pi / ((23 + 56 / 60) * 3600)
    alpha_pole = 0.015949670685007602
    delta_pole = 1.5683471107500062
    return efficient_rodriges_rotation(
        x_proj=gcrs,
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
