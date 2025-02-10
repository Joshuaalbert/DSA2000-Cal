import dataclasses
from functools import partial

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from dsa2000_cal.common.array_types import FloatArray
from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import time_to_jnp

tfpd = tfp.distributions


def compute_ionosphere_intersection(
        x_gcrs, k_gcrs,
        x0_gcrs,
        bottom: FloatArray, width: FloatArray
):
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


def test_compute_ionosphere_intersection():
    x_gcrs = jnp.array([0.0, 0.0, 0.0])
    k_gcrs = jnp.array([0.0, 0.0, 1.0])
    x0_gcrs = jnp.array([0.0, 0.0, 0.0])
    bottom = 0.5
    width = 1.0
    smin, smax = compute_ionosphere_intersection(x_gcrs, k_gcrs, x0_gcrs, bottom, width)
    assert smin == 0.5
    assert smax == 1.5


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
        A = u.T @ self.Sigma_inv @ u
        B = d.T @ self.Sigma_inv @ u
        C = d.T @ self.Sigma_inv @ d

        exponent = -0.5 * (C - B ** 2 / A)
        erf_arg = jnp.sqrt(A / 2)
        s1 = t1 + B / A
        s2 = t2 + B / A

        integral = (jnp.exp(exponent) * jnp.sqrt(np.pi / (2 * A))
                    * (jax.lax.erf(erf_arg * s2) - jax.lax.erf(erf_arg * s1)))
        integral = jnp.where(A < 1e-12, 0., integral)
        return integral

    def infinite_integral(self, a, u):
        if self.mu is not None:
            d = a - self.mu
        else:
            d = a
        A = u.T @ self.Sigma_inv @ u
        B = d.T @ self.Sigma_inv @ u
        C = d.T @ self.Sigma_inv @ d

        exponent = -0.5 * (C - B ** 2 / A)

        integral = jnp.exp(exponent) * jnp.sqrt(2 * np.pi / A)
        integral = jnp.where(A < 1e-12, 0., integral)
        return integral


@dataclasses.dataclass(eq=False)
class IonosphereLayerIntegral:
    """
    An ionosphere layer with Gaussian radial basis function parametrisation. This is equivalent to traditional RBF,
    or exponentiated quadratic kernel.
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

    def __post_init__(self):
        eye = jnp.eye(3)
        Sigma = eye * self.length_scale ** 2
        Sigma_inv = eye / self.length_scale ** 2
        self.gaussian_line_integral = GaussianLineIntegral(None, Sigma, Sigma_inv=Sigma_inv)

    def double_tomographic_integral(self, x1, x2, s1, s2, s1m, s1p, s2m, s2p, resolution: int = 30):
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

        def integrand(_s2):
            a = x1 - x2 - _s2 * s2
            u = s1
            return self.gaussian_line_integral.finite_integral(a, u, s1m, s1p)

        ds2 = (s2p - s2m) / resolution
        _s2_array = jnp.linspace(s2m, s2p, resolution + 1)
        summnd = jax.vmap(integrand)(_s2_array)
        integral = 0.5 * jnp.sum(summnd[:-1] + summnd[1:]) * ds2
        return self.fed_sigma * integral

    def numerical_double_integral(self, x1, x2, s1_hat, s2_hat, s1m, s1p, s2m, s2p, resolution: int):
        def integrand(s1, s2):
            x = x1 + s1 * s1_hat - x2 - s2 * s2_hat
            return self.fed_sigma * self.gaussian_line_integral.gaussian(x)

        s1_array = jnp.linspace(s1m, s1p, resolution + 1)
        s2_array = jnp.linspace(s2m, s2p, resolution + 1)
        ds1 = (s1p - s1m) / resolution
        ds2 = (s2p - s2m) / resolution
        integral = jnp.sum(jax.vmap(lambda s1: jax.vmap(lambda s2: integrand(s1, s2))(s2_array))(s1_array)) * ds1 * ds2
        return integral

    def calibrate_resolution(self, max_sep, max_angle, bottom, width, target_rtol=0.01):
        u_sep = jnp.linspace(0., max_sep, 10)
        u_angle = jnp.linspace(0., max_angle, 10)
        U_sep, U_angle = jnp.meshgrid(u_sep, u_angle, indexing='ij')

        def eval_baseline(resolution):
            def inner_eval(u_sep, u_angle):
                x1 = jnp.array([0., 0., 0.])
                x2 = jnp.array([u_sep, 0., 0.])
                s1 = jnp.array([0., 0., 1.])
                s2 = jnp.array([0., jnp.sin(u_angle), jnp.cos(u_angle)])
                integral = self.numerical_double_integral(x2, x1, s2, s1, 0., bottom, 0., width, resolution=resolution)
                return integral

            return jax.vmap(inner_eval)(U_sep.flatten(), U_angle.flatten())

        def eval(baseline, resolution):
            def inner_eval(u_sep, u_angle):
                x1 = jnp.array([0., 0., 0.])
                x2 = jnp.array([u_sep, 0., 0.])
                s1 = jnp.array([0., 0., 1.])
                s2 = jnp.array([0., jnp.sin(u_angle), jnp.cos(u_angle)])
                integral1 = self.double_tomographic_integral(x1, x2, s1, s2, 0., bottom, 0., width,
                                                             resolution=resolution)
                return integral1

            error = (jax.vmap(inner_eval)(U_sep.flatten(), U_angle.flatten()) - baseline) / baseline
            return jnp.mean(error), jnp.sqrt(jnp.mean(error ** 2))

        eval_jit = jax.jit(eval, static_argnames=['resolution'])

        baseline = eval_baseline(2000)

        for resolution in range(1, 200):
            mean_error, stddev_error = eval_jit(baseline, resolution)
            plt.scatter(resolution, mean_error, s=1, c='black')
            plt.scatter(resolution, stddev_error, s=1, c='red')
            if jnp.abs(mean_error) + stddev_error < target_rtol:
                break
        plt.xlabel('Resolution')
        plt.ylabel('Log10 Relative Error')
        plt.title(f"Rel error for max_sep={max_sep}, max_angle={max_angle}, bottom={bottom}, width={width}")
        plt.show()

    def calc_intersections(self, x, shat):
        d2_bottom = (self.x0_radius + self.bottom) ** 2
        d2_top = (self.x0_radius + self.bottom + self.width) ** 2
        xs = x @ shat
        xx = x @ x
        s_bottom = -xs + jnp.sqrt(xs ** 2 + (xx - d2_bottom))
        s_top = -xs + jnp.sqrt(xs ** 2 + (xx - d2_top))
        return s_bottom, s_top

    def project_geodesic(self, x, shat):
        s_bottom, s_top = self.calc_intersections(x, shat)
        x_bottom = x + s_bottom * shat
        x_top = x + s_top * shat
        return x_bottom, x_top

    def apply_frozen_flow(self, x_proj, t):
        bottom_radius = self.x0_radius + self.bottom
        omega = self.bottom_velocity / bottom_radius
        return efficient_rodriges_rotation(
            x_proj=x_proj,
            rdot=self.radial_velocity,
            omega=omega,
            dt=t,
            alpha_pole=self.longitude_pole,
            delta_pole=self.latitude_pole
        )

    def compute_kernel(self, x1, s1_hat, t1, x2, s2_hat, t2, resolution: int = 30):
        # project both to intersections with layer
        x1_bottom, x1_top = self.project_geodesic(x1, s1_hat)
        x2_bottom, x2_top = self.project_geodesic(x2, s2_hat)
        # Apply frozen flow, to find point in reference field
        x1_bottom = self.apply_frozen_flow(x1_bottom, t1)
        x1_top = self.apply_frozen_flow(x1_top, t1)
        x2_bottom = self.apply_frozen_flow(x2_bottom, t2)
        x2_top = self.apply_frozen_flow(x2_top, t2)
        # Define new vectors
        s1 = x1_top - x1_bottom
        s2 = x2_top - x2_bottom
        s1_norm = jnp.linalg.norm(s1)
        s2_norm = jnp.linalg.norm(s2)
        s1_hat = s1 / s1_norm
        s2_hat = s2 / s2_norm
        return self.double_tomographic_integral(
            x1_bottom, x2_bottom, s1_hat, s2_hat,
            0., s1_norm, 0., s2_norm, resolution=resolution
        )

    def compute_mean(self, x, s, t):
        x_bottom, x_top = self.project_geodesic(x, s)
        # Apply frozen flow, to find point in reference field
        x_bottom = self.apply_frozen_flow(x_bottom, t)
        x_top = self.apply_frozen_flow(x_top, t)
        s = x_top - x_bottom
        intersection_length = jnp.sqrt(jnp.sum(jnp.square(s)))
        return intersection_length * self.fed_mu

    def compute_process_params(self, antennas_gcrs: InterpolatedArray, times: FloatArray,
                               directions_gcrs: InterpolatedArray):
        @partial(
            multi_vmap,
            in_mapping="[T]",
            out_mapping="[T,...],[T,...],[T]"
        )
        def get_coords(time):
            x = antennas_gcrs(time)  # [A, 3]
            s = directions_gcrs(time)  # [D, 3]
            x, s = jnp.broadcast_arrays(x[None, :, :], s[:, None, :])
            t = jnp.broadcast_to(time[None, None], np.shape(x)[:-1])
            return x, s, t

        x, s, t = get_coords(times)  # [T, D, A, 3], [T, D, A, 3], [T, D, A]
        x = jax.lax.transpose(x, [1, 0, 2, 3])
        s = jax.lax.transpose(s, [1, 0, 2, 3])
        t = jax.lax.transpose(t, [1, 0, 2])

        # return x, s, t

        D, T, A = jnp.shape(t)

        @partial(
            multi_vmap,
            in_mapping="[D,T,A,3],[D,T,A,3],[D,T,A]",
            out_mapping="[D,T,A]"
        )
        def get_mean(x1, s1_hat, t1):
            return self.compute_mean(x1, s1_hat, t1)

        @partial(
            multi_vmap,
            in_mapping="[D,T,A,3],[D,T,A,3],[D,T,A],[D',T',A',3],[D',T',A',3],[D',T',A']",
            out_mapping="[D,T,A,D',T',A']"
        )
        def get_kernel(x1, s1_hat, t1, x2, s2_hat, t2):
            return self.compute_kernel(x1, s1_hat, t1, x2, s2_hat, t2)

        K = get_kernel(x, s, t, x, s, t)  # [D,T,A,D',T',A']

        mean = get_mean(x, s, t)
        return K, mean

    def sample(self, key, antennas_gcrs: InterpolatedArray, times: FloatArray, directions_gcrs: InterpolatedArray):
        """
        Sample ionosphere DTEC or TEC.

        Args:
            key: PRNGKey
            antennas_gcrs: [A] antennas (time) -> [A]
            times: [T] times
            directions_gcrs: [D] directions (time) -> [D]

        Returns:
            [D, T, A] shaped array of DTEC or TEC
        """

        K, mean = self.compute_process_params(antennas_gcrs, times, directions_gcrs)
        D, T, A = np.shape(mean)

        K = jax.lax.reshape(K, (D * T * A, D * T * A))
        mean = jax.lax.reshape(mean, [D * T * A])

        # Sample now

        sample = tfpd.MultivariateNormalTriL(
            loc=mean,
            scale_tril=jnp.linalg.cholesky(K + jnp.square(0.5) * jnp.eye(np.shape(K)[-1]))
        ).sample(
            seed=key
        )

        sample = jax.lax.reshape(sample, [D, T, A])
        return sample


def test_ionosphere():
    ref_time = at.Time.now()
    times = ref_time + 600 * np.arange(1) * au.s
    antennas: ac.EarthLocation = ac.EarthLocation.from_geocentric(
        np.random.uniform(6700, 6701, 1) * au.km,
        np.random.uniform(6700, 6701, 1) * au.km,
        np.random.uniform(6700, 6701, 1) * au.km
    )
    x0_radius = np.linalg.norm(antennas[0].get_gcrs(ref_time).cartesian.xyz.to('km')).value
    directions = ac.ICRS(
        np.random.uniform(0, 1, 100) * au.deg, np.random.uniform(0, 2, 100) * au.deg
    )
    model_times = times[[0, -1]]
    model_times_jax = time_to_jnp(model_times, ref_time)
    antennas_gcrs = InterpolatedArray(
        model_times_jax,
        antennas.get_gcrs(model_times[:, None]).cartesian.xyz.to('km').value.transpose((1, 2, 0)),
        axis=0,
        regular_grid=True,
        check_spacing=True
    )

    directions_gcrs = InterpolatedArray(
        model_times_jax,
        directions.transform_to(antennas[0].get_gcrs(model_times[:, None])).cartesian.xyz.value.transpose((1, 2, 0)),
        axis=0,
        regular_grid=True,
        check_spacing=True
    )
    times = time_to_jnp(times, ref_time)

    key = jax.random.PRNGKey(0)

    ionosphere = IonosphereLayerIntegral(
        length_scale=5.,
        longitude_pole=0.,
        latitude_pole=np.pi / 2.,
        bottom_velocity=0.120,
        radial_velocity=0.,
        x0_radius=x0_radius,
        bottom=100,
        width=200,
        fed_mu=1,
        fed_sigma=0.1
    )

    K, mean = ionosphere.compute_process_params(antennas_gcrs, times, directions_gcrs)
    D, T, A = np.shape(mean)
    K = jax.lax.reshape(K, (D * T * A, D * T * A))
    mean = jax.lax.reshape(mean, [D * T * A])

    sc = plt.scatter(directions.ra, directions.dec, c=mean)
    plt.colorbar(sc)
    plt.show()

    plt.imshow(K)
    plt.colorbar()
    plt.show()
    # return

    sample = ionosphere.sample(
        key=key,
        antennas_gcrs=antennas_gcrs,
        directions_gcrs=directions_gcrs,
        times=times
    )

    print(sample)

    sc = plt.scatter(directions.ra, directions.dec, c=sample.flatten())
    plt.colorbar(sc)
    plt.show()


def evolve_antennas(x0_gcrs, dt):
    omega = 7.307418536428037e-05  # 2 * np.pi / ((23 + 56 / 60) * 3600)
    return efficient_rodriges_rotation(
        x_proj=x0_gcrs,
        rdot=0.,
        omega=omega,
        dt=dt,
        alpha_pole=0,
        delta_pole=np.pi / 2.
    )


def test_evolve_antennas():
    import astropy.coordinates as ac
    import astropy.time as at
    import astropy.units as au

    a: ac.EarthLocation = ac.EarthLocation.of_site('vla')
    k = ac.ICRS(0 * au.deg, 10 * au.deg)
    t = at.Time.now()
    t_next = t + 3600 * au.s

    a_gcrs = a.get_gcrs(t)
    cosdec = np.cos(a.lat)
    omega = (a_gcrs.pm_ra_cosdec.to('rad/s') / cosdec)
    print(f"omega = {omega}")
    print(k.transform_to(a_gcrs).cartesian.xyz.value)
    print(k.transform_to(a.get_gcrs(t_next)).cartesian.xyz.value)

    a_gcrs_xyz_from = a_gcrs.cartesian.xyz.to('km').value
    a_gcrs_xyz_to = a.get_gcrs(t_next).cartesian.xyz.to('km').value

    np.testing.assert_allclose(evolve_antennas(a_gcrs_xyz_from, 3600), a_gcrs_xyz_to, atol=2.)


def efficient_rodriges_rotation(x_proj, rdot, omega, dt, alpha_pole, delta_pole):
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


def test_efficient_rodriges_rotation():
    x = jnp.asarray([1., 0., 0.])
    np.testing.assert_allclose(
        efficient_rodriges_rotation(x, 0., np.pi / 2, 1., 0., np.pi / 2.),
        [0., 1., 0.],
        atol=1e-6
    )
    np.testing.assert_allclose(
        efficient_rodriges_rotation(x, 0., np.pi, 1., 0., np.pi / 2.),
        [-1., 0., 0.],
        atol=1e-6
    )
    np.testing.assert_allclose(
        efficient_rodriges_rotation(x, 0., np.pi / 2., 1., np.pi / 2., 0.),
        [0., 0., -1.],
        atol=1e-6
    )
