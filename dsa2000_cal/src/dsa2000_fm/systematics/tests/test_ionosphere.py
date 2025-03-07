import jax
import numpy as np
from astropy import coordinates as ac, units as au, time as at
from jax import numpy as jnp
from scipy.integrate import quad
from tomographic_kernel.frames import ENU

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_cal.solvers.multi_step_lm import MultiStepLevenbergMarquardt
from dsa2000_fm.systematics.ionosphere import GaussianLineIntegral, construct_eval_interp_struct, \
    IonosphereLayer, compute_ionosphere_intersection, calibrate_resolution
from dsa2000_fm.systematics.ionosphere import efficient_rodriges_rotation, evolve_gcrs, calc_intersections


def gaussian_numerical_line_integral(t1, t2, a, u, mu, Sigma):
    """Numerical integration of line integral through 3D Gaussian"""
    Sigma_inv = np.linalg.inv(Sigma)

    def integrand(t):
        x = a + t * u
        dx = x - mu
        return np.exp(-0.5 * dx.T @ Sigma_inv @ dx)

    result, _ = quad(integrand, t1, t2, epsabs=1e-9, epsrel=1e-9)
    return result


def test_gaussian_line_integral():
    # Test parameters
    mu = np.array([1.0, -0.5, 2.0])
    Sigma = np.array([[2.0, 0.5, 0.9],
                      [0.5, 1.5, -0.3],
                      [0.9, -0.3, 3.0]])
    a = np.array([0.5, 1.0, -1.0])
    u = np.array([1.0, 0.5, -0.5])

    # Finite integral test
    t1, t2 = -1.0, 1.0
    closed = GaussianLineIntegral(mu, Sigma).finite_integral(a, u, t1, t2)
    numerical = gaussian_numerical_line_integral(t1, t2, a, u, mu, Sigma)

    print(f"Finite Integral Results ({t1} to {t2}):")
    print(f"Closed-form: {closed:.9f}")
    print(f"Numerical:   {numerical:.9f}")
    print(f"Difference:  {np.abs(closed - numerical):.2e}\n")

    # Infinite integral test (using large bounds)
    t_inf = 1e3  # Practical approximation of infinity
    closed_inf = GaussianLineIntegral(mu, Sigma).infinite_integral(a, u)
    numerical_inf = gaussian_numerical_line_integral(-t_inf, t_inf, a, u, mu, Sigma)

    print("Infinite Integral Results:")
    print(f"Closed-form: {closed_inf:.9f}")
    print(f"Numerical:   {numerical_inf:.9f}")
    print(f"Difference:  {np.abs(closed_inf - numerical_inf):.2e}")


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


def test_infer_params():
    # Infers the rotation parameters
    a: ac.EarthLocation = ac.EarthLocation.of_site('vla')
    k = ac.ICRS(0 * au.deg, 10 * au.deg)
    dt = np.linspace(0., 86400., 100)
    t = at.Time.now() + dt * au.s

    a_gcrs = a.get_gcrs(t).cartesian.xyz.to('km').value.T

    x0 = a_gcrs[0, :]  # [3]
    x_after_expect = a_gcrs[1:, :]
    dt = dt[1:]

    # Perform optimisation with scipy
    @jax.jit
    def run_solver(x0, dt, x_after_expect):
        init_params = jnp.array([7.307418278891732e-05 * 1e5, 0., np.pi / 2.])

        def residual_fn(params):
            omega, alpha_pole, delta_pole = params
            omega *= 1e-5
            x_after = jax.vmap(lambda dt: efficient_rodriges_rotation(
                x_proj=x0,
                rdot=0.,
                omega=omega,
                dt=dt,
                alpha_pole=alpha_pole,
                delta_pole=delta_pole
            ))(dt)
            return x_after - x_after_expect

        solver = MultiStepLevenbergMarquardt(
            residual_fn=residual_fn,
            num_approx_steps=0,
            num_iterations=100,
            verbose=True
        )
        state = solver.create_initial_state(init_params)
        return solver.solve(state)

    state, diagnostics = run_solver(
        x0=x0,
        dt=dt,
        x_after_expect=x_after_expect
    )
    omega, alpha_pole, delta_pole = state.x
    omega *= 1e-5
    print(omega, alpha_pole, delta_pole)


def test_evolve_gcrs():
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
    latdot = a_gcrs.pm_dec.to('rad/s')
    print(f"omega = {omega}")
    print(f"latdot = {latdot}")
    print(k.transform_to(a_gcrs).cartesian.xyz.value)
    print(k.transform_to(a.get_gcrs(t_next)).cartesian.xyz.value)

    a_gcrs_xyz_from = a_gcrs.cartesian.xyz.to('km').value
    a_gcrs_xyz_to = a.get_gcrs(t_next).cartesian.xyz.to('km').value

    np.testing.assert_allclose(evolve_gcrs(a_gcrs_xyz_from, 3600), a_gcrs_xyz_to, atol=0.001)


def test_calc_intersections():
    # Points in same direction
    x = jnp.asarray([0., 0., 1.])
    shat = jnp.asarray([0., 0., 1.])
    x0_radius = jnp.linalg.norm(x)
    bottom = 1.
    width = 1.
    s_bottom, s_top = calc_intersections(x, shat, x0_radius, bottom, width)
    np.testing.assert_allclose(s_bottom, 1.)
    np.testing.assert_allclose(s_top, 2.)

    # Points perp, so can use pythagoras theorem to get distance.
    x = jnp.asarray([0., 0., 1.])
    shat = jnp.asarray([1., 0., 0.])
    x0_radius = jnp.linalg.norm(x)
    bottom = 1.
    width = 1.
    s_bottom, s_top = calc_intersections(x, shat, x0_radius, bottom, width)
    s_bottom2_exp = (x0_radius + bottom) ** 2 - x0_radius ** 2
    s_top2_exp = (x0_radius + bottom + width) ** 2 - x0_radius ** 2
    np.testing.assert_allclose(s_bottom ** 2, s_bottom2_exp)
    np.testing.assert_allclose(s_top ** 2, s_top2_exp)


def test_ionosphere():
    ref_time = at.Time.now()
    times = ref_time + 60 * np.arange(2) * au.s
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))
    antennas = array.get_antennas()[::2]
    ref_location = array.get_array_location()
    phase_center = ENU(0, 0.9, 1, obstime=ref_time, location=ref_location).transform_to(ac.ICRS())

    directions = phase_center[None]

    T = int((times.max() - times.min()) / (1 * au.min)) + 1
    model_times = times.min() + np.arange(0., T) * au.min

    x0_radius, times_jax, antennas_gcrs, directions_gcrs = construct_eval_interp_struct(
        antennas, ref_location, times, ref_time, directions, model_times
    )

    ionosphere = IonosphereLayer(
        length_scale=2.,
        longitude_pole=0.,
        latitude_pole=np.pi / 2.,
        bottom_velocity=0.120,
        radial_velocity=0.,
        x0_radius=x0_radius,
        bottom=100,
        width=200,
        fed_mu=10.,  # 10^11 e-/m^3
        fed_sigma=10.  # 10^11 e-/m^3
    )

    x = antennas_gcrs(times_jax[0])[0]
    s = directions_gcrs(times_jax[0])[0]
    x_bottom, x_top = ionosphere.project_geodesic(x, s)
    np.testing.assert_allclose(np.linalg.norm(x_bottom), x0_radius + ionosphere.bottom)
    np.testing.assert_allclose(np.linalg.norm(x_top), x0_radius + ionosphere.bottom + ionosphere.width)
    x_bottom_ff = ionosphere.apply_frozen_flow(x_bottom, 0)
    x_top_ff = ionosphere.apply_frozen_flow(x_top, 0)
    np.testing.assert_allclose(np.linalg.norm(x_top_ff - x_bottom_ff), ionosphere.width, atol=1e3)
    # Since rdot=0
    np.testing.assert_allclose(np.linalg.norm(x_bottom_ff), x0_radius + ionosphere.bottom)
    np.testing.assert_allclose(np.linalg.norm(x_top_ff), x0_radius + ionosphere.bottom + ionosphere.width)
    # Also preserved over 1 hr
    x_bottom_ff = ionosphere.apply_frozen_flow(x_bottom, 3600)
    x_top_ff = ionosphere.apply_frozen_flow(x_top, 3600)
    np.testing.assert_allclose(np.linalg.norm(x_top_ff - x_bottom_ff), ionosphere.width, atol=1e3)
    # Since rdot=0
    np.testing.assert_allclose(np.linalg.norm(x_bottom_ff), x0_radius + ionosphere.bottom)
    np.testing.assert_allclose(np.linalg.norm(x_top_ff), x0_radius + ionosphere.bottom + ionosphere.width)


def test_calibrate_resolution():
    layer = IonosphereLayer(
        length_scale=5.,
        longitude_pole=0.,
        latitude_pole=np.pi / 2,
        bottom_velocity=0.1,
        radial_velocity=0.,
        x0_radius=6300.,
        bottom=200.,
        width=200.,
        fed_mu=10.,
        fed_sigma=0.1
    )
    fov = 8 / 57  # rad
    max_baseline = 20
    height = 400
    max_sep = 0.5 * fov * height + max_baseline
    resolution = calibrate_resolution(layer, max_sep, fov)
