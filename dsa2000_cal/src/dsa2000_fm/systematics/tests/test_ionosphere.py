from datetime import datetime

import jax
import numpy as np
from astropy import coordinates as ac, units as au, time as at
from jax import numpy as jnp
from scipy.integrate import quad

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_cal.solvers.multi_step_lm import lm_solver
from dsa2000_common.common.enu_frame import ENU
from dsa2000_fm.systematics.ionosphere import GaussianLineIntegral, construct_eval_interp_struct, \
    IonosphereLayer, calibrate_resolution
from dsa2000_fm.systematics.ionosphere_models import construct_ionosphere_model, fetch_ionosphere_data, \
    construct_ionosphere_model_from_didb_db
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

        return lm_solver(residual_fn, init_params, maxiter=100, verbose=True)

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


def test_construct_ionosphere_model():
    # https://giro.uml.edu/didbase/scaled.php
    # # Query for measurement intervals of time:
    # # 2024-04-07T21:00:00.000Z - 2024-04-08T21:01:00.000Z
    # #
    # # Data Selection:
    # # CS is Autoscaling Confidence Score (from 0 to 100, 999 if manual scaling, -1 if unknown)
    # # foF2 [MHz] - F2 layer critical frequency
    # # foF1 [MHz] - F1 layer critical frequency
    # # foE [MHz] - E layer critical frequency
    # # hmE [km] - Peak height of E-layer
    # # yE [km] - Half thickness of E-layer
    # # hmF2 [km] - Peak height F2-layer
    # # hmF1 [km] - Peak height F1-layer
    # # yF2 [km] - Half thickness of F2-layer, parabolic model
    # # yF1 [km] - Half thickness of F1-layer, parabolic model
    # # TEC [10^16 m^-2] - Total electron content
    # #
    # # All GIRO measurements are released under CC-BY-NC-SA 4.0 license
    # # Please follow the Lowell GIRO Data Center RULES OF THE ROAD
    # # https://ulcar.uml.edu/DIDBase/RulesOfTheRoadForDIDBase.htm
    # # Requires acknowledgement of BC840 data provider
    # #
    # #Time                     CS   foF2 QD  foF1 QD   foE QD    hmE QD     yE QD   hmF2 QD   hmF1 QD    yF2 QD    yF1 QD   TEC QD
    # 2024-04-07T21:05:05.000Z 100  9.475 //  5.08 //  3.51 //  101.6 //   11.3 //  288.7 //  171.0 //   95.7 //   55.4 //  29.0 //

    ionosphere = construct_ionosphere_model(
        x0_radius=6371.0,
        f0E = 3.51,
        f0F1 = 5.08,
        f0F2 = 9.475,
        hmE = 101.6,
        hmF1 = 171.0,
        hmF2 = 288.7,
        yE = 11.3,
        yF1 = 55.4,
        yF2 = 95.7,
        vtec = 29.0,
        longitude_pole=0.,
        latitude_pole=np.pi / 2.,
        turbulent=True
    )
    print(ionosphere)


def test_fetch_ionosphere_data():
    # Define the start and end datetimes with timezone awareness (UTC in this case)
    start_dt = datetime(2024, 4, 7, 21, 0, 0)
    end_dt = datetime(2024, 4, 8, 21, 1, 0)
    station_code = "AU930"

    data = fetch_ionosphere_data(start_dt, end_dt, station_code)
    for record in data:
        print(record)

    # concate tree
    data = jax.tree.map(lambda *x: np.stack(x),*data)
    print(data)

    # plot histograms of each column
    import pylab as plt
    plt.hist(data.vtec, bins='auto')
    plt.title("VTEC Histogram")
    plt.xlabel("VTEC [TECU]")
    plt.show()

    plt.hist(data.f0E, bins='auto')
    plt.title("f0E Histogram")
    plt.xlabel("f0E [MHz]")
    plt.show()

    plt.hist(data.hmE, bins='auto')
    plt.title("hmE Histogram")
    plt.xlabel("hmE [km]")
    plt.show()

    plt.hist(data.yE, bins='auto')
    plt.title("yE Histogram")
    plt.xlabel("yE [km]")
    plt.show()

def test_construct_ionosphere_model_from_didb_db():
    ionosphere = construct_ionosphere_model_from_didb_db(
        start=datetime(2024, 4, 7, 21, 0, 0),
        end=datetime(2024, 4, 8, 21, 1, 0),
        ursi_station="AU930",
        x0_radius=6371.0,
    )

    print(ionosphere)