import numpy as np
from scipy.integrate import quad
from scipy.special import fresnel, erf


def make_quadratic(a, b, fa, fm, fb):
    """
    Return the coefficients A, D, E, m, h for the unique quadratic:

        f(t) = A * (t - m)^2 + D * (t - m) + E

    that matches the three conditions:
        f(a) = fa,
        f((a+b)/2) = fm,
        f(b) = fb.

    Here:
        m = (a + b) / 2
        h = (b - a) / 2

    and we solve for A, D, E to satisfy the interpolation conditions.
    """
    m = 0.5 * (a + b)
    h = 0.5 * (b - a)

    # Condition: f(m) = E = fm
    E = fm

    # Let f(t) = A*(t - m)^2 + D*(t - m) + E.
    # Then:
    #   f(a) = A*(-h)^2 + D*(-h) + E = A*h^2 - D*h + E = fa
    #   f(b) = A*(+h)^2 + D*(+h) + E = A*h^2 + D*h + E = fb
    #
    # Substituting E = fm => we get
    #   A*h^2 - D*h + fm = fa   --> (1)
    #   A*h^2 + D*h + fm = fb   --> (2)
    #
    # Subtract (1) from (2): 2*D*h = fb - fa => D = (fb - fa) / (2*h)
    # Add (1) and (2):  2*A*h^2 + 2*fm = fa + fb => 2*A*h^2 = fa + fb - 2*fm => A = (fa + fb - 2*fm) / (2*h^2).
    #
    # But h^2 = ((b-a)/2)^2 = (b-a)^2 / 4. So:
    #   A = 2*(fa + fb - 2*fm) / ((b-a)^2)
    #   D = (fb - fa)/( (b-a) ).

    A = (fa + fb - 2 * fm) / (2.0 * (h ** 2)) + 0j
    D = ((fb - fa) / (2.0 * h))  # same as (fb - fa)/(b-a)

    return A, D, E, m, h


def test_make_quadratic():
    def g(t):
        return -2 * t ** 2 - 3 * t + 10

    def f(A, D, E, m, h, t):
        return A * (t - m) ** 2 + D * (t - m) + E

    a, b = 0.0, 2.0
    fa = g(a)
    fm = g(0.5 * (a + b))
    fb = g(b)

    A, D, E, m, h = make_quadratic(a, b, fa, fm, fb)
    print(f"A = {A}, D = {D}, E = {E}, m = {m}, h = {h}")

    t = np.linspace(a, b, 100)
    y = f(A, D, E, m, h, t)

    np.testing.assert_allclose(y, g(t))


def integral_fresnel(a, b, fa, fm, fb):
    """
    Compute \int_a^b exp(i f(t)) dt
    where f(t) is the unique quadratic passing through:
      (a, fa), (m=(a+b)/2, fm), (b, fb).
    Uses Fresnel integrals.
    """
    A, D, E, m, h = make_quadratic(a, b, fa, fm, fb)

    A *= 2 * np.pi
    D *= 2 * np.pi
    E *= 2 * np.pi

    # I = e^{i(E - D^2/(4A))} * \int_{v_-}^{v_+} e^{i A v^2} dv
    # v = u + D/(2A),  u = t - m
    # v_- = -h + D/(2A),  v_+ = h + D/(2A)

    phase_factor = np.exp(1j * (E - D * D / (4.0 * A)))  # e^{ i(E - D^2/(4A)) }
    v_minus = -h + D / (2.0 * A)
    v_plus = h + D / (2.0 * A)

    # Next, express \int e^{i A v^2} dv in terms of Fresnel integrals.
    # We'll define z = sqrt(2A/pi) * v.
    # Then dv = sqrt(pi/(2A)) dz and the exponent becomes i*(pi/2)*z^2.
    # The Fresnel integrals are:
    #   C(z) = \int_0^z cos( (pi/2) t^2 ) dt
    #   S(z) = \int_0^z sin( (pi/2) t^2 ) dt

    def fresnel_part(x):
        """
        Return \int_0^x e^{i (pi/2) t^2 } dt = C(x) + i S(x).
        scipy.special.fresnel returns (S(x), C(x)) but in the notation:
            S(x) = \int_0^x sin(pi x'^2 / 2) dx'
            C(x) = \int_0^x cos(pi x'^2 / 2) dx'
        So e^{i (pi/2) x'^2} = cos(...) + i sin(...).
        """
        s, c = fresnel(x)
        return c + 1j * s

    # We'll define a helper to compute \int_{z1}^{z2} e^{i (pi/2) z^2 } dz
    # = [fresnel_part(z2) - fresnel_part(z1)].

    def complex_fresnel_diff(z1, z2):
        return fresnel_part(z2) - fresnel_part(z1)

    # Scale factors for z
    scale = np.sqrt(2.0 * A / np.pi)

    z_minus = scale * v_minus
    z_plus = scale * v_plus

    integral_piece = np.sqrt(np.pi / (2.0 * A)) * complex_fresnel_diff(z_minus, z_plus)

    return phase_factor * integral_piece


def integral_erf(a, b, fa, fm, fb):
    """
    Compute \int_a^b exp(i f(t)) dt
    using the closed-form expression in terms of the complex error function (erf).

    If A < 0, the square-root and erf arguments become purely imaginary,
    but Python's scipy.special.erf can handle complex arguments.
    """
    A, D, E, m, h = make_quadratic(a, b, fa, fm, fb)

    A *= 2 * np.pi
    D *= 2 * np.pi
    E *= 2 * np.pi

    # I = e^{ i ( E - D^2/(4A) ) } * \int_{v_-}^{v_+} e^{ i A v^2 } dv
    # where v = u + D/(2A),  u = t - m
    v_minus = -h + D / (2.0 * A)
    v_plus = h + D / (2.0 * A)
    phase_factor = np.exp(1j * (E - (D ** 2) / (4.0 * A)))

    # Known formula:
    # \int e^{ i A v^2 } dv = sqrt(pi/(4A)) e^{ i pi/4 } erf( e^{- i pi/4} sqrt{A} v )
    # Evaluate between v_minus, v_plus.

    # We'll define the antiderivative function:
    def antiderivative_erf(v):
        # sqrtA might be imaginary if A < 0
        sqrtA = np.sqrt(A + 0j)  # force complex
        coeff = np.sqrt(np.pi / (4.0 * A)) * np.exp(1j * np.pi / 4)
        return coeff * erf(np.exp(-1j * np.pi / 4) * sqrtA * v)

    integral_piece = antiderivative_erf(v_plus) - antiderivative_erf(v_minus)
    return phase_factor * integral_piece

def integral_erf_2(a, b, fa, fm, fb):
    """
    Compute \int_a^b exp(i f(t)) dt
    using the closed-form expression in terms of the complex error function (erf).

    If A < 0, the square-root and erf arguments become purely imaginary,
    but Python's scipy.special.erf can handle complex arguments.
    """
    A, D, E, m, h = make_quadratic(a, b, fa, fm, fb)

    A *= 2 * np.pi
    D *= 2 * np.pi
    E *= 2 * np.pi

    # I = e^{ i ( E - D^2/(4A) ) } * \int_{v_-}^{v_+} e^{ i A v^2 } dv
    # where v = u + D/(2A),  u = t - m
    v_minus = -h + D / (2.0 * A)
    v_plus = h + D / (2.0 * A)
    phase_factor = np.exp(1j * (E - (D ** 2) / (4.0 * A)))

    # Known formula:
    # \int e^{ i A v^2 } dv = sqrt(pi/(4A)) e^{ i pi/4 } erf( e^{- i pi/4} sqrt{A} v )
    # Evaluate between v_minus, v_plus.

    # We'll define the antiderivative function:
    def antiderivative_erf(v):
        # sqrtA might be imaginary if A < 0
        sqrtA = np.sqrt(A + 0j)  # force complex
        coeff = np.sqrt(np.pi / (4.0 * A)) * np.exp(1j * np.pi / 4)
        return coeff * erf(np.exp(-1j * np.pi / 4) * sqrtA * v)

    integral_piece = antiderivative_erf(v_plus) - antiderivative_erf(v_minus)
    return phase_factor * integral_piece


def integral_numeric(a, b, fa, fm, fb):
    """
    Numerically compute \int_a^b exp(i f(t)) dt
    using scipy.integrate.quad, where f(t) is the unique quadratic.
    """
    A, D, E, m, h = make_quadratic(a, b, fa, fm, fb)

    def fquad(t):
        # f(t) = A*(t - m)^2 + D*(t - m) + E
        return A * (t - m) ** 2 + D * (t - m) + E

    def integrand(t):
        return np.exp(2j * np.pi * fquad(t))

    val, _ = quad(
        lambda x: integrand(x).real,  # real part
        a, b
    )
    vali, _ = quad(
        lambda x: integrand(x).imag,  # imaginary part
        a, b
    )
    return val + 1j * vali


def test_integrals():
    # Example data
    a, b = 0.0, 2.0

    # Let's pick some "f" values for demonstration:
    fa = 1.0
    fm = 2  # f((a+b)/2)
    fb = 1.0

    # Evaluate each integral
    I_fresnel = integral_fresnel(a, b, fa, fm, fb)
    I_erf = integral_erf(a, b, fa, fm, fb)
    I_num = integral_numeric(a, b, fa, fm, fb)

    print("Fresnel-based integral :", I_fresnel)
    print("erf-based integral     :", I_erf)
    print("Numerical integral     :", I_num)

    # Compare the difference
    err_fresnel = abs(I_fresnel - I_num)
    err_erf = abs(I_erf - I_num)

    print(f"Error (Fresnel vs numeric): {err_fresnel:.6g}")
    print(f"Error (erf vs numeric)    : {err_erf:.6g}")
