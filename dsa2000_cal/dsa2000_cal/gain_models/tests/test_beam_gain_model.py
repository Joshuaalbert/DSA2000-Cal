import numpy as np

from dsa2000_cal.gain_models.beam_gain_model import lmn_from_phi_theta


def test_phi_theta_to_cartesian():
    # L = -Y, M = X, N = Z

    phi = 0.
    theta = 0.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [0, 0, 1], atol=1e-8)

    phi = np.pi / 2.
    theta = np.pi / 2.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [-1, 0, 0], atol=1e-8)

    phi = 0.
    theta = np.pi / 2.
    lmn = lmn_from_phi_theta(phi, theta)
    np.testing.assert_allclose(lmn, [0, 1, 0], atol=1e-8)
