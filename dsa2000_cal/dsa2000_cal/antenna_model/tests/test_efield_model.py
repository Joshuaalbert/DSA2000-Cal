import numpy as np

from dsa2000_cal.antenna_model.h5_efield_model import convert_spherical_e_field_to_cartesian


def test_basic_conversions():
    # Edge case where theta = 0, phi = 0 ==> E_cartesian = [0, 1, 0]
    e_x, e_y, e_z = convert_spherical_e_field_to_cartesian(1, 0, 0, 0)
    np.testing.assert_allclose(e_x, 0, atol=1e-8)
    np.testing.assert_allclose(e_y, 1, atol=1e-8)
    np.testing.assert_allclose(e_z, 0, atol=1e-8)

    # Edge case where theta = pi/2, phi = pi/2 ==> E_cartesian = [-1, 0, -1]
    e_x, e_y, e_z = convert_spherical_e_field_to_cartesian(1, 1, np.pi / 2, np.pi / 2)
    np.testing.assert_allclose(e_x, -1, atol=1e-8)
    np.testing.assert_allclose(e_y, 0, atol=1e-8)  # Corrected expectation
    np.testing.assert_allclose(e_z, -1, atol=1e-8)

    # Edge case where theta = pi, phi = pi ==> E_cartesian = [1, -1, 0]
    e_x, e_y, e_z = convert_spherical_e_field_to_cartesian(1, 1, np.pi, np.pi)
    np.testing.assert_allclose(e_x, 1, atol=1e-8)
    np.testing.assert_allclose(e_y, -1, atol=1e-8)
    np.testing.assert_allclose(e_z, 0, atol=1e-8)
