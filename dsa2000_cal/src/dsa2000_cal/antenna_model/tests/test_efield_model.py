import numpy as np

from src.dsa2000_cal.antenna_model.h5_efield_model import convert_spherical_e_field_to_cartesian


def test_convert_spherical_e_field_to_cartesian():
    # L-M frame is the same as (-Y)-X frame, i.e. Y is -L and X is M.

    # theta=pi/2 for all phi (Looking at horizon): -theta-hat == Bore-sight
    for phi in [0., np.pi / 2, np.pi, 3 * np.pi / 2]:
        e_x, e_y, e_z = convert_spherical_e_field_to_cartesian(0, -1, phi=phi, theta=np.pi / 2)
        np.testing.assert_allclose([e_x, e_y, e_z], [0, 0, 1], atol=1e-8)

    # At theta=pi/2 phi=0 (Looking North): phi-hat == y-hat
    e_x, e_y, e_z = convert_spherical_e_field_to_cartesian(1, 0, phi=0, theta=np.pi / 2)
    np.testing.assert_allclose([e_x, e_y, e_z], [0, 1, 0], atol=1e-8)

    # At theta=pi/2 phi=pi (Looking South): phi-hat == -y-hat
    e_x, e_y, e_z = convert_spherical_e_field_to_cartesian(1, 0, phi=np.pi, theta=np.pi / 2)
    np.testing.assert_allclose([e_x, e_y, e_z], [0, -1, 0], atol=1e-8)

    # When theta=0, phi=0 (Looking down bore): theta-hat == x-hat
    e_x, e_y, e_z = convert_spherical_e_field_to_cartesian(0, 1, phi=0, theta=0)
    np.testing.assert_allclose([e_x, e_y, e_z], [1, 0, 0], atol=1e-8)

    # When theta=0, phi=pi/2 (Looking down bore): theta-hat == y-hat
    e_x, e_y, e_z = convert_spherical_e_field_to_cartesian(0, 1, phi=np.pi / 2, theta=0)
    np.testing.assert_allclose([e_x, e_y, e_z], [0, 1, 0], atol=1e-8)