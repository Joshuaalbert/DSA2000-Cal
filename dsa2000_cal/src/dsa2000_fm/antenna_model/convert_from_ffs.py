import glob
from typing import List, NamedTuple

import h5py
import jax
import numpy as np

from dsa2000_common.common.array_types import FloatArray, ComplexArray
from dsa2000_fm.antenna_model.h5_efield_model import print_h5_structure, H5AntennaModelV1


class FarFieldEField(NamedTuple):
    freqs: FloatArray  # [num_freqs]
    theta: FloatArray  # [num_theta]
    phi: FloatArray  # [num_phi]
    e_field_X_theta: ComplexArray  # [freq, theta, phi]
    e_field_X_phi: ComplexArray  # [freq, theta, phi]
    e_field_Y_theta: ComplexArray  # [freq, theta, phi]
    e_field_Y_phi: ComplexArray  # [freq, theta, phi]


def convert_ffs_files_to_h5(xpol_ffs_files: List[str], h5_file: str, ypol_ffs_files: List[str] | None = None):
    # $ head farfield_source_\(f\=1.35\)_\[FEED\]1\].ffs
    # // CST Farfield Source File
    #
    # // Version:
    # 3.0
    #
    # // Data Type
    # Farfield
    #
    # // #Frequencies
    # 1
    #
    # // Position
    # 0.000000e+00 0.000000e+00 0.000000e+00
    #
    # // zAxis
    # 0.000000e+00 0.000000e+00 1.000000e+00
    #
    # // xAxis
    # 1.000000e+00 0.000000e+00 0.000000e+00
    #
    # // Radiated/Accepted/Stimulated Power , Frequency
    # 4.940758e-01
    # 4.958051e-01
    # 5.000000e-01
    # 1.350000e+09
    #
    #
    # // >> Total #phi samples, total #theta samples
    # 3601 1801
    #
    # // >> Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi):
    #    0.000     0.000   9.44542999e+01  -2.85977631e+02 -3.32716656e+00  -6.71975255e-01
    #    0.000     0.100   9.40814285e+01  -2.85805450e+02 -3.34284925e+00  -5.96707880e-01
    #    0.000     0.200   9.33843994e+01  -2.84941895e+02 -3.35147381e+00  -5.08605301e-01
    #    0.000     0.300   9.23945923e+01  -2.83375763e+02 -3.35068655e+00  -4.16197568e-01

    def parse_single_ffs(ffs_file: str) -> FarFieldEField:
        print(f"Processing {ffs_file}")
        with open(ffs_file, 'r') as f:
            for line in f:
                # Get frequency
                if line.startswith('// Radiated/Accepted/Stimulated Power , Frequency'):
                    next(f)
                    next(f)
                    next(f)
                    freq = float(next(f))
                    break
            # Get number of theta and phi samples
            for line in f:
                if line.startswith('// >> Total #phi samples, total #theta samples'):
                    phi_samples, theta_samples = map(int, next(f).split(' '))
                    break

            # allocate memory for the data
            result = FarFieldEField(
                freqs=np.array(freq),
                theta=np.zeros(theta_samples),
                phi=np.zeros(phi_samples),
                e_field_X_theta=np.zeros((theta_samples, phi_samples), dtype=complex),
                e_field_X_phi=np.zeros((theta_samples, phi_samples), dtype=complex),
                e_field_Y_theta=np.zeros((theta_samples, phi_samples), dtype=complex),
                e_field_Y_phi=np.zeros((theta_samples, phi_samples), dtype=complex)
            )
            # Data is theta major which means, theta changes faster than phi, we can reshape and transpose the data

            for line in f:
                if line.startswith('// >> Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi):'):
                    break
            idx = 0
            for line in f:
                if line.startswith('//'):
                    continue
                phi, theta, rtheta, itheta, rphi, iphi = map(float, line.split())
                # unravel the index
                phi_idx, theta_idx = np.unravel_index(idx, (phi_samples, theta_samples))
                result.theta[theta_idx] = theta
                result.phi[phi_idx] = phi
                result.e_field_X_theta[theta_idx, phi_idx] = rtheta + 1j * itheta
                result.e_field_X_phi[theta_idx, phi_idx] = rphi + 1j * iphi
                idx += 1

            # Set the Y-pol fields to be the same as the X-pol fields rotated by 90 degrees in the phi direction
            shift_amount = phi_samples // 4
            result.e_field_Y_theta[:] = np.roll(result.e_field_X_theta, shift_amount, axis=1)
            result.e_field_Y_phi[:] = np.roll(result.e_field_X_phi, shift_amount, axis=1)
            return result

    data = [parse_single_ffs(ffs_file) for ffs_file in xpol_ffs_files]
    data = jax.tree.map(lambda *x: np.stack(x, axis=0), *data)
    sort_idx = np.argsort(data.freqs)
    data = jax.tree.map(lambda x: x[sort_idx], data)

    if ypol_ffs_files is not None:
        y_data = [parse_single_ffs(ffs_file) for ffs_file in ypol_ffs_files]
        y_data = jax.tree.map(lambda *x: np.stack(x, axis=0), *y_data)
        sort_idx = np.argsort(y_data.freqs)
        y_data = jax.tree.map(lambda x: x[sort_idx], y_data)
        # Replace the rotated Y-pol fields with these data
        data.e_field_Y_theta[:] = y_data.e_field_X_theta
        data.e_field_Y_phi[:] = y_data.e_field_X_phi

    with h5py.File(h5_file, 'w') as hf:
        hf.create_dataset('theta_pts', data=data.theta[0])
        hf.create_dataset('phi_pts', data=data.phi[0])
        hf.create_dataset('freq_Hz', data=data.freqs)

        g1 = hf.create_group('X_pol_Efields')
        g1.create_dataset('etheta', data=data.e_field_X_theta)
        g1.create_dataset('ephi', data=data.e_field_X_phi)

        g2 = hf.create_group('Y_pol_Efields')
        g2.create_dataset('etheta', data=data.e_field_Y_theta)
        g2.create_dataset('ephi', data=data.e_field_Y_phi)


def test_convert_ffs_files_to_h5():
    ffs_files = glob.glob('/home/albert/data/dsa_assets/*.ffs')
    h5_file = 'converted.h5'
    import astropy.units as au
    # convert_ffs_files_to_h5(ffs_files, h5_file)
    print_h5_structure(h5_file)
    antenna_model = H5AntennaModelV1(
        angular_units=au.deg,
        freq_units=au.Hz,
        beam_file=h5_file
    )
    antenna_model.plot_polar_amplitude(0, 0, 0)
    antenna_model.plot_polar_amplitude(0, 0, 1)
    antenna_model.plot_polar_amplitude(0, 1, 0)
    antenna_model.plot_polar_amplitude(0, 1, 1)
