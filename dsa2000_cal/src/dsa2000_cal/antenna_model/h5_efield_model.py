import dataclasses
import os

import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from astropy import units as au

from dsa2000_cal.antenna_model.antenna_beam import AltAzAntennaModel


def convert_spherical_e_field_to_cartesian(e_phi, e_theta, phi, theta):
    """
    Convert the spherical electric field components to cartesian (linear) components.

    Args:
        e_phi: the complex electric field component in the phi-hat direction.
        e_theta: the complex electric field component in the theta-hat direction.
        phi: the azimuthal angle in radians, in the range [0, 2*pi].
        theta: the polar angle in radians, in the range [0, pi].

    Returns:
        The cartesian electric field components.
    """
    # Calculate the cartesian electric field components
    e_x = e_theta * np.cos(theta) * np.cos(phi) - e_phi * np.sin(phi)
    e_y = e_theta * np.cos(theta) * np.sin(phi) + e_phi * np.cos(phi)
    e_z = -e_theta * np.sin(theta)
    return e_x, e_y, e_z


@dataclasses.dataclass(eq=False)
class H5AntennaModelV1(AltAzAntennaModel):
    beam_file: str
    angular_units: au.Unit = au.rad
    freq_units: au.Unit = au.Hz

    def get_amplitude(self) -> au.Quantity:
        return self.amplitude

    def get_phase(self) -> au.Quantity:
        return self.phase

    def get_voltage_gain(self) -> au.Quantity:
        return self.voltage_gain

    def get_freqs(self) -> au.Quantity:
        return self.freqs

    def get_theta(self) -> au.Quantity:
        return self.theta

    def get_phi(self) -> au.Quantity:
        return self.phi

    def plot_e_field(self, nu: int = 0):
        if not os.path.exists(self.beam_file):
            raise ValueError(f"Antenna model file {self.beam_file} does not exist")
        print_h5_structure(self.beam_file)
        with (tb.open_file(self.beam_file, 'r') as file):
            # self.freqs = file.get_node("/Freq(Hz)").read() * au.Hz
            freqs = (file.root.freq_Hz.read() * self.freq_units).to('Hz')  # [num_freqs]
            theta = (file.root.theta_pts.read() * self.angular_units).to('rad')  # [num_theta]
            phi = (file.root.phi_pts.read() * self.angular_units).to('rad')  # [num_phi]

            # e_field_X_theta = file.get_node("/X-pol_Efields/etheta").read()  # [freq, theta, phi]
            e_field_X_theta = file.root.X_pol_Efields.etheta.read()  # [freq, theta, phi]
            e_field_X_theta = np.transpose(e_field_X_theta, (1, 2, 0))  # [theta, phi, freq]
            # e_field_X_phi = file.get_node("/X-pol_Efields/ephi").read()  # [freq, theta, phi]
            e_field_X_phi = file.root.X_pol_Efields.ephi.read()  # [freq, theta, phi]
            e_field_X_phi = np.transpose(e_field_X_phi, (1, 2, 0))  # [theta, phi, freq]

            # e_field_Y_theta = file.get_node("/Y-pol_Efields/etheta").read()  # [freq, theta, phi]
            e_field_Y_theta = file.root.Y_pol_Efields.etheta.read()  # [freq, theta, phi]
            e_field_Y_theta = np.transpose(e_field_Y_theta, (1, 2, 0))  # [theta, phi, freq]
            # e_field_Y_phi = file.get_node("/Y-pol_Efields/ephi").read()  # [freq, theta, phi]
            e_field_Y_phi = file.root.Y_pol_Efields.ephi.read()  # [freq, theta, phi]
            e_field_Y_phi = np.transpose(e_field_Y_phi, (1, 2, 0))  # [theta, phi, freq]

            e_field_X_total = np.log10(np.sqrt(np.abs(e_field_X_theta) ** 2 + np.abs(e_field_X_phi) ** 2))
            e_field_Y_total = np.log10(np.sqrt(np.abs(e_field_Y_theta) ** 2 + np.abs(e_field_Y_phi) ** 2))

            fig, axs = plt.subplots(2, 1, figsize=(10, 10), subplot_kw={'projection': 'polar'})

            # theta_mask = theta <= 1. * au.rad
            # theta = theta[theta_mask]
            # e_field_X_total = e_field_X_total[theta_mask, ...]
            # e_field_Y_total = e_field_Y_total[theta_mask, ...]

            # Convert theta and phi to meshgrid for pcolormesh
            THETA, PHI = np.meshgrid(theta.to('rad').value, phi.to('rad').value, indexing='ij')

            # X-pol E-field
            im = axs[0].pcolormesh(PHI, THETA, e_field_X_total[..., nu], shading='auto', ec='face', cmap='inferno')
            cbar = fig.colorbar(im, ax=axs[0], label='Log10 Amplitude')
            axs[0].set_title(f"X-pol E-field Amplitude at {freqs[nu]}")
            axs[0].set_theta_zero_location('N')  # Set 0 degrees at the top
            axs[0].set_theta_direction(-1)  # Make clockwise
            axs[0].grid(True)

            # Y-pol E-field
            im = axs[1].pcolormesh(PHI, THETA, e_field_Y_total[..., nu], shading='auto', ec='face', cmap='inferno')
            cbar = fig.colorbar(im, ax=axs[1], label='Log10 Amplitude')
            axs[1].set_title(f"Y-pol E-field Amplitude at {freqs[nu]}")
            axs[1].set_theta_zero_location('N')  # Set 0 degrees at the top
            axs[1].set_theta_direction(-1)  # Make clockwise
            axs[1].grid(True)

            plt.tight_layout()
            plt.show()

    def __post_init__(self):
        if not os.path.exists(self.beam_file):
            raise ValueError(f"Antenna model file {self.beam_file} does not exist")
        print_h5_structure(self.beam_file)
        with tb.open_file(self.beam_file, 'r') as file:
            # self.freqs = file.get_node("/Freq(Hz)").read() * au.Hz
            self.freqs = (file.root.freq_Hz.read() * self.freq_units).to('Hz')  # [num_freqs]
            self.theta = (file.root.theta_pts.read() * self.angular_units).to('rad')  # [num_theta]
            self.phi = (file.root.phi_pts.read() * self.angular_units).to('rad')  # [num_phi]

            # e_field_X_theta = file.get_node("/X-pol_Efields/etheta").read()  # [freq, theta, phi]
            e_field_X_theta = file.root.X_pol_Efields.etheta.read()  # [freq, theta, phi]
            e_field_X_theta = np.transpose(e_field_X_theta, (1, 2, 0))  # [theta, phi, freq]
            # e_field_X_phi = file.get_node("/X-pol_Efields/ephi").read()  # [freq, theta, phi]
            e_field_X_phi = file.root.X_pol_Efields.ephi.read()  # [freq, theta, phi]
            e_field_X_phi = np.transpose(e_field_X_phi, (1, 2, 0))  # [theta, phi, freq]

            # e_field_Y_theta = file.get_node("/Y-pol_Efields/etheta").read()  # [freq, theta, phi]
            e_field_Y_theta = file.root.Y_pol_Efields.etheta.read()  # [freq, theta, phi]
            e_field_Y_theta = np.transpose(e_field_Y_theta, (1, 2, 0))  # [theta, phi, freq]
            # e_field_Y_phi = file.get_node("/Y-pol_Efields/ephi").read()  # [freq, theta, phi]
            e_field_Y_phi = file.root.Y_pol_Efields.ephi.read()  # [freq, theta, phi]
            e_field_Y_phi = np.transpose(e_field_Y_phi, (1, 2, 0))  # [theta, phi, freq]

            E_x_X_dipole, E_y_X_dipole, _ = convert_spherical_e_field_to_cartesian(
                *np.broadcast_arrays(e_field_X_phi, e_field_X_theta, self.phi[None, :, None], self.theta[:, None, None])
            )  # [theta, phi, freq]
            E_x_Y_dipole, E_y_Y_dipole, _ = convert_spherical_e_field_to_cartesian(
                *np.broadcast_arrays(e_field_Y_phi, e_field_Y_theta, self.phi[None, :, None], self.theta[:, None, None])
            )  # [theta, phi, freq]

            jones = np.transpose(np.asarray([[E_x_X_dipole, E_y_X_dipole],
                                             [E_x_Y_dipole, E_y_Y_dipole]]),
                                 (2, 3, 4, 0, 1))  # [theta, phi, freq, 2, 2]

            self.amplitude = np.abs(jones) * au.dimensionless_unscaled
            self.phase = np.angle(jones) * au.rad
            self.voltage_gain = np.max(np.max(self.amplitude[..., 0, 0], axis=0),
                                       axis=0) * au.dimensionless_unscaled  # [num_freqs]


def get_pytable_by_path(h5file_path, path):
    """
    Get a PyTables node (group or leaf) by its path in the HDF5 file.

    :param h5file_path: The path to the HDF5 file.
    :param path: The path to the node in the HDF5 file.
    :return: The PyTables node at the specified path.
    """
    # Open the HDF5 file
    with tb.open_file(h5file_path, 'r') as file:
        # Get the node at the specified path
        node = file.get_node(path)
        return node


def print_h5_structure(h5file_path):
    """
    Print the structure of a PyTables HDF5 file, including groups, leaf shapes, and data types.

    :param h5file_path: The path to the HDF5 file.
    """
    # Open the HDF5 file
    with tb.open_file(h5file_path, 'r') as file:
        def traverse_node(node, path=''):
            """
            Recursively traverse nodes (groups and leaves) in the HDF5 file.

            :param node: The current node (group or leaf) in the HDF5 file.
            :param path: The accumulated path to the current node.
            """
            # Check if the node is a group
            if isinstance(node, tb.Group):
                # Print the path of the group
                print(f'Group: {node._v_pathname}')
                # Traverse the children of the group
                for child in node._f_iter_nodes():
                    traverse_node(child, path + node._v_name + '/')
            # Check if the node is a leaf (dataset)
            elif isinstance(node, tb.Leaf):
                # Handle UnImplemented objects specially
                if isinstance(node, tb.UnImplemented):
                    print(f'UnImplemented Leaf: {node._v_pathname} | Reason: Unsupported type or dataset')
                else:
                    # Print the leaf information including shape and dtype
                    print(f'Leaf: {node._v_pathname} | Shape: {node.shape} | Dtype: {node.dtype} | Min: {np.min(node)} | Max: {np.max(node)}')
            # Handle UnImplemented specifically if not already caught
            elif isinstance(node, tb.UnImplemented):
                print(f'UnImplemented Node: {node._v_pathname} | Reason: Unsupported type or dataset')

        # Start the traversal from the root node
        traverse_node(file.root)
