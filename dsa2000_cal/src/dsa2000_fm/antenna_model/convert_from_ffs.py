import glob

import numpy as np


def convert_ffs_to_ffs(cst_files):
    def func(x):
        return float(x.split('=')[-1].split(')')[0])

    all_files = sorted(glob.glob(cst_files + '*'), key=func)
    freq = [func(i) for i in all_files]
    phi, theta = np.genfromtxt(all_files[0], skip_header=31, unpack=True, usecols=(0, 1))
    gain, etheta, ptheta, ephi, pphi = np.zeros((5, len(all_files), len(theta)))
    theta_p = 1801
    phi_p = 3601

    for i, file in enumerate(all_files):
        etheta[i], ptheta[i], ephi[i], pphi[i] = np.genfromtxt(file, skip_header=31, unpack=True, usecols=(2, 3, 4, 5))

    theta = theta.reshape(phi_p, theta_p)[0]
    phi = phi.reshape(phi_p, theta_p)[:, 0]
    # gain = np.transpose(gain.reshape(len(all_files),phi_p,theta_p),(0,2,1))
    etheta = np.transpose(etheta.reshape(len(all_files), phi_p, theta_p), (0, 2, 1))
    ptheta = np.transpose(ptheta.reshape(len(all_files), phi_p, theta_p), (0, 2, 1))
    ephi = np.transpose(ephi.reshape(len(all_files), phi_p, theta_p), (0, 2, 1))
    pphi = np.transpose(pphi.reshape(len(all_files), phi_p, theta_p), (0, 2, 1))

    # with h5py.File('/data05/nmahesh/DSA2000/DSA2000-beam.h5', 'w') as hf:
    # g1 = hf.create_group('X_pol_Efields')
    #     g1.create_dataset('etheta', data=rtheta_cx + 1j * itheta_cx)
    #     g1.create_dataset('ephi', data=rphi_cx + 1j * iphi_cx)
    #     hf.create_dataset('theta_pts', data=theta_c)
    #     hf.create_dataset('phi_pts', data=phi_c)
    #     hf.create_dataset('freq_Hz', data=freq)
    #
    #     g2 = hf.create_group('Y_pol_Efields')
    #     g2.create_dataset('etheta', data=rtheta_cy + 1j * itheta_cy)
    #     g2.create_dataset('ephi', data=rphi_cy + 1j * iphi_cy)
    #
    #     g3 = hf.create_group('Metadata')
    #     for key, item in metadata.items():
    #         print(key, item)
    #         g3.create_dataset(key, data=[item])
    return np.array(freq), theta, phi, etheta, ptheta, ephi, pphi

    #
