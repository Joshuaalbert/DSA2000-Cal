import os.path
from concurrent.futures import ThreadPoolExecutor

import astropy.units as au
import numpy as np
import pylab as plt

from dsa2000_common.common.fourier_utils import ApertureTransform
from dsa2000_common.common.logging import dsa_logger
from dsa2000_fm.array_layout.optimal_transport import compute_uv_gaussian_width
from dsa2000_fm.imaging.base_imagor import fit_beam


def cored_equiripple_window_imp(x, R_core, R_total, attenuation_dB=30):
    """
    Flat-core equiripple taper on an arbitrary-shaped array.

    Parameters
    ----------
    x : array_like
        Positions (can be any shape)—we’ll compare them against R_core/R_total.
    R_core : float
        Radius where the window is exactly 1.
    R_total : float
        Radius where the window falls to 0.
    attenuation_dB : float, optional
        Desired sidelobe attenuation for the Chebyshev edge (default: 60 dB).

    Returns
    -------
    w : ndarray
        Same shape as `x`, with values in [0, 1].
    """
    alpha = np.arccosh(10 ** (attenuation_dB / 20))

    def chebyshev_window(x):
        dx = 0.5 * (x - R_core) / (R_total - R_core)
        return np.cosh(alpha * np.cos(np.pi * dx)) / np.cosh(alpha)

    return np.where(x < R_core, 1., chebyshev_window(x))


def cored_equiripple_window(x_lambda, alpha, R_lambda):
    window = cored_equiripple_window_imp(x_lambda, alpha * R_lambda, R_lambda)
    window = np.where(x_lambda > R_lambda, 0., window)
    return window


def tukey_window(x_lambda, alpha, R_lambda):
    """
    Flat out to +- alpha/2 * total_width, then taper to 0 at the edges with a cosine.

    Args:
        x_lambda: the x-axis values
        alpha: the width of the flat part of the window, as a fraction of total_width
        R_lambda: the total width of the window

    Returns:
        The Tukey window values for the given x-axis values.
    """

    # Calculate the width of the flat part
    flat_width = alpha * R_lambda * 2

    # Create the Tukey window

    # Flat part
    mask_flat = np.abs(x_lambda) <= flat_width / 2
    taper = (1 + np.cos(np.pi * (np.abs(x_lambda) - flat_width / 2) / (R_lambda - flat_width / 2))) / 2

    window = np.where(mask_flat, 1., taper)

    window = np.where(x_lambda > R_lambda, 0., window)

    return window


def find_side_lobe_levels(dist,
                          xvec_lambda,
                          dl: float,
                          n_side_lobes: int = 1,
                          res=32):
    X, Y = np.meshgrid(xvec_lambda, xvec_lambda, indexing='ij')
    xy = np.stack([X, Y], axis=-1)
    mask = dist != 0.
    xy = xy[mask]
    dist = dist[mask]
    radii = np.arange(res) * dl
    lm = np.stack([np.zeros(res), radii], axis=-1)
    radial_profile = 10 * np.log10(np.abs(dft_f(dist, xy, lm)))
    radial_profile -= radial_profile[0]

    # Find the first peak in the radial profile by looking for the first local maximum, take derivative
    # and find the first zero crossing

    grad = np.gradient(radial_profile)
    # Find the first zero crossing
    zero_crossings = np.where(np.diff(np.sign(grad)))[0]
    zero_crossings = zero_crossings[1:n_side_lobes * 2 + 1:2]  # skip the ne
    # zero_crossings = zero_crossings[1:n_side_lobes*2+1:2] # skip the ne
    radii = radii * 3600 * 180 / np.pi
    zero_crossing_radii = radii[zero_crossings]
    values_at_crossing = radial_profile[zero_crossings]
    return zero_crossing_radii, values_at_crossing, (radii, radial_profile)


def compute_target_dist(rho_lambda, gamma_lambda, R_lambda, underlying_type):
    norm = np.reciprocal(2 * np.pi * gamma_lambda ** 2)
    mask = rho_lambda < R_lambda
    dist = np.where(mask, norm * np.exp(-0.5 * rho_lambda ** 2 / gamma_lambda ** 2), 0.)
    if underlying_type == 'mod':
        cusp = norm * np.exp(-0.5 * R_lambda ** 2 / gamma_lambda ** 2)
        dist -= cusp
        dist /= np.max(dist)
        dist = np.maximum(dist, 0.)
    return dist


def compute_uv_profile(target_fwhm, alpha, underlying_type):
    freq = 1.4 * au.GHz
    c = 2.99792458e8 * au.m / au.s  # m/s
    wavelength = (c / freq).to('m')  # m
    wavelength_m = wavelength.value  # m

    gamma_m = compute_uv_gaussian_width(target_fwhm, freq).to('m').value
    gamma_lambda = gamma_m / wavelength_m  # lambda

    R_m = 16000.
    R_lambda = R_m / wavelength_m

    expand = 1.1
    total_width_m = R_m * 2 * expand
    total_width_lambda = float(total_width_m / wavelength_m)

    dx_m = 10.
    dx_lambda = float(dx_m / wavelength_m)  # lambda

    N = int(total_width_lambda / dx_lambda) // 2 * 2
    xvec_m = (-0.5 * N + np.arange(N)) * dx_m

    X, Y = np.meshgrid(xvec_m, np.zeros((1,)), indexing='ij')
    rho_m = np.sqrt(X ** 2 + Y ** 2)
    rho_lambda = rho_m / wavelength_m  # lambda

    dl = 1 / (N * dx_lambda)

    target_dist = compute_target_dist(rho_lambda, gamma_lambda, R_lambda, underlying_type)

    def compute_window(alpha):
        return tukey_window(rho_lambda, alpha, R_lambda)

    taper = compute_window(alpha)
    dist = taper * target_dist
    return xvec_m, dist.flatten(), target_dist.flatten()


def dft_f(f, xy, lm):
    output = []
    for i in range(lm.shape[0]):
        output.append(np.sum(f * np.exp(-2j * np.pi * np.sum(xy * lm[i], axis=-1))))
    return np.asarray(output)


def run_window(target_fwhm, alpha_num, underlying_type):
    """
    Test the tukey_window function.
    """

    freq = 1.4 * au.GHz
    c = 2.99792458e8 * au.m / au.s  # m/s
    wavelength = (c / freq).to('m')  # m
    # target_fwhm = 2.1 * au.arcsec
    wavelength_m = wavelength.value  # m

    gamma_m = compute_uv_gaussian_width(target_fwhm, freq).to('m').value
    gamma_lambda = gamma_m / wavelength_m  # lambda

    R_m = 16000.
    R_lambda = R_m / wavelength_m

    alpha = alpha_num / 16
    expand = 3
    total_width_m = R_m * 2 * expand
    total_width_lambda = float(total_width_m / wavelength_m)

    dx_m = 10.
    dx_lambda = float(dx_m / wavelength_m)  # lambda

    N = int(total_width_lambda / dx_lambda) // 2 * 2
    xvec_m = (-0.5 * N + np.arange(N)) * dx_m
    xvec_lambda = xvec_m / wavelength_m

    X, Y = np.meshgrid(xvec_m, xvec_m, indexing='ij')
    rho_m = np.sqrt(X ** 2 + Y ** 2)
    rho_lambda = rho_m / wavelength_m  # lambda

    dl = 1 / (N * dx_lambda)
    dl_arcsec = dl * 180 * 3600 / np.pi  # arcsec

    target_dist = compute_target_dist(rho_lambda, gamma_lambda, R_lambda, underlying_type)

    lvec = (-0.5 * 256 + np.arange(256)) * dl
    lvec_arcsec = (-0.5 * N + np.arange(N)) * dl_arcsec

    def compute_window(alpha):
        return tukey_window(rho_lambda, alpha, R_lambda)
        # return cored_equiripple_window(rho_lambda, alpha, R_lambda)

    profile_xvec, profile_dist, profile_dist_underlying = compute_uv_profile(target_fwhm, alpha, underlying_type)
    taper = compute_window(alpha)
    dist = taper * target_dist

    a = ApertureTransform()
    img = a.to_image(dist, axes=(-2, -1), dx=dx_lambda, dy=dx_lambda)
    img = np.abs(img)
    img /= np.max(img)

    major, minor, posang = fit_beam(img, dl, dl)
    major_arcsec = major * 3600 * 180 / np.pi
    minor_arcsec = minor * 3600 * 180 / np.pi

    img = 10 * np.log10(img)
    # rel_img = img - base
    # plt.close('all')
    zero_crossing_radii, zero_crossing_values, (radii, radial_profile) = find_side_lobe_levels(dist, xvec_lambda,
                                                                                               (0.125 * au.arcsec).to(
                                                                                                   'rad').value,
                                                                                               n_side_lobes=3,
                                                                                               res=64)

    dsa_logger.info(
        f"Underlying FWHM {target_fwhm} | alpha={alpha_num}/16 | Beam size: {major_arcsec:.2f} x {minor_arcsec:.2f} arcsec, PA: {posang:.2f} deg | sidelobe radii: {zero_crossing_radii} arcsec | sidelobe levels: {zero_crossing_values} dB")

    fig, all_axs = plt.subplots(1, 4, figsize=(24, 6), squeeze=False)

    axs = all_axs[0]

    im = axs[0].imshow(img.T, extent=(lvec_arcsec.min(), lvec_arcsec.max(), lvec_arcsec.min(), lvec_arcsec.max()),
                       origin='lower',
                       interpolation='none', cmap='jet', vmin=-40, vmax=0)
    fig.colorbar(im, ax=axs[0], label='Kernel Value (dB)')
    axs[0].set_title(f"FWHM={target_fwhm:.2f} | alpha={alpha_num}/16 | Beam size={major_arcsec:.2f} arcsec")
    axs[0].set_xlabel('l (arcsec)')
    axs[0].set_ylabel('m (arcsec)')
    axs[0].grid()
    axs[0].set_xlim(-16, 16)
    axs[0].set_ylim(-16, 16)

    horizontal_profile = img[N // 2, :]
    axs[1].plot(lvec_arcsec, horizontal_profile)
    axs[1].set_title('Radial Profile')
    axs[1].set_xlabel('l (arcsec)')
    axs[1].set_ylabel('Intensity (dB)')
    axs[1].grid()
    # axs[1].set_xlim(-16, 16)

    axs[2].plot(profile_xvec, profile_dist, label='Tapered Distribution')
    axs[2].plot(profile_xvec, profile_dist_underlying, label='Underlying Distribution')
    axs[2].set_title('UV Distribution')
    axs[2].set_xlabel('u (m)')
    axs[2].set_ylabel('Distribution Value')
    axs[2].grid()
    axs[2].legend()

    axs[3].plot(radii, radial_profile)
    axs[3].set_title('Radial Profile')
    axs[3].set_xlabel('l (arcsec)')
    axs[3].set_ylabel('Intensity (dB)')
    for r, v in zip(zero_crossing_radii, zero_crossing_values):
        axs[3].axvline(r, color='red')
        axs[3].axhline(v, color='red')
    axs[3].grid()

    fig.tight_layout()
    if underlying_type == 'mod':
        fig_name = f"mod_tukey_window_fwhm{target_fwhm.value:.2f}_alpha{alpha_num}_16.png"
    else:
        fig_name = f"tukey_window_fwhm{target_fwhm.value:.2f}_alpha{alpha_num}_16.png"
    fig.savefig(fig_name, dpi=300)
    plt.close(fig)
    return fig_name, major_arcsec, zero_crossing_values


def search(num_iterations: int, trials_per_iter):
    for iteration in range(num_iterations):
        tasks = []
        for _ in range(trials_per_iter):
            tasks.append(
                ( # 2.45,,14.62
                    #2.58,,13.89,2.69,-20.00,-21.96,-23.44
                    # 2.41,mod,14.91,2.74,-21.87,-22.07,-24.66
                    np.random.uniform(2.0, 3.5) * au.arcsec,
                    np.random.uniform(9, 16),
                    np.random.choice(['mod', ''])
                )
            )
        results = []
        if trials_per_iter == 1:
            results = [run_window(*tasks[0])]
        else:
            # 2) spawn a pool of 32 threads
            with ThreadPoolExecutor(max_workers=trials_per_iter) as executor:
                # executor.map will schedule run_window(*args) for each tuple in tasks
                # it returns results in order
                for result in executor.map(lambda args: run_window(*args), tasks):
                    results.append(result)

        for result, task in zip(results, tasks):
            fig_name, major_arcsec, zero_crossing_values = result
            target_fwhm, alpha_num, underlying_type = task
            if not os.path.exists('results_highres.txt'):
                with open("results_highres.txt", "w") as f:
                    # Create the file and write the header
                    f.write("#underlying_fwhm,underlying_type,alpha_num,beam_size,sidelobe_1,sidelobe_2,sidelobe_2\n")
            with open("results_highres.txt", "a") as f:
                # Append all results to the file
                if len(zero_crossing_values) < 3:
                    zero_crossing_values = np.concatenate(
                        [zero_crossing_values, np.zeros(3 - len(zero_crossing_values))])
                f.write(
                    f"{target_fwhm.value:.2f},{underlying_type},{alpha_num:.2f},{major_arcsec:.2f},{zero_crossing_values[0]:.2f},{zero_crossing_values[1]:.2f},{zero_crossing_values[2]:.2f}\n"
                )


def explore():
    import pylab as plt

    with open('results_highres.txt', 'r') as f:
        fwhm, _type, alpha_num, beam_size, sidelobe_1, sidelobe_2, sidelobe_3 = [], [], [], [], [], [], []
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split(',')

            if float(parts[4]) < float(parts[5]):
                continue
            if float(parts[6]) == 0:
                continue
            fwhm.append(float(parts[0]))
            _type.append(parts[1])
            alpha_num.append(float(parts[2]))
            beam_size.append(float(parts[3]))
            sidelobe_1.append(float(parts[4]))
            sidelobe_2.append(float(parts[5]))
            sidelobe_3.append(float(parts[6]))

        lines = list(zip(fwhm, _type, alpha_num, beam_size, sidelobe_1, sidelobe_2, sidelobe_3))
        lines = sorted(lines, key=lambda x: x[4])

        for line in lines:
            if 3.0 < line[3] < 3.2 and line[4] < -20:
                print(line)

    plt.scatter(beam_size, sidelobe_1, s=1)
    plt.xlabel('Beam Size (arcsec)')
    plt.ylabel('Sidelobe (dB)')
    plt.title('1st Sidelobe vs Resolution')
    plt.grid()
    plt.savefig('beam_size_vs_sidelobe.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    search(num_iterations=1000, trials_per_iter=32)
