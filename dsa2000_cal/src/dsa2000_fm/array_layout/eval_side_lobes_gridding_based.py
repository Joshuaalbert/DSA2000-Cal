import os
from functools import partial

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import array_registry
from dsa2000_common.common.astropy_utils import create_spherical_spiral_grid, mean_itrs
from dsa2000_common.common.ellipse_utils import Gaussian
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.fits_utils import ImageModel, save_image_to_fits
from dsa2000_common.common.jax_utils import multi_vmap
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.noise import calc_image_noise, calc_baseline_noise
from dsa2000_common.common.one_factor import get_one_factors
from dsa2000_common.common.quantity_utils import quantity_to_jnp, time_to_jnp
from dsa2000_common.common.sum_utils import scan_sum
from dsa2000_common.delay_models.uvw_utils import geometric_uvw_from_gcrs, perley_lmn_from_icrs
from dsa2000_fm.array_layout.fast_psf_evaluation_evolving import compute_psf_from_gcrs
from dsa2000_fm.imaging.base_imagor import fit_beam
from dsa2000_fm.systematics.ionosphere import evolve_gcrs


def make_test_image(key, antennas: ac.EarthLocation,
                    transit_dec: au.Quantity, faint_peak_thermal_ratio,
                    obsfreqs: au.Quantity, channel_width: au.Quantity,
                    integration_time: au.Quantity, ref_time: at.Time,
                    obstimes: au.Quantity,
                    system_equivalent_flux_density: au.Quantity,
                    num_point_sources: int, num_gaussians: int,
                    pixel_size: au.Quantity,
                    fov: au.Quantity, gaussian_major: au.Quantity,
                    gaussian_minor: au.Quantity,
                    plot_folder: str, image_name: str):
    os.makedirs(plot_folder, exist_ok=True)
    bandwidth = channel_width * len(obsfreqs)
    total_int_time = integration_time * len(obstimes)
    array_location = mean_itrs(antennas.get_itrs()).earth_location

    dl = pixel_size
    num_pixels = (int(fov / dl) // 2) * 2

    pointing = ENU(0, 0, 1, obstime=ref_time, location=array_location).transform_to(ac.ICRS())
    pointing = ac.ICRS(ra=pointing.ra, dec=transit_dec)
    ra0 = quantity_to_jnp(pointing.ra, 'rad')
    dec0 = quantity_to_jnp(pointing.dec, 'rad')
    point_directions = create_spherical_spiral_grid(
        pointing=pointing,
        num_points=num_point_sources,
        inner_radius=0.05 * fov,
        angular_radius=0.5 * fov * 0.8
    )
    gaussian_directions = create_spherical_spiral_grid(
        pointing=pointing,
        num_points=num_gaussians,
        angular_radius=0.5 * fov * 0.8
    )

    image_thermal = calc_image_noise(
        system_equivalent_flux_density=quantity_to_jnp(system_equivalent_flux_density, 'Jy'),
        bandwidth_hz=quantity_to_jnp(bandwidth, 'Hz'),
        t_int_s=quantity_to_jnp(total_int_time, 's'),
        num_antennas=len(antennas),
        flag_frac=0.33
    )
    dsa_logger.info(f"Image noise: {image_thermal * 1e6:.2f} muJy")
    baseline_noise = calc_baseline_noise(
        system_equivalent_flux_density=quantity_to_jnp(system_equivalent_flux_density, 'Jy'),
        chan_width_hz=quantity_to_jnp(channel_width, 'Hz'),
        t_int_s=quantity_to_jnp(integration_time, 's')
    )
    dsa_logger.info(f"Baseline noise: {baseline_noise:.2f} Jy")

    N = len(antennas)

    peak_density = jnp.ones(num_gaussians) * (N / 4) * image_thermal * faint_peak_thermal_ratio
    major_gaussian = quantity_to_jnp(gaussian_major, 'rad') * jnp.ones(num_gaussians)
    minor_gaussian = quantity_to_jnp(gaussian_minor, 'rad') * jnp.ones(num_gaussians)
    pa_gaussian = jax.random.uniform(key, (num_gaussians,), minval=0., maxval=2 * np.pi)

    A_gaussian = Gaussian.total_flux_from_peak(peak_density, major_gaussian, minor_gaussian)
    A_point = jnp.ones(num_point_sources)
    ra_gaussian = quantity_to_jnp(gaussian_directions.ra, 'rad')
    dec_gaussian = quantity_to_jnp(gaussian_directions.dec, 'rad')
    ra_point = quantity_to_jnp(point_directions.ra, 'rad')
    dec_point = quantity_to_jnp(point_directions.dec, 'rad')

    antennas_gcrs = quantity_to_jnp(antennas.get_gcrs(ref_time).cartesian.xyz.T, 'm')
    times_jax = time_to_jnp(obstimes, ref_time)
    freqs_jax = quantity_to_jnp(obsfreqs, 'Hz')

    baseline_partitions = get_one_factors(len(antennas))
    baseline_partitions = jnp.asarray(baseline_partitions)

    img = compute_image_from_sources(
        key=key,
        baseline_partitions=baseline_partitions,
        ra_point=ra_point,
        dec_point=dec_point,
        A_point=A_point,
        ra_gaussian=ra_gaussian,
        dec_gaussian=dec_gaussian,
        A_gaussian=A_gaussian,
        major_gaussian=major_gaussian,
        minor_gaussian=minor_gaussian,
        pa_gaussian=pa_gaussian,
        antennas_gcrs=antennas_gcrs,
        times=times_jax,
        freqs=freqs_jax,
        baseline_noise=baseline_noise,
        ra0=ra0,
        dec0=dec0,
        dl=quantity_to_jnp(dl, 'rad'),
        num_pixels=num_pixels,
        accum_dtype=jnp.float32
    )
    img = np.asarray(img)

    lvec = (-0.5 * num_pixels + jnp.arange(num_pixels)) * quantity_to_jnp(dl, 'rad')
    L, M = jnp.meshgrid(lvec, lvec, indexing='ij')
    lmn = jnp.stack([L, M, jnp.sqrt(1. - L ** 2 - M ** 2)], axis=-1)  # [num_pixels, num_pixels, 3]

    psf = compute_psf_from_gcrs(
        antennas_gcrs=antennas_gcrs,
        ra=ra0,
        dec=dec0,
        lmn=lmn,
        times=times_jax,
        freqs=freqs_jax,
        with_autocorr=False,
        accumulate_dtype=jnp.float32
    )  # [num_pixels, num_pixels]

    major_beam, minor_beam, pa_beam = fit_beam(
        psf=psf,
        dl=quantity_to_jnp(dl, 'rad'),
        dm=quantity_to_jnp(dl, 'rad')
    )

    rad2arcsec = 3600 * 180 / np.pi
    dsa_logger.info(
        f"Beam major: {major_beam * rad2arcsec:.2f}arcsec, "
        f"minor: {minor_beam * rad2arcsec:.2f}arcsec, "
        f"posang: {pa_beam * 180 * np.pi:.2f}deg"
    )

    image_model = ImageModel(
        phase_center=pointing,
        obs_time=ref_time,
        dl=dl,
        dm=dl,
        freqs=np.mean(obsfreqs)[None],
        bandwidth=bandwidth,
        coherencies=('I',),
        beam_major=np.asarray(major_beam) * au.rad,
        beam_minor=np.asarray(minor_beam) * au.rad,
        beam_pa=np.asarray(pa_beam) * au.rad,
        unit='JY/PIXEL',
        object_name=f'DSA1650',
        image=img[:, :, None, None] * au.Jy  # [num_l, num_m, 1, 1]
    )
    save_image_to_fits(os.path.join(plot_folder, f"{image_name}.fits"), image_model=image_model,
                       overwrite=True)

    dsa_logger.info(f"Image saved to {os.path.join(plot_folder, f'{image_name}.fits')}")

    image_model = ImageModel(
        phase_center=pointing,
        obs_time=ref_time,
        dl=dl,
        dm=dl,
        freqs=np.mean(obsfreqs)[None],
        bandwidth=bandwidth,
        coherencies=('I',),
        beam_major=np.asarray(major_beam) * au.rad,
        beam_minor=np.asarray(minor_beam) * au.rad,
        beam_pa=np.asarray(pa_beam) * au.rad,
        unit='JY/PIXEL',
        object_name=f'DSA1650_PSF',
        image=psf[:, :, None, None] * au.Jy  # [num_l, num_m, 1, 1]
    )
    save_image_to_fits(os.path.join(plot_folder, f"{image_name}_psf.fits"), image_model=image_model,
                       overwrite=True)
    dsa_logger.info(f"PSF saved to {os.path.join(plot_folder, f'{image_name}.fits')}")


@partial(jax.jit, static_argnames=['num_pixels', 'accum_dtype'])
def compute_image_from_sources(
        key,
        baseline_partitions,
        ra_point,
        dec_point,
        A_point,
        ra_gaussian,
        dec_gaussian,
        A_gaussian,
        major_gaussian,
        minor_gaussian,
        pa_gaussian,
        antennas_gcrs,
        times,
        freqs,
        baseline_noise,
        ra0,
        dec0,
        dl,
        num_pixels: int,
        accum_dtype=jnp.float32
):
    if num_pixels % 2 != 0:
        raise ValueError("num_pixels must be even")

    lvec = (-0.5 * num_pixels + jnp.arange(num_pixels)) * dl
    L, M = jnp.meshgrid(lvec, lvec, indexing='ij')
    lmn = jnp.stack([L, M, jnp.sqrt(1. - L ** 2 - M ** 2)], axis=-1)  # [num_pixels, num_pixels, 3]
    lmn_point = jnp.stack(perley_lmn_from_icrs(ra_point, dec_point, ra0, dec0), axis=-1)  # [D, 3]
    lmn_gaussian = jnp.stack(perley_lmn_from_icrs(ra_gaussian, dec_gaussian, ra0, dec0), axis=-1)  # [E, 3]
    zero_accum = jnp.zeros(np.shape(lmn)[:-1], dtype=accum_dtype)

    adjoint_normalising_factor = np.shape(baseline_partitions)[0] * np.shape(baseline_partitions)[1] * np.shape(freqs)[
        0] * np.shape(times)[0]

    def accum_time(carry):
        key, time = carry
        antennas_uvw = geometric_uvw_from_gcrs(evolve_gcrs(antennas_gcrs, time), ra0, dec0)

        def accum_freq(carry):
            key, freq = carry
            wavelength = 299792458 / freq
            return compute_image_to_image(
                key=key,
                antenna_uvw=antennas_uvw / wavelength,
                baseline_partitions=baseline_partitions,
                lmn=lmn,
                lmn_point=lmn_point,
                lmn_gaussian=lmn_gaussian,
                major_gaussian=major_gaussian,
                minor_gaussian=minor_gaussian,
                pa_gaussian=pa_gaussian,
                A_gaussian=A_gaussian,
                A_point=A_point,
                baseline_noise=baseline_noise,
                adjoint_normalising_factor=adjoint_normalising_factor,
                accum_dtype=accum_dtype
            )

        accum = scan_sum(accum_freq, zero_accum, (jax.random.split(key, len(freqs)), freqs)) / len(freqs)
        return accum.astype(accum_dtype)

    accum = scan_sum(accum_time, zero_accum, (jax.random.split(key, len(times)), times)) / len(times)

    return accum


def compute_image_to_image(
        key, antenna_uvw, baseline_partitions,
        lmn,
        lmn_point, lmn_gaussian, major_gaussian, minor_gaussian,
        pa_gaussian, A_gaussian, A_point, baseline_noise, adjoint_normalising_factor,
        accum_dtype=jnp.float32
):
    """
    Compute the image from the image, going through sampling function.

    Args:
        key: PRNGKey
        antenna_uvw: [N, 3] the uvw coordinates of the antennas
        baseline_partitions: [M, B, 2] the M baseline partitions
        lmn: [..., 3] the lmn coordinates
        lmn_point: [D, 3] the lmn coordinates of the point sources
        lmn_gaussian: [E, 3] the lmn coordinates of the gaussian sources
        major_gaussian: [E] the major axis of the gaussian sources in radians
        minor_gaussian: [E] the minor axis of the gaussian sources in radians
        pa_gaussian: [E] the position angle of the gaussian sources in radians
        A_gaussian: [E] the amplitude of the gaussian sources in Jy
        A_point: [D] the amplitude of the point sources in Jy
        baseline_noise: the baseline noise in Jy
        accum_dtype: the dtype to use for accumulation

    Returns:
        [...] the image
    """
    zero_accum = jnp.zeros(lmn.shape[:-1], dtype=accum_dtype)

    num_partitions, B, _ = np.shape(baseline_partitions)

    def accum_partition(baseline_partition):
        antenna1 = baseline_partition[:, 0]
        antenna2 = baseline_partition[:, 1]
        uvw = antenna_uvw[antenna1] - antenna_uvw[antenna2]
        w = uvw[:, 2]

        # Point source vis
        delay_point = -2 * jnp.pi * (jnp.sum(uvw[:, None, :] * lmn_point, axis=-1) - w[:, None])  # [B, D]
        n_point = lmn_point[:, 2]
        vis_point = jnp.sum((A_point / n_point) * jax.lax.complex(jnp.cos(delay_point), jnp.sin(delay_point)),
                            axis=-1)  # [B]

        # Gaussian vis
        @partial(
            multi_vmap,
            in_mapping="[E,3],[E],[E],[E],[E],[B,3]",
            out_mapping="[B,E]",
            verbose=True
        )
        def single_gaussian(lmn_gaussian, major_gaussian, minor_gaussian, pa_gaussian, A_gaussian, uv_scaled):
            gaussian = Gaussian(
                x0=lmn_gaussian[:2],
                major_fwhm=major_gaussian,
                minor_fwhm=minor_gaussian,
                pos_angle=pa_gaussian,
                total_flux=A_gaussian
            )  # []
            return gaussian.fourier(uv_scaled)  # []

        vis_gaussian = single_gaussian(
            lmn_gaussian, major_gaussian, minor_gaussian, pa_gaussian,
            A_gaussian, uvw[:, :2]
        )  # [B,E]
        n_gaussian = lmn_gaussian[:, 2]
        phase = -2 * jnp.pi * w[:, None] * (n_gaussian - 1)
        w_kernel = jax.lax.complex(jnp.cos(phase), jnp.sin(phase)) / n_gaussian
        vis_gaussian = jnp.sum(w_kernel * vis_gaussian, axis=1)  # [B]
        vis_gaussian *= (num_partitions * B)  # since we divide by this later and fourier is exact

        # Sum and sample
        vis = vis_point + vis_gaussian
        real_key, imag_key = jax.random.split(key, 2)
        vis_noise = baseline_noise / np.sqrt(2) * jax.lax.complex(
            jax.random.normal(real_key, vis.shape, dtype=accum_dtype),
            jax.random.normal(imag_key, vis.shape, dtype=accum_dtype))
        vis += vis_noise

        # compute image
        delay_out = 2 * jnp.pi * (jnp.sum(uvw * lmn[..., None, :], axis=-1) - w)  # [..., B])
        img_out = jnp.sum(vis * jax.lax.complex(jnp.cos(delay_out), jnp.sin(delay_out)), axis=-1)  # [...]
        n_out = lmn[..., 2]
        img_out *= n_out
        return img_out.astype(accum_dtype) / (num_partitions * B)

    accum = scan_sum(accum_partition, zero_accum, baseline_partitions)  # [...]
    return accum


if __name__ == '__main__':
    np.random.seed(0)
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa1650_9P'))
    system_equivalent_flux_density = array.get_system_equivalent_flux_density()
    obsfreqs = array.get_channels()
    channel_width = array.get_channel_width()
    integration_time = array.get_integration_time()
    ref_time = at.Time("2025-06-10T00:00:00", scale='utc')
    obstimes = ref_time + np.arange(1) * integration_time

    x, y, z = [], [], []
    with open('pareto_opt_v6_e_2.61arcsec_lst_sq_v2/best_config_0380.txt', 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            _x, _y, _z = line.strip().split(',')
            x.append(float(_x))
            y.append(float(_y))
            z.append(float(_z))
        antennas = ac.EarthLocation.from_geocentric(
            x * au.m,
            y * au.m,
            z * au.m
        )

    make_test_image(
        key=jax.random.PRNGKey(0),
        antennas=antennas,
        faint_peak_thermal_ratio=10,
        transit_dec=0 * au.deg,
        obsfreqs=obsfreqs,
        channel_width=channel_width,
        integration_time=integration_time,
        ref_time=ref_time,
        obstimes=obstimes,
        system_equivalent_flux_density=system_equivalent_flux_density,
        num_point_sources=3,
        num_gaussians=10,
        fov=3 * au.arcmin,
        pixel_size=1 * au.arcsec,
        gaussian_major=40 * au.arcsec,
        gaussian_minor=30 * au.arcsec,
        plot_folder='test_make_image_plots',
        image_name='test_image'
    )
