import os

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import itertools
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import source_model_registry, array_registry
from dsa2000_common.common.astropy_utils import mean_itrs, get_time_of_local_meridean
from dsa2000_common.common.fits_utils import ImageModel, save_image_to_fits
from dsa2000_common.common.logging import dsa_logger
from dsa2000_common.common.noise import calc_baseline_noise
from dsa2000_common.common.quantity_utils import quantity_to_np, time_to_jnp, quantity_to_jnp
from dsa2000_common.common.wgridder import image_to_vis_np, vis_to_image_np
from dsa2000_common.delay_models.uvw_utils import geometric_uvw_from_gcrs
from dsa2000_common.visibility_model.source_models.celestial.base_fits_source_model import \
    build_fits_source_model_from_wsclean_components
from dsa2000_fm.imaging.base_imagor import fit_beam
from dsa2000_fm.systematics.ionosphere import evolve_gcrs


@jax.jit
def compute_uvw(antennas_gcrs, time, ra0, dec0):
    antennas_uvw = geometric_uvw_from_gcrs(evolve_gcrs(antennas_gcrs, time), ra0, dec0)
    i_idxs, j_idxs = jnp.asarray(list(itertools.combinations(range(antennas_uvw.shape[0]), 2))).T
    # i_idxs, j_idxs = np.triu_indices(antennas_uvw.shape[0], k=1)
    uvw = antennas_uvw[i_idxs] - antennas_uvw[j_idxs]
    return uvw


def main(config_file, plot_folder, source_name, num_threads, duration, freq_block_size, spectral_line: bool,
         spectral_bandwidth: au.Quantity | None,
         with_noise: bool, with_earth_rotation: bool, with_freq_synthesis: bool, num_reduced_obsfreqs: int | None,
         num_reduced_obstimes: int | None):
    image_name_base = f"{source_name}_{os.path.basename(config_file).replace('.txt', '')}"
    plot_folder = os.path.join(plot_folder, image_name_base)
    os.makedirs(plot_folder, exist_ok=True)

    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa1650_9P'))

    x, y, z = [], [], []
    with open(config_file, 'r') as f:
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

    channel_width = array.get_channel_width()
    system_equivalent_flux_density = 3360 * au.Jy
    integration_time = array.get_integration_time()

    antennas: ac.EarthLocation
    array_location = mean_itrs(antennas.get_itrs()).earth_location
    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match(source_name)).get_wsclean_fits_files()
    source_model = build_fits_source_model_from_wsclean_components(
        wsclean_fits_files=wsclean_fits_files,
        model_freqs=array.get_channels(),
        full_stokes=False,
        repoint_centre=None,
        crop_box_size=None,
        num_facets_per_side=1
    )

    pointing = ac.ICRS(ra=source_model.ra[0, 0] * au.rad, dec=source_model.dec[0, 0] * au.rad)

    ref_time = get_time_of_local_meridean(coord=pointing, location=array_location,
                                          ref_time=at.Time("2025-06-10T00:00:00", format='isot', scale='utc'))

    obstimes = ref_time + np.arange(int(duration / integration_time)) * integration_time
    if num_reduced_obstimes is not None:
        times_before = len(obstimes)
        obstimes = obstimes[::len(obstimes) // num_reduced_obstimes]
        times_after = len(obstimes)
        integration_time *= times_before / times_after
        dsa_logger.info(f"Adjusted integration time: {integration_time}")

    obsfreqs = array.get_channels()
    if not spectral_line and num_reduced_obsfreqs is not None:
        chans_before = len(obsfreqs)
        obsfreqs = obsfreqs[::len(obsfreqs) // num_reduced_obsfreqs]
        chans_after = len(obsfreqs)
        channel_width *= chans_before / chans_after
        dsa_logger.info(f"Adjusted channel width: {channel_width}")

    if spectral_line:
        obsfreqs = np.asarray(source_model.model_freqs) * au.Hz
        assert spectral_bandwidth is not None
        channel_width = spectral_bandwidth

    baseline_noise = calc_baseline_noise(
        system_equivalent_flux_density=quantity_to_jnp(system_equivalent_flux_density, 'Jy'),
        chan_width_hz=quantity_to_jnp(channel_width, 'Hz'),
        t_int_s=quantity_to_jnp(integration_time, 's')
    )

    if not with_earth_rotation:
        obstimes = ref_time + np.arange(1) * integration_time
        simulated_noise_reduction = np.sqrt(float(integration_time / duration))
        dsa_logger.info(f"Simulated earth rotation noise reduction: {simulated_noise_reduction:.2f}")
        baseline_noise *= simulated_noise_reduction

    bandwidth = channel_width * len(obsfreqs)

    if not with_freq_synthesis:
        obsfreqs = np.mean(obsfreqs)[None]
        simulated_noise_reduction = np.sqrt(float(channel_width / bandwidth))
        dsa_logger.info(f"Simulated freq synth noise reduction: {simulated_noise_reduction:.2f}")
        baseline_noise *= simulated_noise_reduction

    if not with_noise:
        baseline_noise *= 0.

    dsa_logger.info(f"Baseline noise: {baseline_noise:.6f} Jy")

    dsa_logger.info(f"Observing {pointing} at {ref_time} (Transit).")

    antennas_gcrs = quantity_to_np(antennas.get_gcrs(ref_time).cartesian.xyz.T)

    # We grid(degrid(f) + noise)

    N = np.shape(antennas_gcrs)[0]

    dirty = np.array(source_model.image[0, 0, :, :], dtype=np.float32)
    dl = np.array(source_model.dl[0, 0])
    dm = np.array(source_model.dm[0, 0])
    ra0 = pointing.ra.rad
    dec0 = pointing.dec.rad
    freqs = quantity_to_np(obsfreqs, 'Hz')
    times = time_to_jnp(obstimes, ref_time)
    num_rows = N * (N - 1) // 2
    # output_vis_buffer = np.zeros((num_rows, freq_block_size), dtype=np.complex64, order='F')

    num_l, num_m = dirty.shape
    output_img_buffer = np.zeros((num_l, num_m), dtype=np.float32, order='F')
    output_img_accum = np.zeros((num_l, num_m), dtype=np.float32, order='F')

    output_psf_buffer = np.zeros((num_l, num_m), dtype=np.float32, order='F')
    output_psf_accum = np.zeros((num_l, num_m), dtype=np.float32, order='F')

    count = 0
    for t_idx in range(len(times)):
        uvw = np.array(compute_uvw(antennas_gcrs, times[t_idx], ra0, dec0), order='F')
        for nu_start_idx in range(0, len(freqs), freq_block_size):
            nu_end_idx = min(nu_start_idx + freq_block_size, len(freqs))
            vis = image_to_vis_np(
                uvw=uvw,
                freqs=freqs[nu_start_idx:nu_end_idx],
                dirty=dirty,
                pixsize_m=dm,
                pixsize_l=dl,
                center_m=0.,
                center_l=0.,
                mask=None,
                epsilon=1e-5,
                num_threads=num_threads,
                # output_buffer=output_vis_buffer
            )  # [B, 1]
            # Add noise
            noise = (baseline_noise / np.sqrt(2)) * (
                    np.random.normal(size=np.shape(vis)) + 1j * np.random.normal(size=np.shape(vis)))
            vis += noise.astype(np.complex64)  # sqrt(2) for real imag

            # Degrid into output accumulation image buffer

            vis_to_image_np(
                uvw=uvw,
                freqs=freqs[nu_start_idx:nu_end_idx],
                vis=vis,
                pixsize_l=dl,
                pixsize_m=dm,
                center_l=0.,
                center_m=0.,
                npix_m=num_m,
                npix_l=num_l,
                wgt=None,
                mask=None,
                epsilon=1e-5,
                double_precision_accumulation=False,
                scale_by_n=True,
                normalise=True,
                output_buffer=output_img_buffer,
                num_threads=num_threads
            )
            output_img_accum += output_img_buffer

            vis_to_image_np(
                uvw=uvw,
                freqs=freqs[nu_start_idx:nu_end_idx],
                vis=np.ones_like(vis),
                pixsize_l=dl,
                pixsize_m=dm,
                center_l=0.,
                center_m=0.,
                npix_m=num_m,
                npix_l=num_l,
                wgt=None,
                mask=None,
                epsilon=1e-5,
                double_precision_accumulation=False,
                scale_by_n=True,
                normalise=True,
                output_buffer=output_psf_buffer,
                num_threads=num_threads
            )
            output_psf_accum += output_psf_buffer

            count += 1
    output_img_accum /= count
    output_psf_accum /= count
    psf = output_psf_accum

    rad2arcsec = 3600 * 180 / np.pi

    major_beam, minor_beam, pa_beam = fit_beam(
        psf=psf,
        dl=dl * rad2arcsec,
        dm=dm * rad2arcsec
    )

    major_beam /= rad2arcsec
    minor_beam /= rad2arcsec

    dsa_logger.info(
        f"Beam major: {major_beam * rad2arcsec:.2f}arcsec, "
        f"minor: {minor_beam * rad2arcsec:.2f}arcsec, "
        f"posang: {pa_beam * 180 / np.pi:.2f}deg"
    )

    beam_solid_angle = np.pi / (4 * np.log(2)) * (major_beam * au.rad) * (minor_beam * au.rad)
    pixel_area = (dl * au.rad) * (dm * au.rad)
    pixel_per_beam = float(beam_solid_angle / pixel_area)

    # unit = 'JY/PIXEL'
    # output_img_buffer /= pixel_per_beam # convert to JY/PIXEL

    image_model = ImageModel(
        phase_center=pointing,
        obs_time=ref_time,
        dl=dl * au.rad,
        dm=dm * au.rad,
        freqs=np.mean(obsfreqs)[None],
        bandwidth=bandwidth,
        coherencies=('I',),
        beam_major=np.asarray(major_beam) * au.rad,
        beam_minor=np.asarray(minor_beam) * au.rad,
        beam_pa=np.asarray(pa_beam) * au.rad,
        unit='JY/BEAM',
        object_name=f'{image_name_base.upper()}',
        image=output_img_buffer[:, :, None, None] * au.Jy  # [num_l, num_m, 1, 1]
    )
    save_image_to_fits(os.path.join(plot_folder, f"{image_name_base}.fits"), image_model=image_model,
                       overwrite=True, radian_angles=True)

    dsa_logger.info(f"Image saved to {os.path.join(plot_folder, f'{image_name_base}.fits')}")

    image_model = ImageModel(
        phase_center=pointing,
        obs_time=ref_time,
        dl=dl * au.rad,
        dm=dm * au.rad,
        freqs=np.mean(obsfreqs)[None],
        bandwidth=bandwidth,
        coherencies=('I',),
        beam_major=np.asarray(major_beam) * au.rad,
        beam_minor=np.asarray(minor_beam) * au.rad,
        beam_pa=np.asarray(pa_beam) * au.rad,
        unit='JY/BEAM',
        object_name=f'{image_name_base.upper()}_PSF',
        image=psf[:, :, None, None] * au.Jy  # [num_l, num_m, 1, 1]
    )
    save_image_to_fits(os.path.join(plot_folder, f"{image_name_base}_psf.fits"), image_model=image_model,
                       overwrite=True, radian_angles=True)
    dsa_logger.info(f"PSF saved to {os.path.join(plot_folder, f'{image_name_base}_psf.fits')}")


if __name__ == '__main__':
    config_files = []
    # for prefix in ['a', 'e']:
    #     for res in ['2.61', '2.88', '3.14']:
    #         config_file = f'dsa1650_{prefix}_{res}.txt'
    #         config_files.append(config_file)
    # config_files.append('dsa1650_9P_e_optimal_v1.2.txt')

    config_files.append('dsa1650_9P_a_optimal_v1.2.txt')

    config_files.append('dsa1650_a_2.52_v2.2.txt')
    config_files.append('dsa1650_a_2.61_v2.2.txt')
    config_files.append('dsa1650_a_2.88_v2.2.txt')
    config_files.append('dsa1650_a_3.14_v2.2.txt')

    config_files.append('dsa1650_a_2.95_v2.3.txt')

    config_files.append('dsa1650_a_P279_v2.4.txt')
    config_files.append('dsa1650_a_P295_v2.4.txt')
    config_files.append('dsa1650_a_P305_v2.4.txt')

    config_files.append('dsa1650_a_P279_v2.4.1.txt')
    config_files.append('dsa1650_a_P295_v2.4.1.txt')
    config_files.append('dsa1650_a_P305_v2.4.1.txt')

    config_files.append('dsa1650_a_P279_v2.4.2.txt')
    config_files.append('dsa1650_a_P295_v2.4.2.txt')
    config_files.append('dsa1650_a_P305_v2.4.2.txt')

    config_files.append('dsa1650_a_P279_v2.4.3.txt')
    config_files.append('dsa1650_a_P295_v2.4.3.txt')
    config_files.append('dsa1650_a_P305_v2.4.3.txt')

    for num_reduced_obsfreqs, num_reduced_obstimes in [(100, 10), (200, 20), (400, 40)]:
        for with_noise in [False, True]:
            for config_file in config_files:
                if with_noise:
                    plot_folder = f'plots_{num_reduced_obsfreqs}freqs_{num_reduced_obstimes}times_with_noise'
                else:
                    plot_folder = f'plots_{num_reduced_obsfreqs}freqs_{num_reduced_obstimes}times_no_noise'
                main(
                    config_file=config_file,
                    plot_folder=plot_folder,
                    source_name='point_sources',
                    num_threads=os.cpu_count(),
                    freq_block_size=200,
                    duration=7 * au.min,
                    spectral_line=False,
                    spectral_bandwidth=None,
                    with_noise=with_noise,
                    with_earth_rotation=True,
                    with_freq_synthesis=True,
                    num_reduced_obsfreqs=num_reduced_obsfreqs,
                    num_reduced_obstimes=num_reduced_obstimes
                )

                main(
                    config_file=config_file,
                    plot_folder=plot_folder,
                    source_name='skamid_b1_1000h',
                    num_threads=os.cpu_count(),
                    freq_block_size=200,
                    duration=7 * au.min,
                    spectral_line=False,
                    spectral_bandwidth=None,
                    with_noise=with_noise,
                    with_earth_rotation=True,
                    with_freq_synthesis=True,
                    num_reduced_obsfreqs=num_reduced_obsfreqs,
                    num_reduced_obstimes=num_reduced_obstimes
                )

                main(
                    config_file=config_file,
                    plot_folder=plot_folder,
                    source_name='ncg_5194',
                    num_threads=os.cpu_count(),
                    freq_block_size=200,
                    duration=7 * au.min,
                    spectral_line=True,
                    spectral_bandwidth=5 * au.MHz,
                    with_noise=with_noise,
                    with_earth_rotation=True,
                    with_freq_synthesis=True,
                    num_reduced_obsfreqs=num_reduced_obsfreqs,
                    num_reduced_obstimes=num_reduced_obstimes
                )

                main(
                    config_file=config_file,
                    plot_folder=plot_folder,
                    source_name='meerkat_gc',
                    num_threads=os.cpu_count(),
                    freq_block_size=200,
                    duration=7 * au.min,
                    spectral_line=False,
                    spectral_bandwidth=None,
                    with_noise=with_noise,
                    with_earth_rotation=True,
                    with_freq_synthesis=True,
                    num_reduced_obsfreqs=num_reduced_obsfreqs,
                    num_reduced_obstimes=num_reduced_obstimes
                )
