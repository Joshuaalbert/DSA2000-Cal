import os

from dsa2000_common.common.ellipse_utils import Gaussian

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


def main(num_threads, duration, freq_block_size, spectral_line: bool,
         with_noise: bool, with_earth_rotation: bool, with_freq_synthesis: bool, num_reduced_obsfreqs: int | None,
         num_reduced_obstimes: int | None):

    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa1650_9P'))
    antennas = array.get_antennas()

    channel_width = array.get_channel_width()
    system_equivalent_flux_density = 3360 * au.Jy
    integration_time = array.get_integration_time()

    antennas: ac.EarthLocation
    array_location = mean_itrs(antennas.get_itrs()).earth_location
    pointing = ac.ICRS(0*au.deg, 0*au.deg)
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

    num_l = 512
    num_m = 512
    dl = dm = np.array(1*np.pi/180/3600)


    lvec = (-0.5 * num_l + np.arange(num_l)) * dl
    mvec = (-0.5 * num_m + np.arange(num_m)) * dm
    g = Gaussian(x0=np.array([0.,0.]), major_fwhm=dl*50, minor_fwhm=dm*50, pos_angle=0., total_flux=1.)
    L,M = np.meshgrid(lvec, mvec, indexing='ij')
    lm = np.stack((L,M), axis=-1)

    dirty = np.zeros((num_l, num_m), np.float32)
    dirty[:, :] = jax.vmap(jax.vmap(g.compute_flux_density))(lm) * dl**2

    assert np.sum(dirty) == 1 #  ==> units are Jy/pixel
    print(dirty.max()) # 0.00035301695 ==> This is the peak flux density in Jy/pixel

    ra0 = pointing.ra.rad
    dec0 = pointing.dec.rad
    freqs = quantity_to_np(obsfreqs, 'Hz')
    times = time_to_jnp(obstimes, ref_time)

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

    print(np.abs(output_img_buffer / pixel_per_beam).max()) # 0.0003417481 ==> This is the peak flux density in Jy/pixel
    print(np.abs(output_img_buffer).max()) # 0.01920634 ==> This is the peak flux density in Jy/beam

if __name__ == '__main__':
    main(
        num_threads=8,
        duration=7*au.min,
        freq_block_size=1,
        spectral_line=False,
        with_noise=False,
        with_earth_rotation=True,
        with_freq_synthesis=True,
        num_reduced_obsfreqs=1,
        num_reduced_obstimes=1
    )
