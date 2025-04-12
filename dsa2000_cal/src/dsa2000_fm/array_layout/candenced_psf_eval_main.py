import os

from dsa2000_common.common.logging import dsa_logger

os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np

from dsa2000_common.common.astropy_utils import mean_itrs
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.common.fits_utils import ImageModel, save_image_to_fits
from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_common.common.ray_utils import TimerLog
from dsa2000_fm.array_layout.fast_psf_evaluation_evolving import compute_psf_from_gcrs
from dsa2000_fm.imaging.base_imagor import fit_beam

compute_psf_from_gcrs_jit = jax.jit(compute_psf_from_gcrs, static_argnames=['with_autocorr', 'accumulate_dtype'])


def main(save_folder, save_name, array_config, fov, pixel_size, transit_dec: au.Quantity, num_times: int,
         dt: au.Quantity,
         num_freqs: int):
    os.makedirs(save_folder, exist_ok=True)
    # fill_registries()
    # array = array_registry.get_instance(array_registry.get_match(array_name))
    x, y, z = [], [], []
    with open(array_config, 'r'):
        for line in open(array_config, 'r'):
            if line.startswith('#'):
                continue
            line = line.split(',')
            x.append(float(line[0]))
            y.append(float(line[1]))
            z.append(float(line[2]))
    antennas = ac.EarthLocation(x * au.m, y * au.m, z * au.m)
    array_location = mean_itrs(antennas.get_itrs()).earth_location

    obstime = at.Time('2025-06-10T16:00:00', scale='utc')

    antennas_gcrs = quantity_to_jnp(antennas.get_gcrs(obstime=obstime).cartesian.xyz.T)
    n = int(fov / pixel_size)
    if n % 2 == 1:
        n += 1

    lvec = mvec = (-n / 2 + np.arange(n)) * pixel_size.to(au.rad).value
    L, M = np.meshgrid(lvec, lvec, indexing='ij')
    N = np.sqrt(1 - L ** 2 - M ** 2)
    lmn = jnp.stack([L.flatten(), M.flatten(), N.flatten()], axis=-1)
    freqs = np.linspace(700e6, 2000e6, num_freqs) * au.Hz
    freqs_jax = quantity_to_jnp(freqs, 'Hz')

    zenith = ENU(0, 0, 1, obstime=obstime, location=array_location).transform_to(ac.ICRS())

    times = jnp.arange(num_times) * quantity_to_jnp(dt, 's')

    with TimerLog("PSF evaluation"):
        # Compute the PSF
        psf = np.asarray(
            jax.block_until_ready(
                compute_psf_from_gcrs_jit(
                    antennas_gcrs=antennas_gcrs,
                    ra=quantity_to_jnp(zenith.ra, 'rad'),
                    dec=quantity_to_jnp(transit_dec, 'rad'),
                    lmn=lmn,
                    times=times,
                    freqs=freqs_jax,
                    with_autocorr=True,
                    accumulate_dtype=jnp.float32
                ).reshape(L.shape)
            )
        )

    with TimerLog("Fitting beam and saving"):
        major, minor, posang = fit_beam(
            psf=psf,
            dl=quantity_to_jnp(pixel_size, 'rad'),
            dm=quantity_to_jnp(pixel_size, 'rad')
        )
        dsa_logger.info(
            f"Beam fit: {major * 3600 * 180 / np.pi:.2f}arcsec, {minor * 3600 * 180 / np.pi:.2f}arcsec, {posang * 180 / np.pi:.2f}dec")

        image_model = ImageModel(
            phase_center=ac.ICRS(zenith.ra, transit_dec),
            obs_time=obstime,
            dl=pixel_size,
            dm=pixel_size,
            freqs=np.mean(freqs)[None],
            bandwidth=(freqs[-1] - freqs[0]),
            coherencies=('I',),
            beam_major=np.asarray(major) * au.rad,
            beam_minor=np.asarray(minor) * au.rad,
            beam_pa=np.asarray(posang) * au.rad,
            unit='JY/PIXEL',
            object_name=f'{save_name}_PSF',
            image=psf[:, :, None, None] * au.Jy  # [num_l, num_m, 1, 1]
        )
        save_image_to_fits(
            file_path=os.path.join(save_folder, f'{save_name}_psf.fits'),
            image_model=image_model,
            overwrite=True
        )


if __name__ == '__main__':
    for prefix in ['e', 'f', 'g', 'h', 'full', 'a', 'b', 'c', 'd']:
        for transit_dec in [0, -30, 30, 60, 90] * au.deg:
            for cadence in [1, 2, 4, 8]:
                save_name = f"dsa1650_9P_{prefix}_optimal_v1_cadence_{cadence}_dec_{transit_dec.value}"
                with TimerLog(f"Working on {save_name}"):
                    main(
                        save_folder='cadenced_psfs',
                        save_name=save_name,
                        array_config=f"pareto_opt_solution_{prefix}/dsa1650_9P_{prefix}_optimal_v1.txt",
                        pixel_size=0.8 * au.arcsec,
                        fov=3 * au.arcmin,
                        transit_dec=transit_dec,
                        num_times=420 * cadence,
                        dt=1.5 * au.s,
                        num_freqs=10000
                    )
