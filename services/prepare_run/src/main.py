import os
import shutil
import subprocess

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au

from dsa2000_cal.assets.content_registry import fill_registries

fill_registries()

from dsa2000_cal.assets.arrays.array import AbstractArray
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.astropy_utils import mean_itrs
from dsa2000_cal.bbs_sky_model import create_sky_model
from dsa2000_cal.create_ms_cfg import create_makems_config
from dsa2000_cal.faint_sky_model import transform_to_wsclean_model
from dsa2000_cal.models.run_config import RunConfig, PrepareRunConfig


def main(prepare_run_config: PrepareRunConfig):
    # Create pointing direction
    dsa_array: AbstractArray = array_registry.get_instance(array_registry.get_match(prepare_run_config.array_name))
    antennas = dsa_array.get_antennas()
    array_centre = mean_itrs(antennas)
    location = array_centre.earth_location

    frame = ac.AltAz(
        obstime=at.Time(prepare_run_config.start_dt),
        location=location
    )
    pointing_centre = ac.SkyCoord(
        alt=prepare_run_config.alt_deg * au.deg,
        az=prepare_run_config.az_deg * au.deg,
        frame=frame
    ).transform_to(ac.ICRS())
    pointing_centre = ac.ICRS(ra=pointing_centre.ra, dec=pointing_centre.dec)

    # Create bright sky model
    if prepare_run_config.num_bright_sources > 0:
        bright_sky_model_bbs = os.path.abspath('bright_sky_model.txt')
        # Create bright sky model
        create_sky_model(
            filename=bright_sky_model_bbs,
            num_sources=prepare_run_config.num_bright_sources,
            spacing_deg=prepare_run_config.spacing_deg,
            pointing_centre=pointing_centre
        )
    else:
        bright_sky_model_bbs = None

    # Rotate faint sky model
    if prepare_run_config.faint_sky_model_fits is not None:
        faint_sky_model_fits = os.path.abspath(prepare_run_config.faint_sky_model_fits)
        # Rotate faint sky model
        transform_to_wsclean_model(
            fits_file=faint_sky_model_fits,
            output_file=faint_sky_model_fits,
            pointing_centre=pointing_centre,
            ref_freq_hz=prepare_run_config.start_freq_hz,
            bandwidth_hz=prepare_run_config.channel_width_hz * prepare_run_config.num_channels
        )
    else:
        faint_sky_model_fits = None

    makems_config_file = create_makems_config(
        array_name=prepare_run_config.array_name,
        ms_name='visibilities.ms',
        start_freq=prepare_run_config.start_freq_hz,
        step_freq=prepare_run_config.channel_width_hz,
        start_time=prepare_run_config.start_dt,
        step_time=prepare_run_config.integration_time_s,
        pointing_direction=pointing_centre,
        num_freqs=prepare_run_config.num_channels,
        num_times=prepare_run_config.num_times
    )

    # Run `makems ${makems_config_file}`
    completed_process = subprocess.run(["makems", makems_config_file])
    if completed_process.returncode != 0:
        raise RuntimeError(f"makems failed with return code {completed_process.returncode}")

    # Copy visibilties.ms to outputs
    dft_ms_file = os.path.abspath('dft_visibilities.ms')
    fft_ms_file = os.path.abspath('fft_visibilities.ms')
    rfi_ms_file = os.path.abspath('rfi_visibilities.ms')
    output_ms_file = os.path.abspath('visibilities.ms')
    shutil.copytree('visibilities.ms_p0', dft_ms_file)
    shutil.copytree('visibilities.ms_p0', fft_ms_file)
    shutil.copytree('visibilities.ms_p0', rfi_ms_file)
    shutil.move('visibilities.ms_p0', output_ms_file)

    ionosphere_h5parm = os.path.abspath('ionosphere.h5parm')
    ionosphere_fits = os.path.abspath('ionosphere.fits')
    quartical_h5parm = os.path.abspath('quartical.h5parm')
    quartical_fits = os.path.abspath('quartical.fits')
    beam_h5parm = os.path.abspath('beam.h5parm')
    beam_fits = os.path.abspath('beam.fits')

    image_prefix = 'image'

    run_config = RunConfig(
        array_name=prepare_run_config.array_name,
        start_dt=prepare_run_config.start_dt,
        pointing_centre=pointing_centre,
        bright_sky_model_bbs=bright_sky_model_bbs,
        faint_sky_model_fits=faint_sky_model_fits,
        start_freq_hz=prepare_run_config.start_freq_hz,
        channel_width_hz=prepare_run_config.channel_width_hz,
        num_channels=prepare_run_config.num_channels,
        num_times=prepare_run_config.num_times,
        integration_time_s=prepare_run_config.integration_time_s,
        ionosphere_specification=prepare_run_config.ionosphere_specification,
        ionosphere_h5parm=ionosphere_h5parm,
        ionosphere_fits=ionosphere_fits,
        visibilities_path=output_ms_file,
        dft_visibilities_path=dft_ms_file,
        fft_visibilities_path=fft_ms_file,
        rfi_visibilities_path=rfi_ms_file,
        rfi_sim_config=prepare_run_config.rfi_sim_config,
        beam_h5parm=beam_h5parm,
        beam_fits=beam_fits,
        quartical_h5parm=quartical_h5parm,
        quartical_fits=quartical_fits,
        image_size=prepare_run_config.image_size,
        image_pixel_arcsec=prepare_run_config.image_pixel_arcsec,
        calibration_freq_interval=prepare_run_config.calibration_freq_interval,
        calibration_time_interval=prepare_run_config.calibration_time_interval,
        image_prefix=image_prefix
    )
    with open('run_config.json', 'w') as f:
        f.write(run_config.json(indent=2))


if __name__ == '__main__':
    prepare_run_config = os.environ.get('RUN_CONFIG')
    if not os.path.exists(prepare_run_config):
        raise ValueError(
            f"RUN_CONFIG environment variable must point to a valid run config file. Got {prepare_run_config}")
    main(prepare_run_config=PrepareRunConfig.parse_file(prepare_run_config))
