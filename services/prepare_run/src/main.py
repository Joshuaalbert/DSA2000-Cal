import os
import shutil
import subprocess
from datetime import datetime

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
from pydantic import conint, confloat, Field
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.assets.arrays.array import AbstractArray
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.mocks.mock_data import MockData
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.astropy_utils import mean_itrs
from dsa2000_cal.bbs_sky_model import create_sky_model
from dsa2000_cal.create_ms_cfg import create_makems_config
from dsa2000_cal.faint_sky_model import repoint_fits
from dsa2000_cal.run_config import RunConfig
from dsa2000_cal.utils import SerialisableBaseModel

fill_registries()


class PrepareRunConfig(SerialisableBaseModel):
    """
    Represents the configuration for preparing a run.
    """
    array_name: str = Field(
        description="The name of the array to use.",
        example="dsa2000W",
    )
    start_dt: datetime = Field(
        description="The start datetime of the run.",
        example=datetime.fromisoformat("2023-10-10T12:00:00"),
    )
    alt_deg: confloat(ge=0, le=90) = Field(
        description="The altitude of the pointing direction in degrees, measured from horizon to zenith.",
        example=90,
    )
    az_deg: confloat(ge=0, le=360) = Field(
        description="The azimuth of the pointing direction in degrees measured East from North.",
        example=0,
    )
    num_bright_sources: conint(ge=0) = Field(
        description="The number of bright sources to use in the simulation, if any.",
        example=10,
    )
    spacing_deg: confloat(ge=0) = Field(
        description="The spacing between bright sources in degrees",
        example=1.,
    )
    faint_sky_model_fits: str | None = Field(
        description="The path to the faint sky model fits file, if given.",
        example=MockData().faint_sky_model(),
    )
    start_freq_hz: confloat(gt=0) = Field(
        description="The start frequency of the simulation in Hz.",
        example=800e6,
    )
    channel_width_hz: confloat(gt=0) = Field(
        description="The channel width of the simulation in Hz.",
        example=2e6,
    )
    num_channels: conint(ge=1) = Field(
        description="The number of channels in the simulation.",
        example=32
    )
    num_times: conint(ge=1) = Field(
        description="The number of times in the simulation.",
        example=10
    )
    integration_time_s: confloat(gt=0) = Field(
        description="The integration time of the simulation in seconds.",
        example=1.5
    )
    ionosphere_specification: SPECIFICATION = Field(
        description="The ionosphere specification, one of ['simple', 'light_dawn', 'dawn', 'dusk', 'dawn_challenge', 'dusk_challenge']",
        example="light_dawn"
    )


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
        repoint_fits(fits_file=faint_sky_model_fits,
                     output_file=faint_sky_model_fits,
                     pointing_centre=pointing_centre)
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
        visibilities_path=output_ms_file,
        dft_visibilities_path=dft_ms_file,
        fft_visibilities_path=fft_ms_file,
        rfi_visibilities_path=rfi_ms_file
    )
    with open('run_config.json', 'w') as f:
        f.write(run_config.json(indent=2))


if __name__ == '__main__':
    prepare_run_config = os.environ.get('RUN_CONFIG')
    if not os.path.exists(prepare_run_config):
        raise ValueError('Environment variable RUN_CONFIG must be set.')
    main(prepare_run_config=PrepareRunConfig.parse_file(prepare_run_config))
