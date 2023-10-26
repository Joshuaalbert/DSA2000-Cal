from datetime import datetime
from typing import Union

import astropy.coordinates as ac
import astropy.units as au
from pydantic import Field, confloat, conint, constr
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.rfi.rfi_simulation import RFISimConfig
from dsa2000_cal.utils import SerialisableBaseModel


class RunConfig(SerialisableBaseModel):
    """
    Represents the configuration for a run.
    """
    array_name: str = Field(
        description="The name of the array to use.",
        example="dsa2000W_small",
    )
    start_dt: datetime = Field(
        description="The start datetime of the run.",
        example=datetime.fromisoformat("2023-10-10T12:00:00"),
    )
    pointing_centre: ac.ICRS = Field(
        description="The pointing direction in ICRS.",
        example=ac.ICRS(ra=10 * au.deg, dec=45 * au.deg),
    )
    bright_sky_model_bbs: Union[str, None] = Field(
        description="The path to the bright sky model bbs file, if given.",
        example="bright_sky_model.bbs",
    )
    faint_sky_model_fits: Union[constr(regex=r".*-model\.fits$"), None] = Field(
        description="The path to the faint sky model fits file, if given must end in '-model.fits'.",
        example="faint_sky-model.fits",
    )
    start_freq_hz: confloat(gt=0) = Field(
        description="The start frequency of the simulation in Hz.",
        example=700e6,
    )
    channel_width_hz: confloat(gt=0) = Field(
        description="The channel width of the simulation in Hz.",
        example=162.5e3,
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
    visibilities_path: str = Field(
        description="The path to the output measurement set.",
        example="visibilities.ms",
    )
    dft_visibilities_path: str = Field(
        description="The path to the DFT measurement set.",
        example="dft_visibilities.ms",
    )
    fft_visibilities_path: str = Field(
        description="The path to the FFT measurement set.",
        example="fft_visibilities.ms",
    )
    rfi_visibilities_path: str = Field(
        description="The path to the RFI measurement set.",
        example="rfi_visibilities.ms",
    )
    ionosphere_specification: SPECIFICATION = Field(
        description="The ionosphere specification, one of ['simple', 'light_dawn', 'dawn', 'dusk', 'dawn_challenge', 'dusk_challenge']",
        example="light_dawn"
    )
    ionosphere_h5parm: str = Field(
        description="The path to the output ionosphere h5parm file.",
        example="ionosphere.h5parm",
    )
    ionosphere_fits: str = Field(
        description="The path to the output ionosphere fits file.",
        example="ionosphere.fits",
    )
    beam_h5parm: str = Field(
        description="The path to the output beam h5parm file.",
        example="beam.h5parm",
    )
    beam_fits: str = Field(
        description="The path to the output beam fits file.",
        example="beam.fits",
    )
    rfi_sim_config: RFISimConfig = Field(
        default_factory=RFISimConfig,
        description="The RFI simulation configuration.",
        example=RFISimConfig()
    )
    calibration_time_interval: conint(ge=1) = Field(
        description="The time interval to use for calibration in units of integrations.",
        example=2
    )
    calibration_freq_interval: conint(ge=1) = Field(
        description="The frequency interval to use for calibration in units of channels.",
        example=32
    )
    image_pixel_arcsec: confloat(gt=0) = Field(
        description="The pixel size of the image in arcseconds.",
        example=2.
    )
    image_size: conint(ge=1) = Field(
        description="The size of the image in pixels, assuming square images.",
        example=512
    )
