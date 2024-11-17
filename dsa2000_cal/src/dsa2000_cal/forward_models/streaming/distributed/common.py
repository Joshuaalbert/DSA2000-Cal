from astropy import units as au

from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.common.types import DishEffectsParams
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMeta


class ChunkParams(SerialisableBaseModel):
    num_channels: int
    num_integrations: int
    num_baselines: int
    # Freq dimension
    num_freqs_per_sol_int: int
    num_sol_ints_per_sub_band: int
    num_sub_bands_per_image: int
    # Time dimension
    num_times_per_sol_int: int
    num_sol_ints_per_accumlate: int
    num_accumulates_per_image: int

    num_sol_ints_freq: int | None = None
    num_sub_bands: int | None = None
    num_images_freq: int | None = None
    num_sol_ints_time: int | None = None
    num_accumulates: int | None = None
    num_images_time: int | None = None

    num_sol_ints_freq_per_image: int | None = None
    num_sol_ints_time_per_image: int | None = None

    num_baselines_per_sol_int: int | None = None

    num_freqs_per_sub_band: int | None = None
    num_times_per_accumulate: int | None = None

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(ChunkParams, self).__init__(**data)
        # Use _check_measurement_set_meta_v0 as instance-wise validator
        _check_chunk_params(self)


def _check_chunk_params(chunk_params: ChunkParams):
    # Check divisibility
    if chunk_params.num_channels % chunk_params.num_freqs_per_sol_int != 0:
        raise ValueError(
            f"Number of channels {chunk_params.num_channels} not divisible by num_freqs_per_sol_int {chunk_params.num_freqs_per_sol_int}")
    num_sol_ints_freq = chunk_params.num_channels // chunk_params.num_freqs_per_sol_int
    if num_sol_ints_freq % chunk_params.num_sol_ints_per_sub_band != 0:
        raise ValueError(
            f"Number of sol ints {num_sol_ints_freq} not divisible by num_sol_ints_per_sub_band {chunk_params.num_sol_ints_per_sub_band}")
    num_sub_bands = num_sol_ints_freq // chunk_params.num_sol_ints_per_sub_band
    if num_sub_bands % chunk_params.num_sub_bands_per_image != 0:
        raise ValueError(
            f"Number of sub bands {num_sub_bands} not divisible by num_sub_bands_per_image {chunk_params.num_sub_bands_per_image}")
    num_images_freq = num_sub_bands // chunk_params.num_sub_bands_per_image

    if chunk_params.num_integrations % chunk_params.num_times_per_sol_int != 0:
        raise ValueError(
            f"Number of integrations {chunk_params.num_integrations} not divisible by num_times_per_sol_int {chunk_params.num_times_per_sol_int}")
    num_sol_ints_time = chunk_params.num_integrations // chunk_params.num_times_per_sol_int
    if num_sol_ints_time % chunk_params.num_sol_ints_per_accumlate != 0:
        raise ValueError(
            f"Number of sol ints {num_sol_ints_time} not divisible by num_sol_ints_per_accumlate {chunk_params.num_sol_ints_per_accumlate}")
    num_accumulates = num_sol_ints_time // chunk_params.num_sol_ints_per_accumlate
    if num_accumulates % chunk_params.num_accumulates_per_image != 0:
        raise ValueError(
            f"Number of accumulates {num_accumulates} not divisible by num_accumulates_per_image {chunk_params.num_accumulates_per_image}")
    num_images_time = num_accumulates // chunk_params.num_accumulates_per_image
    # Set derived values
    chunk_params.num_sol_ints_freq = num_sol_ints_freq
    chunk_params.num_sub_bands = num_sub_bands
    chunk_params.num_images_freq = num_images_freq
    chunk_params.num_sol_ints_time = num_sol_ints_time
    chunk_params.num_accumulates = num_accumulates
    chunk_params.num_images_time = num_images_time

    chunk_params.num_sol_ints_freq_per_image = chunk_params.num_sol_ints_per_sub_band * chunk_params.num_sub_bands_per_image
    chunk_params.num_sol_ints_time_per_image = chunk_params.num_sol_ints_per_accumlate * chunk_params.num_accumulates_per_image

    chunk_params.num_baselines_per_sol_int = chunk_params.num_baselines * chunk_params.num_times_per_sol_int

    chunk_params.num_freqs_per_sub_band = chunk_params.num_freqs_per_sol_int * chunk_params.num_sol_ints_per_sub_band
    chunk_params.num_times_per_accumulate = chunk_params.num_times_per_sol_int * chunk_params.num_sol_ints_per_accumlate


class ImageParams(SerialisableBaseModel):
    l0: au.Quantity
    m0: au.Quantity
    dl: au.Quantity
    dm: au.Quantity
    num_l: int
    num_m: int
    epsilon: float = 1e-6


class ForwardModellingRunParams(SerialisableBaseModel):
    ms_meta: MeasurementSetMeta
    dish_effects_params: DishEffectsParams
    chunk_params: ChunkParams
    image_params: ImageParams
    full_stokes: bool
    num_facets: int
    plot_folder: str
    run_name: str
