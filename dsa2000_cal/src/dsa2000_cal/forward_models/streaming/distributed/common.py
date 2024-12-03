from astropy import units as au

from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.common.types import DishEffectsParams
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMeta


class ChunkParams(SerialisableBaseModel):
    # Freq dimension
    num_channels: int
    num_sub_bands: int
    num_freqs_per_sol_int: int

    # Time dimension
    num_integrations: int
    num_times_per_sol_int: int

    num_baselines: int

    # Calibration average rules
    num_model_times_per_solution_interval: int
    num_model_freqs_per_solution_interval: int

    num_sol_ints_freq: int | None = None
    num_sol_ints_per_sub_band: int | None = None
    num_freqs_per_sub_band: int | None = None
    num_sol_ints_time: int | None = None

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(ChunkParams, self).__init__(**data)
        # Use _check_measurement_set_meta_v0 as instance-wise validator
        _check_chunk_params(self)


def _check_chunk_params(chunk_params: ChunkParams):
    # Check divisibility
    if chunk_params.num_channels % chunk_params.num_sub_bands != 0:
        raise ValueError(
            f"Number of channels {chunk_params.num_channels} not divisible by num_sub_bands {chunk_params.num_sub_bands}"
        )
    chunk_params.num_freqs_per_sub_band = chunk_params.num_channels // chunk_params.num_sub_bands

    if chunk_params.num_channels % chunk_params.num_freqs_per_sol_int != 0:
        raise ValueError(
            f"Number of channels {chunk_params.num_channels} not divisible by num_freqs_per_sol_int {chunk_params.num_freqs_per_sol_int}"
        )
    chunk_params.num_sol_ints_freq = chunk_params.num_channels // chunk_params.num_freqs_per_sol_int

    if chunk_params.num_sol_ints_freq % chunk_params.num_sub_bands != 0:
        raise ValueError(
            f"Number of solution integrations per frequency {chunk_params.num_sol_ints_freq} not divisible by num_sub_bands {chunk_params.num_sub_bands}"
        )
    chunk_params.num_sol_ints_per_sub_band = chunk_params.num_sol_ints_freq // chunk_params.num_sub_bands

    if chunk_params.num_integrations % chunk_params.num_times_per_sol_int != 0:
        raise ValueError(
            f"Number of integrations {chunk_params.num_integrations} not divisible by num_times_per_sol_int {chunk_params.num_times_per_sol_int}"
        )
    chunk_params.num_sol_ints_time = chunk_params.num_integrations // chunk_params.num_times_per_sol_int

    # Set derived values
    if chunk_params.num_integrations % chunk_params.num_times_per_sol_int != 0:
        raise ValueError(
            f"Number of integrations {chunk_params.num_integrations} not divisible by num_times_per_sol_int {chunk_params.num_times_per_sol_int}"
        )
    chunk_params.num_sol_ints_time_per_image = chunk_params.num_integrations // chunk_params.num_times_per_sol_int


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
    num_cal_facets: int
    plot_folder: str
    run_name: str
