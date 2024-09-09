from abc import ABC, abstractmethod

from astropy import units as au
from pydantic import Field

from dsa2000_cal.assets.base_content import BaseContent
from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel


class RFIEmitterSourceModelParams(SerialisableBaseModel):
    freqs: au.Quantity  # [num_chans]
    delay_acf: InterpolatedArray  # [E]
    position_enu: au.Quantity = Field(
        description=" [E, 3] Location in ENU [m] from antenna[0]."
    )
    spectral_flux_density: au.Quantity = Field(
        description="[E, num_chans[, 2, 2]] Spectral flux density of RFI transmitter at the source [W/Hz] or [Jy m^2] per channel."
    )

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(RFIEmitterSourceModelParams, self).__init__(**data)
        _check_lte_model_params(self)


def _check_lte_model_params(params: RFIEmitterSourceModelParams):
    if not params.freqs.unit.is_equivalent(au.Hz):
        raise ValueError("Frequency must be in Hz.")
    if not params.spectral_flux_density.unit.is_equivalent(au.Jy * au.m ** 2):
        raise ValueError("Luminosity must be in Jy m^2.")
    if not params.position_enu.unit.is_equivalent(au.m):
        raise ValueError("Location must be in meters.")
    # Check shapes
    if len(params.position_enu.shape) != 2:
        raise ValueError(f"Location must be a [E, 3], got {params.position_enu.shape}.")
    E, _ = params.position_enu.shape
    num_chan = len(params.freqs)
    if not ((params.spectral_flux_density.shape == (E, num_chan)) or (
            params.spectral_flux_density.shape == (E, num_chan, 2, 2))):
        raise ValueError(f"Luminosity must be [E, num_chans[,2 ,2]], got {params.spectral_flux_density.shape}.")


class AbstractRFIEmitterData(ABC, BaseContent):
    def __init__(self, *args, **kwargs):
        BaseContent.__init__(self, *args, **kwargs)

    @abstractmethod
    def make_source_params(self, freqs: au.Quantity, central_freq: au.Quantity | None = None,
                           full_stokes: bool = False) -> RFIEmitterSourceModelParams:
        """
        Make the source parameters for the LTE RFI model.

        Args:
            freqs: [num_chans] Frequencies of the observation [Hz]
            central_freq: Central frequency of the observation [Hz]
            full_stokes: Whether to return full stokes parameters

        Returns:
            The source parameters
        """
        ...
