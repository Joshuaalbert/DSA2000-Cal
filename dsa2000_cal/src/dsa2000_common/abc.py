from abc import ABC, abstractmethod
from typing import List

from astropy import units as au, coordinates as ac

from dsa2000_common.common.types import DishEffectsParams
from dsa2000_common.visibility_model.source_models.rfi.abc import AbstractRFIAutoCorrelationFunction
from dsa2000_fm.antenna_model.abc import AbstractAntennaModel


class AbstractArray(ABC):
    """
    Abstract array class.
    """

    def __init__(self, *args, **kwargs):
        def _check_quantity(value: au.Quantity, unit, is_scalar):
            if not isinstance(value, au.Quantity):
                raise TypeError(f"Expected a Quantity, got {type(value)}")
            if not value.unit.is_equivalent(unit):
                raise ValueError(f"Expected unit {unit}, got {value.unit}")
            if is_scalar and not value.isscalar:
                raise ValueError(f"Expected a scalar quantity, got {value}")

        _check_quantity(self.get_channel_width(), au.Hz, is_scalar=True)
        _check_quantity(self.get_antenna_diameter(), au.m, is_scalar=True)
        _check_quantity(self.get_system_equivalent_flux_density(), au.Jy, is_scalar=True)
        _check_quantity(self.get_system_efficiency(), au.dimensionless_unscaled, is_scalar=True)
        _array_location = self.get_array_location()
        if not isinstance(_array_location, ac.EarthLocation):
            raise TypeError(f"Expected an EarthLocation, got {type(_array_location)}")
        if not _array_location.isscalar:
            raise ValueError(f"Expected a scalar EarthLocation, got {_array_location}")
        _antennas = self.get_antennas()
        if not isinstance(_antennas, ac.EarthLocation):
            raise TypeError(f"Expected an EarthLocation, got {type(_antennas)}")
        if _antennas.isscalar:
            raise ValueError(f"Expected a vector EarthLocation, got {_antennas}")

    @abstractmethod
    def get_channel_width(self) -> au.Quantity:
        """
        Get channel width (Hz)

        Returns:
            channel width
        """
        ...

    @abstractmethod
    def get_channels(self) -> au.Quantity:
        """
        Get channels.

        Returns:
            channels
        """
        ...

    @abstractmethod
    def get_array_location(self) -> ac.EarthLocation:
        """
        Get array location.

        Returns:
            array center
        """
        ...

    @abstractmethod
    def get_antennas(self) -> ac.EarthLocation:
        """
        Get antenna positions.

        Returns:
            antenna positions in ITRS frame
        """
        ...

    @abstractmethod
    def get_antenna_names(self) -> List[str]:
        """
        Get antenna names.

        Returns:
            antenna names
        """
        ...

    @abstractmethod
    def get_array_file(self) -> str:
        """
        Get array file.

        Returns:
            array file
        """
        ...

    @abstractmethod
    def get_antenna_diameter(self) -> au.Quantity:
        """
        Get antenna diameter (m)

        Returns:
            antenna diameter
        """
        ...

    @abstractmethod
    def get_focal_length(self) -> au.Quantity:
        """
        Get focal length (m)

        Returns:
            focal length
        """
        ...

    @abstractmethod
    def get_mount_type(self) -> str:
        """
        Get mount type.

        Returns:
            mount type
        """
        ...

    @abstractmethod
    def get_station_name(self) -> str:
        """
        Get station name.

        Returns:
            station name
        """
        ...

    @abstractmethod
    def get_system_equivalent_flux_density(self) -> au.Quantity:
        """
        Get system equivalent flux density (Jy)

        Returns:
            system equivalent flux density
        """
        ...

    @abstractmethod
    def get_system_efficiency(self) -> au.Quantity:
        """
        Get system efficiency

        Returns:
            system efficiency
        """
        ...

    @abstractmethod
    def get_antenna_model(self) -> AbstractAntennaModel:
        """
        Get antenna beam.

        Returns:
            antenna beam
        """
        ...

    @abstractmethod
    def get_integration_time(self) -> au.Quantity:
        """
        Get integration time (s)

        Returns:
            integration time
        """
        ...

    @abstractmethod
    def get_dish_effect_params(self) -> DishEffectsParams:
        """
        Get dish effects parameters.

        Returns:
            dish effects parameters
        """
        ...


class AbstractBeamModel(ABC):

    @abstractmethod
    def get_antenna_model(self) -> AbstractAntennaModel:
        """
        Get the beam model.

        Returns:
            beam model
        """
        ...


class AbstractRFIEmitterData(ABC):

    @abstractmethod
    def make_rfi_acf(self, freqs: au.Quantity, central_freq: au.Quantity | None = None,
                     full_stokes: bool = False) -> AbstractRFIAutoCorrelationFunction:
        """
        Make the auto-correlation function for the RFI emitter.

        Args:
            freqs: [num_chans] Frequencies of the observation [Hz]
            central_freq: Central frequency of the observation [Hz]
            full_stokes: Whether to return full stokes parameters

        Returns:
            The source parameters
        """
        ...


class AbstractWSCleanSourceModel(ABC):
    """
    Abstract array class.
    """

    @abstractmethod
    def get_wsclean_clean_component_file(self) -> str:
        """
        Get the wsclean source file.

        Returns:
            the wsclean source file
        """
        ...

    @abstractmethod
    def get_wsclean_fits_files(self) -> List[str]:
        """
        Get the files for the wsclean model.

        Returns:
            the fits files
        """
        ...
