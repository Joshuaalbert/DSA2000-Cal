from abc import ABC, abstractmethod
from typing import Tuple, List

import astropy.coordinates as ac
import astropy.units as au
from astropy.coordinates import CartesianRepresentation, ITRS

from src.dsa2000_cal.abc import AbstractAntennaModel
from src.dsa2000_cal.assets.base_content import BaseContent
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsParams


def extract_itrs_coords(filename: str, delim=' ') -> Tuple[List[str], ac.ITRS]:
    """
    Extract stations and antenna ITRS coordinates from a file.

    Args:
        filename: file to read
        delim: delimiter to use for splitting the file

    Returns:
        a tuple of lists of stations and antenna ITRS coordinates
    """
    header = []
    # Initialize lists to store stations and coordinates
    stations = []
    coordinates = []
    station_idx = 0
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if header:
                    raise ValueError(f"Multiple header lines found in {filename}")
                header = list(filter(lambda s: len(s) > 0, map(str.strip, line[1:].lower().split(delim))))
                continue
            if not header:
                raise ValueError(f"No header line found in {filename}")

            # Process each line in the file

            # Split the line into its components
            parsed_line = list(filter(lambda s: len(s) > 0, map(str.strip, line.split(delim))))
            line_dict = dict(zip(header, parsed_line))

            # Convert x, y, z to float and append to the coordinates list
            coordinates.append(
                ITRS(CartesianRepresentation(float(line_dict['x']) * au.m,
                                             float(line_dict['y']) * au.m,
                                             float(line_dict['z']) * au.m))
            )

            # Append the station name to the stations list
            stations.append(line_dict.get('station', f"station_{station_idx}"))
            station_idx += 1
    if len(stations) != len(coordinates):
        raise ValueError(
            f"Number of stations ({len(stations)}) does not match number of coordinates ({len(coordinates)})")
    if len(coordinates) == 0:
        raise ValueError(f"No coordinates found in {filename}")
    if len(coordinates) == 1:
        return stations, coordinates[0].reshape((1,))
    return stations, ac.concatenate(coordinates).transform_to(ITRS())


def _check_quantity(value: au.Quantity, unit, is_scalar):
    if not isinstance(value, au.Quantity):
        raise TypeError(f"Expected a Quantity, got {type(value)}")
    if not value.unit.is_equivalent(unit):
        raise ValueError(f"Expected unit {unit}, got {value.unit}")
    if is_scalar and not value.isscalar:
        raise ValueError(f"Expected a scalar quantity, got {value}")


class AbstractArray(ABC, BaseContent):
    """
    Abstract array class.
    """

    def __init__(self, *args, **kwargs):
        BaseContent.__init__(self, *args, **kwargs)
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
    def integration_time(self) -> au.Quantity:
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
