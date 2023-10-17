from abc import ABC, abstractmethod
from typing import Tuple, List

import astropy.coordinates as ac
import astropy.units as au
from astropy.coordinates import CartesianRepresentation, ITRS

from dsa2000_cal.abc import AbstractAntennaBeam
from dsa2000_cal.assets.base_content import BaseContent


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


class AbstractArray(ABC, BaseContent):
    """
    Abstract array class.
    """

    @abstractmethod
    def get_antennas(self) -> ac.ITRS:
        """
        Get antenna positions.

        Returns:
            antenna positions in ITRS frame
        """
        ...

    @abstractmethod
    def get_antenna_names(self) -> list[str]:
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
    def get_antenna_diameter(self) -> float:
        """
        Get antenna diameter (m)

        Returns:
            antenna diameter
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
    def system_equivalent_flux_density(self) -> float:
        """
        Get system equivalent flux density (Jy)

        Returns:
            system equivalent flux density
        """
        ...

    @abstractmethod
    def system_efficency(self) -> float:
        """
        Get system efficiency

        Returns:
            system efficiency
        """
        ...

    @abstractmethod
    def antenna_beam(self) -> AbstractAntennaBeam:
        """
        Get antenna beam.

        Returns:
            antenna beam
        """
        ...
