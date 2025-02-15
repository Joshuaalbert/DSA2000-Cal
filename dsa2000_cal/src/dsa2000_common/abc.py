import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Callable, Tuple

import geopandas as gpd
import numpy as np
from astropy import units as au, coordinates as ac
from matplotlib import pyplot as plt
from numpy.random import uniform
from pydantic import Field
from shapely import Point, Polygon, MultiPolygon
from shapely.ops import nearest_points

from dsa2000_assets.array_constraints.array_constraint_content import haversine
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_common.common.base_content import BaseContent
from dsa2000_common.common.interp_utils import InterpolatedArray
from dsa2000_common.common.types import DishEffectsParams
from dsa2000_common.visibility_model.source_models.rfi.parametric_rfi_emitter import ParametricDelayACF
from dsa2000_fm.antenna_model.abc import AbstractAntennaModel


class AbstractArray(ABC, BaseContent):
    """
    Abstract array class.
    """

    def __init__(self, *args, **kwargs):
        BaseContent.__init__(self, *args, **kwargs)

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


class AbstractBeamModel(ABC, BaseContent):

    @abstractmethod
    def get_antenna_model(self) -> AbstractAntennaModel:
        """
        Get the beam model.

        Returns:
            beam model
        """
        ...


class RFIEmitterSourceModelParams(SerialisableBaseModel):
    freqs: au.Quantity  # [num_chans]
    delay_acf: InterpolatedArray | ParametricDelayACF  # [E,chan[,2,2]]
    position_enu: au.Quantity = Field(
        description=" [E, 3] Location in ENU [m] from antenna[0]."
    )

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(RFIEmitterSourceModelParams, self).__init__(**data)
        _check_lte_model_params(self)


def _check_lte_model_params(params: RFIEmitterSourceModelParams):
    if not params.freqs.unit.is_equivalent(au.Hz):
        raise ValueError("Frequency must be in Hz.")
    if not params.position_enu.unit.is_equivalent(au.m):
        raise ValueError("Location must be in meters.")
    # Check shapes
    if len(params.position_enu.shape) != 2:
        raise ValueError(f"Location must be a [E, 3], got {params.position_enu.shape}.")
    E, _ = params.position_enu.shape
    num_chan = len(params.freqs)
    if not ((params.delay_acf.shape == (E, num_chan)) or (
            params.delay_acf.shape == (E, num_chan, 2, 2))):
        raise ValueError(f"ACF must be [E, num_chans[,2 ,2]], got {params.delay_acf.shape}.")


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


class AbstractWSCleanSourceModel(ABC, BaseContent):
    """
    Abstract array class.
    """

    def __init__(self, *args, **kwargs):
        BaseContent.__init__(self, *args, **kwargs)

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


class RegionSampler:
    def __init__(self, shapefile_path, target_crs="EPSG:4326",
                 filter: Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame] | None = None):
        self.source = shapefile_path
        # Load the shapefile using geopandas
        self.gdf = gpd.read_file(shapefile_path)
        if filter is not None:
            self.gdf = filter(self.gdf)

        # # Validate geometries
        # self.gdf = self.gdf[self.gdf.is_valid]
        # self.gdf = self.gdf.buffer(0)  # Fix invalid geometries if necessary

        # Check if the shapefile has a CRS; if not, raise an error
        if self.gdf.crs is None:
            raise ValueError("The shapefile does not have a CRS.")

        # Reproject the GeoDataFrame to the target CRS (default: WGS84 - EPSG:4326)
        self.target_crs = target_crs
        self.gdf = self.gdf.to_crs(self.target_crs)
        self.polygon = self.gdf.union_all()

    def contains(self, lon, lat):
        return self.polygon.contains(Point(lon, lat))

    @property
    def name(self):
        return os.path.basename(self.source).split('.')[0]

    def _get_bbox(self, buffer_ratio=1.0):
        """Returns the bounding box of the polygon, scaled by buffer_ratio."""
        minx, miny, maxx, maxy = self.polygon.bounds
        width, height = maxx - minx, maxy - miny
        buffer_x = width * (buffer_ratio - 1) / 2
        buffer_y = height * (buffer_ratio - 1) / 2
        return (minx - buffer_x, miny - buffer_y, maxx + buffer_x, maxy + buffer_y)

    def get_samples_within(self, num_samples):
        """Generates num_samples points within the polygon."""
        samples = []
        minx, miny, maxx, maxy = self._get_bbox()

        while len(samples) < num_samples:
            # Generate random points within the bounding box
            random_points = np.column_stack([uniform(minx, maxx, num_samples),
                                             uniform(miny, maxy, num_samples)])
            for lon, lat in random_points:
                point = Point(lon, lat)
                # Check if point is within the polygon
                if self.polygon.contains(point):
                    samples.append((lon, lat))  # Assuming elevation 0, adjust as needed
                if len(samples) >= num_samples:
                    break
        return np.asarray(samples)

    def get_samples_complement(self, num_samples):
        """Generates num_samples points outside the polygon but within a 10% larger bounding box."""
        samples = []
        minx, miny, maxx, maxy = self._get_bbox(buffer_ratio=1.1)  # 10% larger

        while len(samples) < num_samples:
            # Generate random points within the expanded bounding box
            random_points = np.column_stack([uniform(minx, maxx, num_samples),
                                             uniform(miny, maxy, num_samples)])
            for lon, lat in random_points:
                point = Point(lon, lat)
                # Check if point is outside the polygon
                if not self.polygon.contains(point):
                    samples.append((lon, lat))  # Assuming elevation 0, adjust as needed
                if len(samples) >= num_samples:
                    break
        return np.asarray(samples)

    def closest_approach(self, lon, lat):
        """
        Calculates the closest point on the polygon to the provided lon/lat and returns
        the closest point along with the distance to the given point.
        """
        # Create a shapely Point from the input lon/lat
        input_point = Point(lon, lat)

        # Find the nearest point on the polygon to the input point
        _, closest_point = nearest_points(input_point, self.polygon)
        # Calculate the distance between the input point and the closest point
        angular_dist = haversine(lon * np.pi / 180, lat * np.pi / 180, closest_point.x * np.pi / 180,
                                 closest_point.y * np.pi / 180) * 180 / np.pi

        # Return the closest point (longitude, latitude) and the distance
        return (closest_point.x, closest_point.y), angular_dist

    def _closest_approach_to_boundary(self, lon, lat):
        """
        Calculates the closest point on the boundary of the polygon to the provided lon/lat.
        This method will return a point on the perimeter even if the input point is inside
        the polygon.
        """
        # Create a shapely Point from the input lon/lat
        input_point = Point(lon, lat)

        # Find the exterior (boundary) of the polygon
        polygon_boundary = self.polygon.exterior

        # If dealing with multiple polygons (i.e., multi-polygon), take the boundary of all of them
        if self.polygon.geom_type == 'MultiPolygon':
            polygon_boundary = self.polygon.boundary

        # Find the nearest point on the polygon boundary to the input point
        _, closest_point = nearest_points(input_point, polygon_boundary)

        # Calculate the distance between the input point and the closest point on the boundary

        angular_dist = haversine(lon * np.pi / 180, lat * np.pi / 180, closest_point.x * np.pi / 180,
                                 closest_point.y * np.pi / 180) * 180 / np.pi

        # Return the closest point (longitude, latitude) on the boundary and the distance
        return (closest_point.x, closest_point.y), angular_dist

    def closest_approach_to_boundary(self, lon, lat):
        """
        Calculates the closest point on the boundary of the polygon to the provided lon/lat.
        This method will return a point on the perimeter even if the input point is inside
        the polygon.
        """
        # Create a shapely Point from the input lon/lat
        input_point = Point(lon, lat)

        # Determine the polygon boundary
        if isinstance(self.polygon, Polygon):
            polygon_boundary = self.polygon.exterior
        elif isinstance(self.polygon, MultiPolygon):
            polygon_boundary = self.polygon.boundary
        else:
            raise ValueError("The geometry is not a Polygon or MultiPolygon.")

        # Find the nearest point on the polygon boundary to the input point
        _, closest_point = nearest_points(input_point, polygon_boundary)

        # Calculate the distance between the input point and the closest point on the boundary
        angular_dist = haversine(lon * np.pi / 180, lat * np.pi / 180, closest_point.x * np.pi / 180,
                                 closest_point.y * np.pi / 180) * 180 / np.pi

        # Return the closest point (longitude, latitude) on the boundary and the distance
        return (closest_point.x, closest_point.y), angular_dist

    def plot_region(self, ax: plt.Axes, color='none'):
        """Plots the polygon on a matplotlib axis."""
        # Plot the polygon(s)
        self.gdf.plot(ax=ax, edgecolor='black', facecolor=color, alpha=0.1)
        ax.set_title('Polygon Region')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    @cached_property
    def total_area(self):
        # Total area
        # UserWarning: Geometry is in a geographic CRS. Results from 'area' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.
        total_area = self.gdf.geometry.area.sum()
        return total_area

    def info(self):
        """Prints out key information about the region."""
        # Number of polygons
        num_polygons = len(self.gdf)

        # Total area
        total_area = self.gdf.geometry.area.sum()

        # Total perimeter (length of boundary)
        total_perimeter = self.gdf.geometry.length.sum()

        # Centroid of the whole region
        centroid = self.polygon.centroid

        # Units of the region
        proj = self.gdf.crs.to_proj4()

        # Bounding box
        minx, miny, maxx, maxy = self.polygon.bounds

        columns = self.gdf.columns

        # Display the information
        print("Region Information:")
        print(f"  - File: {self.source}")
        print(f"  - Projection: {proj}")
        print(f"  - Number of polygons: {num_polygons}")
        print(f"  - Total area: {total_area:.2f} square units")
        print(f"  - Total perimeter: {total_perimeter:.2f} units")
        print(f"  - Centroid (longitude, latitude): ({centroid.x:.4f}, {centroid.y:.4f})")
        print(f"  - Bounding box: [({minx:.4f}, {miny:.4f}), ({maxx:.4f}, {maxy:.4f})]")
        print(f"  - Columns: {columns}")


class AbstractArrayConstraint(ABC, BaseContent):
    @abstractmethod
    def get_constraint_regions(self) -> List[Tuple[RegionSampler, float]]:
        """
        Returns the buffer constraints for the array.

        Returns:
            list of tuples: (RegionSampler, buffer distance m)
        """
        ...

    @abstractmethod
    def get_area_of_interest_regions(self) -> List[Tuple[RegionSampler, float]]:
        """
        Returns the buffer constraints for the array.

        Returns:
            list of tuples: (RegionSampler, buffer distance m)
        """
        ...
