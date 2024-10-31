import os
from abc import ABC
from functools import cached_property
from typing import Tuple, List, Callable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
from shapely.geometry import Point
from shapely.ops import nearest_points

from src.dsa2000_cal.assets.base_content import BaseContent


class RegionSampler:
    def __init__(self, shapefile_path, target_crs="EPSG:4326",
                 filter: Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame] | None = None):
        self.source = shapefile_path
        # Load the shapefile using geopandas
        self.gdf = gpd.read_file(shapefile_path)
        if filter is not None:
            self.gdf = filter(self.gdf)

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

    def closest_approach_to_boundary(self, lon, lat):
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


class ArrayConstraint(ABC, BaseContent):
    """
    Abstract array class.
    """

    def __init__(self, *args, **kwargs):
        BaseContent.__init__(self, seed='array_constraint')

    def get_array_constraint_folder(self) -> str:
        return str(os.path.join(*self.content_path, 'spring_valley_31b'))

    def get_constraint_regions(self) -> List[Tuple[RegionSampler, float]]:
        """
        Returns the buffer constraints for the array.

        Returns:
            list of tuples: (RegionSampler, buffer distance m)
        """
        # Buffers are minimal distance from the edge of the shapefile

        folder = self.get_array_constraint_folder()
        return [
            (RegionSampler(os.path.join(folder, "Avoidance_Area_9_9.shp")), 20.0),
            (RegionSampler(os.path.join(folder, "Fences.shp")), 20.0),
            (RegionSampler(os.path.join(folder, "Private Property within AOI.shp")), 20.0),
            (RegionSampler(os.path.join(folder, "Soil Data - 3_0.shp"),
                           filter=lambda gdf: gdf[gdf['SUMMARY_TY'] == "Bedrock"]), 61.0),
            (RegionSampler(os.path.join(folder, "Spring Valley Ephemeral Streams.shp")), 20.0),
            (RegionSampler(os.path.join(folder, "Spring Valley Intermittent Streams.shp")), 20.0),
            (RegionSampler(os.path.join(folder, "Spring Valley Perennial Streams- Observed.shp")), 20.0),
            (RegionSampler(os.path.join(folder, "Spring Valley Road Segments.shp")), 9.0),
            (RegionSampler(os.path.join(folder, "Spring Valley Water Bodies.shp")), 61.0),
            (RegionSampler(os.path.join(folder, "Tertiary Roads.shp")), 9.0)
        ]

    def get_area_of_interest_regions(self) -> List[Tuple[RegionSampler, float]]:
        """
        Returns the buffer constraints for the array.

        Returns:
            list of tuples: (RegionSampler, buffer distance m)
        """
        # Prepend the folder path
        folder = self.get_array_constraint_folder()

        return [
            (RegionSampler(os.path.join(folder, "Spring Valley AOI 3.1b.shp")), 20.0)
        ]


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    """

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c
