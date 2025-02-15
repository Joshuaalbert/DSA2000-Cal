import os
from typing import Tuple, List

from dsa2000_fm.abc import RegionSampler, AbstractArrayConstraint
from dsa2000_assets.base_content import BaseContent

try:
    import geopandas as gpd
except ImportError:
    print(f"Geopandas is not installed. Please install it to use array constraints.")


def _test_region_sampler():
    files = [
        "/home/albert/git/DSA2000-Cal/dsa2000_cal/src/dsa2000_cal/assets/array_constraints/spring_valley_31b/Buildable Area - Outside AOI.shp",
        "/home/albert/git/DSA2000-Cal/dsa2000_cal/src/dsa2000_cal/assets/array_constraints/spring_valley_31b/Western Expansion Areas.shp"
    ]

    for file in files:
        sampler = RegionSampler(file)
        assert sampler.total_area > 0
        sample = sampler.get_samples_within(1)[0]
        assert sampler.closest_approach(*sample)[1] == 0
        assert sampler.closest_approach_to_boundary(*sample)[1] > 0


class ArrayConstraintsV1(BaseContent, AbstractArrayConstraint):
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


class ArrayConstraintsV2(BaseContent, AbstractArrayConstraint):
    """
    Abstract array class.
    """

    def __init__(self, *args, **kwargs):
        BaseContent.__init__(self, seed='array_constraint_v2')

    def get_array_constraint_folder(self) -> str:
        return str(os.path.join(*self.content_path, 'spring_valley_31b'))

    def get_constraint_regions(self) -> List[Tuple[RegionSampler, float]]:
        """
        Returns the buffer constraints for the array.

        Returns:
            list of tuples: (RegionSampler, buffer distance m)
        """
        _30ft = 9.144
        _50ft = 15.24
        _60ft = 18.288
        _100ft = 30.48
        _200ft = 60.96

        folder = self.get_array_constraint_folder()

        return [
            (RegionSampler(os.path.join(folder, "aoi_NHD_High_Resolution_Flowline.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "aoi_NHD_V2_Flowlines.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "aoi_NHD_V2_Waterbodies.shp")), _200ft),
            (RegionSampler(os.path.join(folder, "aoi_NWI_Wetland.shp")), _200ft),
            (RegionSampler(os.path.join(folder, "Avoidance_Area_9_9.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Avoidance - Terrain.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Existing Access Paths.shp")), _30ft),
            (RegionSampler(os.path.join(folder, "Fences.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Private Property within AOI.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Soil Data - 3_0.shp"),
                           filter=lambda gdf: gdf[gdf['SUMMARY_TY'] == "Bedrock"]), _200ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Ephemeral Streams.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Intermittent Streams.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Perennial Streams- Observed.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Road Segments.shp")), _30ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Water Bodies.shp")), _200ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Watershed.shp")), _200ft),
            (RegionSampler(os.path.join(folder, "Tertiary Roads.shp")), _30ft)
        ]

    def get_area_of_interest_regions(self) -> List[Tuple[RegionSampler, float]]:
        """
        Returns the buffer constraints for the array.

        Returns:
            list of tuples: (RegionSampler, buffer distance m)
        """
        # Prepend the folder path
        folder = self.get_array_constraint_folder()

        _30ft = 9.144
        _50ft = 15.24
        _60ft = 18.288
        _100ft = 30.48
        _200ft = 60.96

        return [
            (RegionSampler(os.path.join(folder, "Spring Valley AOI 3.1b.shp")), _60ft),
            # (RegionSampler(os.path.join(folder, "Buildable Area - Outside AOI.shp")), _60ft),
            # (RegionSampler(os.path.join(folder, "Western Expansion Areas.shp")), _60ft)
        ]


class ArrayConstraintsV3(BaseContent, AbstractArrayConstraint):
    """
    Abstract array class.
    """

    def __init__(self, *args, **kwargs):
        BaseContent.__init__(self, seed='array_constraint_v3')

    def get_array_constraint_folder(self) -> str:
        return str(os.path.join(*self.content_path, 'spring_valley_31b'))

    def get_constraint_regions(self) -> List[Tuple[RegionSampler, float]]:
        """
        Returns the buffer constraints for the array.

        Returns:
            list of tuples: (RegionSampler, buffer distance m)
        """
        _30ft = 9.144
        _50ft = 15.24
        _60ft = 18.288
        _100ft = 30.48
        _200ft = 60.96

        folder = self.get_array_constraint_folder()

        return [
            (RegionSampler(os.path.join(folder, "aoi_NHD_High_Resolution_Flowline.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "aoi_NHD_V2_Flowlines.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "aoi_NHD_V2_Waterbodies.shp")), _200ft),
            (RegionSampler(os.path.join(folder, "aoi_NWI_Wetland.shp")), _200ft),
            (RegionSampler(os.path.join(folder, "Avoidance_Area_9_9.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Avoidance - Terrain.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Existing Access Paths.shp")), _30ft),
            (RegionSampler(os.path.join(folder, "Fences.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Private Property within AOI.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Soil Data - 3_0.shp"),
                           filter=lambda gdf: gdf[gdf['SUMMARY_TY'] == "Bedrock"]), _200ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Ephemeral Streams.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Intermittent Streams.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Perennial Streams- Observed.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Road Segments.shp")), _30ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Water Bodies.shp")), _200ft),
            (RegionSampler(os.path.join(folder, "Spring Valley Watershed.shp")), _200ft),
            (RegionSampler(os.path.join(folder, "Tertiary Roads.shp")), _30ft)
        ]

    def get_area_of_interest_regions(self) -> List[Tuple[RegionSampler, float]]:
        """
        Returns the buffer constraints for the array.

        Returns:
            list of tuples: (RegionSampler, buffer distance m)
        """
        # Prepend the folder path
        folder = self.get_array_constraint_folder()

        _30ft = 9.144
        _50ft = 15.24
        _60ft = 18.288
        _100ft = 30.48
        _200ft = 60.96

        return [
            (RegionSampler(os.path.join(folder, "Spring Valley AOI 3.1b.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Buildable Area - Outside AOI.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Western Expansion Areas.shp")), _60ft)
        ]


