import os
from typing import List, Tuple

from dsa2000_assets.base_content import BaseContent
from dsa2000_fm.abc import AbstractArrayConstraint
from dsa2000_fm.array_layout.sample_constraints import RegionSampler


class ArrayConstraintsV5Extended(BaseContent, AbstractArrayConstraint):
    """
    Abstract array class.
    """

    def __init__(self):
        # self.ellipses_version = ellipses_version
        BaseContent.__init__(self, seed='array_constraint_v6')

    def get_array_constraint_folder(self) -> str:
        return str(os.path.join(*self.content_path, 'shape_files'))

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
            (RegionSampler(os.path.join(folder, "Avoidance - Observed Rocky Conditions.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Avoidance - Terrain.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Environmental Avoidance (Combined).shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Fences.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Existing Access Paths.shp")), _30ft),
            (RegionSampler(os.path.join(folder, "aoi_NHD_V2_Flowlines.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "aoi_NHD_V2_Waterbodies.shp")), _200ft),
            (RegionSampler(os.path.join(folder, "aoi_NWI_Wetland.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Ephemeral Streams.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Intermittent Streams.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Perennial Streams- Observed.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "Water Bodies.shp")), _200ft)
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

        aoi_data = [
            # (RegionSampler(os.path.join(folder, "Western Expansion Areas.shp")), _60ft),
            (RegionSampler(os.path.join(folder, "AOI 3.1b.shp")), _60ft),
            # (RegionSampler(os.path.join(folder, "Eastern Expansion Areas.shp")), _60ft),
        ]
        # if self.ellipses_version == "A2":
        #     aoi_data.append((RegionSampler(os.path.join(folder, "Serving Areas (Ellipses) 5.0 A2.shp")), _60ft))
        # elif self.ellipses_version == "C":
        #     aoi_data.append((RegionSampler(os.path.join(folder, "Serving Areas (Ellipses) 5.0 C.shp")), _60ft))
        return aoi_data
