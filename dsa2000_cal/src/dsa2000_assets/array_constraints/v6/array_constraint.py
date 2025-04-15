import os
from typing import List, Tuple

from matplotlib import pyplot as plt

from dsa2000_assets.base_content import BaseContent
from dsa2000_fm.abc import AbstractArrayConstraint
from dsa2000_fm.array_layout.sample_constraints import RegionSampler


class ArrayConstraintsV6(BaseContent, AbstractArrayConstraint):
    """
    Abstract array class.
    """

    def __init__(self, extension: str):
        # self.ellipses_version = ellipses_version
        BaseContent.__init__(self, seed='array_constraint_v6')
        self.extension = extension

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
            (RegionSampler(os.path.join(folder, "Avoidance - PHMA within AOI.shp")), _60ft),
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

        if self.extension == 'full':
            aoi_data = [
                (RegionSampler(os.path.join(folder, f"AOI 6.1.shp")), _60ft),
            ]
        else:
            aoi_data = [
                (RegionSampler(os.path.join(folder, f"AOI 6.1{self.extension}.shp")), _60ft),
            ]
        return aoi_data


def _test_merge_aoi():
    aoi = ArrayConstraintsV6(extension='full')
    aoi.get_area_of_interest_regions()
    merged_aoi = RegionSampler.merge([sampler for sampler, _ in aoi.get_area_of_interest_regions()])
    merged_aoi.info()
    area = sum([sampler.total_area for sampler, _ in aoi.get_area_of_interest_regions()])
    assert merged_aoi.total_area <= area
    fig, axs = plt.subplots(2, 1)
    merged_aoi.plot_region(axs[0])
    for sampler, _ in aoi.get_area_of_interest_regions():
        sampler.plot_region(axs[1])
    plt.show()


if __name__ == '__main__':
    _test_merge_aoi()
