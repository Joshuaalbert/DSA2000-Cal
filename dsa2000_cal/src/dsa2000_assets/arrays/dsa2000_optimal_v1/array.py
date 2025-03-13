import os

from dsa2000_fm.antenna_model.abc import AbstractAntennaModel
from dsa2000_assets.arrays.dsa2000W.array import DSA2000WArray
from dsa2000_assets.registries import array_registry, beam_model_registry


@array_registry(template='dsa2000_optimal_v1')
class DSA2000OptimalV1(DSA2000WArray):
    """
    DSA2000W array class, optimised array layout.
    """

    def get_array_file(self) -> str:
        return os.path.join(*self.content_path, 'antenna_config.txt')

    def get_antenna_model(self) -> AbstractAntennaModel:
        beam_model = beam_model_registry.get_instance(beam_model_registry.get_match('dsa_nominal'))
        return beam_model.get_antenna_model()


def add_station_names():
    # add station names
    idx = 0
    with open('antenna_config.txt', 'r') as g:
        with open('antenna_config_named.txt', 'w') as f:
            f.write("#station,X,Y,Z\n")
            for line in g:
                if line.startswith("#"):
                    continue
                if line.strip() == "":
                    continue
                antenna_label = f"dsa-{idx:04d}"
                f.write(f"{antenna_label},{line}")
                idx += 1
