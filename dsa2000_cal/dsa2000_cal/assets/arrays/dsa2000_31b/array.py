import os

from dsa2000_cal.abc import AbstractAntennaModel
from dsa2000_cal.assets.arrays.dsa2000W.array import DSA2000WArray
from dsa2000_cal.assets.registries import array_registry


@array_registry(template='dsa2000_31b')
class DSA200031b(DSA2000WArray):
    """
    DSA2000W array class, smaller for testing
    """

    def get_array_file(self) -> str:
        return os.path.join(*self.content_path, 'antenna_config.txt')

    def get_antenna_model(self) -> AbstractAntennaModel:
        return DSA2000WArray.get_antenna_model(self)
        # return MatlabAntennaModelV1(
        #     antenna_model_file=os.path.join(*self.content_path, 'dsa2000_antenna_model.mat'),
        #     model_name='coPolPattern_dBi_Freqs_15DegConicalShield'
        # )


if __name__ == '__main__':
    import astropy.coordinates as ac
    import astropy.units as au

    lats = []
    lons = []
    elevations = []
    with open('antenna_config.txt', 'w') as g:
        g.write("#station,X,Y,Z\n")
        with open('dsa2000_3.1b.csv', 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                if line.strip() == "":
                    continue
                antenna_label, lat, lon, elevation = line.split(',')
                antenna = ac.EarthLocation.from_geodetic(lon=float(lon) * au.deg, lat=float(lat) * au.deg,
                                                         height=float(elevation) * au.m)
                # Convert to ITRS
                antenna_itrs = antenna.get_itrs()
                antenna_xyz = antenna_itrs.cartesian.xyz.to('m').value
                lats.append(lat)
                lons.append(lon)
                elevations.append(elevation)
                g.write(f"{antenna_label},{antenna_xyz[0]},{antenna_xyz[1]},{antenna_xyz[2]}\n")