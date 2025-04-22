import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np

from dsa2000_assets.content_registry import fill_registries
from dsa2000_assets.registries import source_model_registry, array_registry
from dsa2000_common.common.astropy_utils import mean_itrs
from dsa2000_common.common.enu_frame import ENU
from dsa2000_common.visibility_model.source_models.celestial.base_fits_source_model import \
    build_calibration_fits_source_models_from_wsclean


def main():
    fill_registries()

    array = array_registry.get_instance(array_registry.get_match('dsa1650_a35'))
    antennas = array.get_antennas()

    antennas: ac.EarthLocation
    array_location = mean_itrs(antennas.get_itrs()).earth_location
    wsclean_fits_files = source_model_registry.get_instance(
        source_model_registry.get_match('ncg_5194')).get_wsclean_fits_files()
    obsfreqs = np.linspace(700, 2000, 10000) * au.MHz

    obstime = at.Time("2025-06-10T00:00:00", format='isot', scale='utc')

    pointing = ENU(0, 0, 1, obstime=obstime, location=array_location).transform_to(ac.ICRS())

    source_model = build_calibration_fits_source_models_from_wsclean(
        wsclean_fits_files=wsclean_fits_files,
        model_freqs=obsfreqs,
        full_stokes=False,
        repoint_centre=pointing,
        crop_box_size=1 * au.arcmin,
        num_facets=1
    )
    print(source_model)
    print(antennas)

    # vis = image_to_vis_np(
    #
    # )


if __name__ == '__main__':
    main()
