import os

import numpy as np
from astropy import coordinates as ac, units as au, time as at
from jax import config

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.forward_models.dsa2000_forward_model import DSA2000ForwardModel
from dsa2000_cal.forward_models.synthetic_sky_model import SyntheticSkyModelProducer
from dsa2000_cal.common.types import DishEffectsParams
from dsa2000_cal.measurement_sets.measurement_set import  MeasurementSetMeta, MeasurementSet

# Set num jax devices
config.update("jax_enable_x64", True)
config.update('jax_threefry_partitionable', True)
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"


def main(ms_folder: str):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('dsa2000W'))

    if os.path.exists(ms_folder):
        ms = MeasurementSet(ms_folder)
    else:
        array_location = array.get_array_location()
        antennas = array.get_antennas()
        meta = MeasurementSetMeta(
            array_name='dsa2000W',
            array_location=array_location,
            phase_center=ac.ICRS(0 * au.deg, 0 * au.deg),
            channel_width=array.get_channel_width(),
            integration_time=au.Quantity(1.5, 's'),
            coherencies=['XX', 'XY', 'YX', 'YY'],
            pointings=ac.ICRS(0 * au.deg, 0 * au.deg),
            times=at.Time("2021-01-01T00:00:00", scale='utc') + 1.5 * np.arange(1) * au.s,
            freqs=au.Quantity(np.linspace(700, 2000, 8000)[:32], unit=au.MHz),
            antennas=antennas,
            antenna_names=array.get_antenna_names(),
            antenna_diameters=array.get_antenna_diameter(),
            with_autocorr=True,
            mount_types='ALT-AZ',
            system_equivalent_flux_density=array.get_system_equivalent_flux_density()
        )
        ms = MeasurementSet.create_measurement_set(ms_folder=ms_folder, meta=meta)

    sky_model_producer = SyntheticSkyModelProducer(
        phase_center=ms.meta.pointings,
        obs_time=ms.ref_time,
        freqs=ms.meta.freqs,
        num_bright_sources=7,
        num_faint_sources=7,
        field_of_view=2 * au.deg
    )
    sky_model = sky_model_producer.create_sky_model()

    with open('sky_model.json', 'w') as fp:
        fp.write(sky_model.json(indent=2))

    sky_model_source_models = sky_model.to_wsclean_source_models()

    sky_model_calibrators = sky_model_producer.create_sky_model(include_faint=False)

    with open('sky_model_calibrators.json', 'w') as fp:
        fp.write(sky_model_calibrators.json(indent=2))

    sky_model_calibrators_source_models = sky_model_calibrators.to_wsclean_source_models()

    forward_model = DSA2000ForwardModel(
        dish_effect_params=DishEffectsParams(
            dish_diameter=array.get_antenna_diameter(),
            focal_length=array.get_focal_length(),
            # elevation_pointing_error_stddev=0. * au.deg,
            # cross_elevation_pointing_error_stddev=0. * au.deg,
            # axial_focus_error_stddev=0. * au.m,
            # elevation_feed_offset_stddev=0. * au.m,
            # cross_elevation_feed_offset_stddev=0. * au.m,
            # horizon_peak_astigmatism_stddev=0. * au.m,
            # surface_error_mean=0. * au.m,
            # surface_error_stddev=0. * au.m
        ),
        ionosphere_specification='light_dawn',
        plot_folder='forward_model_plots',
        cache_folder='forward_model_cache',
        simulation_wsclean_source_models=sky_model_source_models,
        calibration_wsclean_source_models=sky_model_calibrators_source_models,
        simulation_fits_source_models=[],
        calibration_fits_source_models=[],
        num_shards=len(ms.meta.freqs),
        oversample_factor=1.5,
        field_of_view=4 * au.deg,
    )
    forward_model.forward(ms=ms)


if __name__ == '__main__':
    ms_folder = os.environ.get('MEASUREMENT_SET')
    main(ms_folder)
