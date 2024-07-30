import os

import jax.numpy as jnp
import numpy as np
from astropy import coordinates as ac, units as au, time as at
from jax import config
from tomographic_kernel.frames import ENU

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.forward_models.dsa2000_forward_model import DSA2000ForwardModel
from dsa2000_cal.forward_models.synthetic_sky_model.synthetic_sky_model_producer import SyntheticSkyModelProducer
from dsa2000_cal.forward_models.systematics.dish_effects_simulation import DishEffectsParams
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0, MeasurementSet

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
        obstimes = at.Time("2021-01-01T00:00:00", scale='utc') + 1.5 * np.arange(10) * au.s
        phase_tracking = zenith = ENU(0, 0, 1, obstime=obstimes[0], location=antennas[0]).transform_to(ac.ICRS())
        meta = MeasurementSetMetaV0(
            array_name='dsa2000W',
            array_location=array_location,
            phase_tracking=phase_tracking,
            channel_width=array.get_channel_width(),
            integration_time=au.Quantity(1.5, 's'),
            coherencies=['I'],
            pointings=phase_tracking,
            times=obstimes,
            freqs=au.Quantity([700], unit=au.MHz),
            antennas=antennas,
            antenna_names=array.get_antenna_names(),
            antenna_diameters=array.get_antenna_diameter(),
            with_autocorr=True,
            mount_types='ALT-AZ',
            system_equivalent_flux_density=array.get_system_equivalent_flux_density()
        )
        ms = MeasurementSet.create_measurement_set(ms_folder=ms_folder, meta=meta)

    sky_model_producer = SyntheticSkyModelProducer(
        phase_tracking=ms.meta.phase_tracking,
        freqs=ms.meta.freqs,
        field_of_view=2 * au.deg
    )

    forward_model = DSA2000ForwardModel(
        synthetic_sky_model_producer=sky_model_producer,
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
        solution_folder='forward_model_solution',
        num_shards=len(ms.meta.freqs),
        oversample_factor=5.,
        field_of_view=2 * au.deg,
        dtype=jnp.complex128
    )
    forward_model.forward(ms=ms)


if __name__ == '__main__':
    main(ms_folder='forward_model_dsa2000W_ms')
