import os

import numpy as np
from astropy import coordinates as ac, units as au, time as at
from jax import config

# Set num jax devices
config.update("jax_enable_x64", True)
config.update('jax_threefry_partitionable', True)
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

from tomographic_kernel.frames import ENU
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.forward_models.lwa_forward_model import LWAForwardModel
from dsa2000_cal.forward_models.synthetic_sky_model.synthetic_sky_model_producer import SyntheticSkyModelProducer
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSetMetaV0, MeasurementSet


def main(ms_folder: str):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match('lwa'))

    if os.path.exists(ms_folder):
        ms = MeasurementSet(ms_folder)
    else:
        array_location = array.get_array_location()
        antennas = array.get_antennas()
        obstimes = at.Time("2021-01-01T00:00:00", scale='utc') + 10 * np.arange(1) * au.s
        freqs = au.Quantity([55], unit=au.MHz)
        phase_tracking = zenith = ENU(0, 0, 1, obstime=obstimes[0], location=array_location).transform_to(ac.ICRS())
        meta = MeasurementSetMetaV0(
            array_name='lwa',
            array_location=array_location,
            phase_tracking=phase_tracking,
            channel_width=array.get_channel_width(),
            integration_time=au.Quantity(10, 's'),
            coherencies=['XX', 'XY', 'YX', 'YY'],
            pointings=None,
            times=obstimes,
            freqs=freqs,
            antennas=antennas,
            antenna_names=array.get_antenna_names(),
            antenna_diameters=array.get_antenna_diameter(),
            with_autocorr=True,
            mount_types='ALT-AZ',
            system_equivalent_flux_density=array.get_system_equivalent_flux_density(),
            convention='physical'
        )
        ms = MeasurementSet.create_measurement_set(ms_folder=ms_folder, meta=meta)

    sky_model_producer = SyntheticSkyModelProducer(
        phase_tracking=ms.meta.phase_tracking,
        freqs=ms.meta.freqs,
        field_of_view=180 * au.deg
    )

    forward_model = LWAForwardModel(
        synthetic_sky_model_producer=sky_model_producer,
        run_folder='forward_model_lwa',
        num_shards=len(ms.meta.freqs),
        oversample_factor=7.,
        field_of_view=180 * au.deg,
        weighting='natural',
        epsilon=1e-6,
        add_noise=True,
        include_calibration=False,
        include_simulation=True
    )
    forward_model.forward(ms=ms)


if __name__ == '__main__':
    main(ms_folder='forward_model_lwa_ms')
