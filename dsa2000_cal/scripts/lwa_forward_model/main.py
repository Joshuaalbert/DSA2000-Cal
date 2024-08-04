import os

import jax.numpy as jnp
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
        obstimes = at.Time("2021-01-01T00:00:00", scale='utc') + 60 * np.arange(1) * au.s
        phase_tracking = zenith = ENU(0, 0, 1, obstime=obstimes[0], location=array_location).transform_to(ac.ICRS())
        meta = MeasurementSetMetaV0(
            array_name='lwa',
            array_location=array_location,
            phase_tracking=phase_tracking,
            channel_width=array.get_channel_width(),
            integration_time=au.Quantity(60, 's'),
            coherencies=['I'],
            pointings=None,
            times=obstimes,
            freqs=au.Quantity([55], unit=au.MHz),
            antennas=antennas,
            antenna_names=array.get_antenna_names(),
            antenna_diameters=array.get_antenna_diameter(),
            with_autocorr=True,
            mount_types='ALT-AZ',
            system_equivalent_flux_density=array.get_system_equivalent_flux_density()
        )
        ms = MeasurementSet.create_measurement_set(ms_folder=ms_folder, meta=meta)

    # # propagation delay
    # rfi_model = LWACellTower(seed='abc')
    # rfi_source = rfi_model.make_source_params(freqs=ms.meta.freqs, full_stokes=False)
    # print(ms.near_field_delay_engine.x_antennas_gcrs)
    # print(ms.near_field_delay_engine.enu_coords_gcrs)
    # # return
    # for i2 in range(len(ms.meta.antennas)):
    #     delay, dist20, dist10 = ms.near_field_delay_engine.compute_delay_from_projection_jax(
    #         a_east=0.,
    #         a_north=0.,
    #         a_up=20e3,
    #         t1=ms.time_to_jnp(ms.meta.times[0]),
    #         i1=0,
    #         i2=i2
    #     )  # [], [], []
    #     delay_s = delay / 299792458.
    #     print(f"Delay 0 to {i2}: {delay} m ({delay_s} s), dist20: {dist20}, dist10: {dist10}")
    #     acf_val = rfi_source.delay_acf(delay_s)
    #     print(f"ACF value: {acf_val}, angle: {np.angle(acf_val)}")
    #
    # return

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
        dtype=jnp.complex128,
        weighting='natural',
        epsilon=1e-6,
        add_noise=False
    )
    forward_model.forward(ms=ms)


if __name__ == '__main__':
    main(ms_folder='forward_model_lwa_ms')
