import os

from jax import config

from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel

config.update("jax_enable_x64", True)
# Set num jax devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import source_model_registry
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet


def main(ms_folder: str):
    ms = MeasurementSet(ms_folder=ms_folder)

    # Create a sky model for calibration
    fill_registries()
    cas_a_source_file = source_model_registry.get_instance(
        source_model_registry.get_match('cas_a')).get_wsclean_clean_component_file()
    cas_a_sky_model = WSCleanSourceModel.from_wsclean_model(
        wsclean_clean_component_file=cas_a_source_file,
        freqs=ms.meta.freqs,
        time=ms.meta.times[0],
        phase_tracking=ms.meta.phase_tracking
    )
    cyg_a_source_file = source_model_registry.get_instance(
        source_model_registry.get_match('cyg_a')).get_wsclean_clean_component_file()
    cyg_a_sky_model = WSCleanSourceModel.from_wsclean_model(
        wsclean_clean_component_file=cyg_a_source_file,
        freqs=ms.meta.freqs,
        time=ms.meta.times[0],
        phase_tracking=ms.meta.phase_tracking
    )
    sky_model = cas_a_sky_model + cyg_a_sky_model

    # Run calibration
    ...

    # Save the subtracted visibilities
    ...

    beam_gain_model = beam_gain_model_factory(ms.meta.array_name)
    calibration = Calibration(
        num_iterations=15,
        wsclean_source_models=self.calibration_wsclean_source_models,
        fits_source_models=self.calibration_fits_source_models,
        preapply_gain_model=beam_gain_model,
        inplace_subtract=False,
        residual_ms_folder='residual_ms',
        average_interval=None,
        solution_cadence=None,
        verbose=self.verbose,
        seed=self.calibration_seed,
        num_shards=self.num_shards,
        plot_folder=os.path.join(self.plot_folder, 'calibration')
    )

    return calibration.calibrate(ms=ms)


if __name__ == '__main__':
    ms_folder = os.environ.get('MEASUREMENT_SET')
    main(ms_folder)
