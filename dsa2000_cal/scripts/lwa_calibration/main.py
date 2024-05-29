import os

import jax
from jax import config

from dsa2000_cal.adapter.from_casa_ms import transfer_from_casa
from dsa2000_cal.antenna_model.utils import get_dish_model_beam_widths
from dsa2000_cal.calibration.gain_prior_models import UnconstrainedGain
from dsa2000_cal.forward_model.sky_model import SkyModel
from dsa2000_cal.imaging.dirty_imaging import DirtyImaging
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel

config.update("jax_enable_x64", True)
config.update('jax_threefry_partitionable', True)
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"
import numpy as np
from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry, source_model_registry


def main(casa_ms: str, ms_folder: str, array_name: str):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))

    ms = transfer_from_casa(
        ms_folder=ms_folder,
        casa_ms=casa_ms
    )
    print(f"Created {ms}")

    # Create a sky model for calibration
    source_models = []
    for source in ['cas_a', 'cyg_a', 'tau_a', 'vir_a']:
        source_model_asset = source_model_registry.get_instance(source_model_registry.get_match(source))
        source_model = WSCleanSourceModel.from_wsclean_model(
            wsclean_clean_component_file=source_model_asset.get_wsclean_clean_component_file(),
            time=ms.ref_time,
            freqs=ms.meta.freqs,
            phase_tracking=ms.meta.phase_tracking
        )
        source_models.append(source_model)

    sky_model = SkyModel(component_models=source_models, fits_models=[])

    beam_gain_model = beam_gain_model_factory(array_name, zenith_pointing=True)
    num_shards = len(jax.devices())
    while len(ms.meta.freqs) % num_shards != 0:
        num_shards -= 1
    print(f"Using {num_shards} shards")

    calibration = Calibration(
        sky_model=sky_model,
        preapply_gain_model=beam_gain_model,
        num_iterations=15,
        inplace_subtract=True,
        plot_folder='plots/calibration',
        validity_interval=None,
        solution_interval=None,
        num_shards=num_shards,
        seed=56789,
        gain_probabilistic_model=UnconstrainedGain()
    )
    subtracted_ms = calibration.calibrate(ms=ms)

    # Save the subtracted visibilities
    _, field_of_view = get_dish_model_beam_widths(array.get_antenna_model())
    field_of_view = np.mean(field_of_view)
    imagor = DirtyImaging(
        plot_folder='plots/imaging_residuals',
        field_of_view=field_of_view,
        seed=12345,
        nthreads=1
    )
    return imagor.image(image_name='residuals', ms=subtracted_ms)


if __name__ == '__main__':
    casa_ms = "/home/albert/data/forward_modelling/data_dir/lwa01.ms"
    output_ms_folder = '/home/albert/data/forward_modelling/lwa01_run_dir_002'
    main(casa_ms=casa_ms, ms_folder=output_ms_folder, array_name='lwa')
