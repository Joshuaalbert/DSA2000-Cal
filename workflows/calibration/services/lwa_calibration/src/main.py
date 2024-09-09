import os

from jax import config

from dsa2000_cal.adapter.from_casa_ms import transfer_from_casa
from dsa2000_cal.antenna_model.antenna_model_utils import get_dish_model_beam_widths
from dsa2000_cal.calibration.probabilistic_models.gain_prior_models import UnconstrainedGain
from dsa2000_cal.forward_models.synthetic_sky_model import SyntheticSkyModelProducer
from dsa2000_cal.imaging.imagor import Imagor
from dsa2000_cal.visibility_model.rime_model import RIMEModel

config.update("jax_enable_x64", True)
config.update('jax_threefry_partitionable', True)
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"
import numpy as np
from dsa2000_cal.calibration.calibration import Calibration
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry

import astropy.units as au


def main(casa_ms: str, ms_folder: str, array_name: str):
    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))

    ms = transfer_from_casa(
        ms_folder=ms_folder,
        casa_ms=casa_ms,
        convention='engineering'
    )
    print(f"Created {ms}")

    # Create a sky model for calibration
    freqs, field_of_view = get_dish_model_beam_widths(array.get_antenna_model())
    field_of_view = np.mean(field_of_view)
    sky_model_producer = SyntheticSkyModelProducer(
        phase_tracking=ms.meta.pointings,
        obs_time=ms.ref_time,
        freqs=ms.meta.freqs,
        num_bright_sources=0,
        num_faint_sources=0,
        field_of_view=field_of_view,
        a_team_sources=['cas_a', 'cyg_a', 'tau_a', 'vir_a']
    )
    sky_model = sky_model_producer.create_sky_model(include_a_team=True)

    beam_gain_model = build_beam_gain_model(array_name)

    gain_prior_model = UnconstrainedGain()

    rime_model = RIMEModel(
        sky_model=sky_model,
        convention=ms.meta.convention
    )
    num_shards = len(ms.meta.freqs)
    calibration = Calibration(
        sky_model=sky_model,
        preapply_gain_model=beam_gain_model,
        num_iterations=15,
        inplace_subtract=True,
        plot_folder='plots/calibration',
        validity_interval=None,
        solution_interval=10 * au.s,
        num_shards=num_shards,
        seed=56789,
        probabilistic_model=gain_prior_model,
        rime_model=rime_model
    )
    subtracted_ms = calibration.calibrate(ms=ms)

    # Save the subtracted visibilities
    imagor = Imagor(
        plot_folder='plots/imaging_residuals',
        field_of_view=field_of_view,
        seed=12345,
        nthreads=1
    )
    imagor.image(image_name='residuals', ms=subtracted_ms)


if __name__ == '__main__':
    casa_ms = os.environ.get('INPUT_CASA_MS')
    output_ms_folder = os.environ.get('OUTPUT_MEASUREMENT_SET')
    main(casa_ms=casa_ms, ms_folder=output_ms_folder, array_name='lwa')
