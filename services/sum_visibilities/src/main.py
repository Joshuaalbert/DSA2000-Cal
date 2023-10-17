import os

from dsa2000_cal.assets.content_registry import fill_registries

fill_registries()
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.noise import sum_and_add_noise
from dsa2000_cal.run_config import RunConfig


def main(run_config: RunConfig):
    array = array_registry.get_instance(array_registry.get_match(run_config.array_name))
    sum_and_add_noise(
        output_ms_file=run_config.visibilities_path,
        input_ms_files=[
            run_config.dft_visibilities_path,
            run_config.fft_visibilities_path,
            run_config.rfi_visibilities_path
        ],
        array=array,
        channel_width_hz=run_config.channel_width_hz,
        integration_time_s=run_config.integration_time_s
    )


if __name__ == '__main__':
    if not os.path.exists('run_config.json'):
        raise ValueError('run_config.json must be present in the current directory.')
    run_config = RunConfig.parse_file('run_config.json')
    main(run_config)
