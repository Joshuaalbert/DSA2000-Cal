import os

if 'num_cpus' not in os.environ:
    num_cpus = os.cpu_count()
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cpus}"
else:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.environ.get('num_cpus')}"
from dsa2000_cal.assets.content_registry import fill_registries

fill_registries()
from dsa2000_cal.run_config import RunConfig
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.ionosphere.ionosphere_simulation import Simulation
import astropy.time as at


def main(run_config: RunConfig):
    if run_config.bright_sky_model_bbs is None:
        raise ValueError("Bright sky model must be specified to run ionosphere simulation.")
    array = array_registry.get_instance(array_registry.get_match(run_config.array_name))

    sim = Simulation(
        specification=run_config.ionosphere_specification,
        S_marg=25
    )

    duration = (run_config.num_times - 1) * run_config.integration_time_s
    start_time_mjd = at.Time(run_config.start_dt.isoformat(), format='isot').mjd

    sim.run(output_h5parm=run_config.ionosphere_h5parm,
            duration=duration,
            time_resolution=run_config.integration_time_s,
            start_time=start_time_mjd,
            array=array,
            pointing_centre=run_config.pointing_centre,
            start_freq_hz=run_config.start_freq_hz,
            channel_width_hz=run_config.channel_width_hz,
            num_channels=run_config.num_channels,
            sky_model=run_config.bright_sky_model_bbs,
            grid_res_m=1000.)


if __name__ == '__main__':
    if not os.path.exists('run_config.json'):
        raise ValueError('run_config.json must be present in the current directory.')
    run_config = RunConfig.parse_file('run_config.json')
    main(run_config)
