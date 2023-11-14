import os

if 'num_cpus' not in os.environ:
    num_cpus = os.cpu_count()
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cpus}"
else:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.environ.get('num_cpus')}"

from dsa2000_cal.assets.content_registry import fill_registries

fill_registries()

from dsa2000_cal.bbs_sky_model import BBSSkyModel
from dsa2000_cal.dft import im_to_vis_with_gains
from dsa2000_cal.gains import extract_scalar_gains
from dsa2000_cal.run_config import RunConfig

import jax
from jax import numpy as jnp
import numpy as np
import astropy.units as au
from pyrap import tables as pt


def main(run_config: RunConfig):
    if run_config.bright_sky_model_bbs is None:
        raise ValueError("Bright sky model must be specified to run ionosphere simulation.")

    bbs_sky_model = BBSSkyModel(bbs_sky_model=run_config.bright_sky_model_bbs,
                                pointing_centre=run_config.pointing_centre,
                                chan0=run_config.start_freq_hz * au.Hz,
                                chan_width=run_config.channel_width_hz * au.Hz,
                                num_channels=run_config.num_channels
                                )

    source_model = bbs_sky_model.get_source()
    with pt.table(run_config.dft_visibilities_path, readonly=True) as vis_table:
        antenna_1 = vis_table.getcol('ANTENNA1')
        antenna_2 = vis_table.getcol('ANTENNA2')
        times, time_idx = np.unique(vis_table.getcol('TIME'), return_inverse=True)
        uvw = vis_table.getcol('UVW')

    ionosphere_gains = extract_scalar_gains(h5parm=run_config.ionosphere_h5parm, components=['phase'])  # [time, ant, source, chan, 2, 2]
    beam_gains = extract_scalar_gains(h5parm=run_config.beam_h5parm, components=['amplitude'])  # [time, ant, source, chan, 2, 2]
    # Multiply diagonal matrices with simple * operator
    gains = ionosphere_gains * beam_gains # [time, ant, source, chan, 2, 2]

    # Uncomment for no gains
    # gains = np.tile(np.eye(2)[None, None, None, None, :, :], gains.shape[:-2] + (1, 1))

    vis = im_to_vis_with_gains(
        image=jnp.asarray(source_model.image),  # [source, chan, 2, 2]
        gains=jnp.asarray(gains),  # [time, ant, source, chan, 2, 2]
        antenna_1=jnp.asarray(antenna_1),  # [row]
        antenna_2=jnp.asarray(antenna_2),  # [row]
        time_idx=jnp.asarray(time_idx),  # [row]
        uvw=jnp.asarray(uvw),  # [row, 3]
        lm=jnp.asarray(source_model.lm),  # [source, 2]
        frequency=jnp.asarray(source_model.freqs),  # [chan]
        convention='fourier',
        chunksize=len(jax.devices())
    )  # [row, chan, 2, 2]
    row, chan, _, _ = vis.shape
    with pt.table(run_config.dft_visibilities_path, readonly=False) as vis_table:
        dtype = vis_table.getcol('DATA').dtype
        vis_table.putcol('DATA', np.reshape(np.asarray(vis, dtype=dtype), (row, chan, 4)))


if __name__ == '__main__':
    if not os.path.exists('run_config.json'):
        raise ValueError('run_config.json must be present in the current directory.')
    run_config = RunConfig.parse_file('run_config.json')
    main(run_config)
