import os

from h5parm import DataPack

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.bbs_sky_model import BBSSkyModel
from dsa2000_cal.dft import im_to_vis_with_gains

fill_registries()

if 'num_cpus' not in os.environ:
    num_cpus = os.cpu_count()
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cpus}"
else:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.environ.get('num_cpus')}"

import jax
from jax import numpy as jnp
import numpy as np
from dsa2000_cal.run_config import RunConfig
import astropy.units as au
from pyrap import tables as pt


def main(run_config: RunConfig):
    if run_config.bright_sky_model_bbs is None:
        raise ValueError("Bright sky model must be specified to run ionosphere simulation.")

    bbs_sky_model = BBSSkyModel(bbs_sky_model=run_config.bright_sky_model_bbs,
                                pointing_centre=run_config.pointing_centre,
                                chan0=run_config.start_freq_hz * au.hz,
                                chan_width=run_config.channel_width_hz * au.hz,
                                num_channels=run_config.num_channels
                                )

    source_model = bbs_sky_model.get_source()
    with pt.table(run_config.dft_visibilities_path, readonly=True) as vis_table:
        antenna_1 = vis_table.getcol('ANTENNA1')
        antenna_2 = vis_table.getcol('ANTENNA2')
        times, time_idx = np.unique(vis_table.getcol('TIME'), return_inverse=True)
        uvw = vis_table.getcol('UVW')

    with DataPack(run_config.ionosphere_h5parm, readonly=True) as dp:
        assert dp.axes_order == ['pol', 'dir', 'ant', 'freq', 'time']
        dp.current_solset = 'sol000'
        dp.select(pol=slice(0, 1, 1))
        phase, axes = dp.phase
        _, Nd, Na, Nf, Nt = phase.shape
        phase = np.transpose(phase, (4, 2, 1, 3, 0))  # [time, ant, dir, freq, pol]
        gains = np.zeros((Nt, Na, Nd, Nf, 2, 2), dtype=np.complex64)
        gains[..., 0, 0] = np.exp(1j * phase)
        gains[..., 1, 1] = gains[..., 0, 0]
        # if amplitude is present, multiply by it
        if 'amplitude000' in dp.soltabs:
            amplitude, axes = dp.amplitude
            amplitude = np.transpose(amplitude, (4, 2, 1, 3, 0))  # [time, ant, dir, freq, pol]
            gains[..., 0, 0] *= amplitude
            gains[..., 1, 1] *= amplitude
        else:
            print(f"Amplitude not present in h5parm.")

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
