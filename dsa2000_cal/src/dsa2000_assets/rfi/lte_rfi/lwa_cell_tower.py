import os

import jax.numpy as jnp
import numpy as np
import pylab as plt
from astropy import units as au
from scipy.io import loadmat

from dsa2000_assets.registries import rfi_model_registry
from dsa2000_common.abc import RFIEmitterSourceModelParams, AbstractRFIEmitterData
from dsa2000_cal.common.astropy_utils import fraunhofer_far_field_limit
from dsa2000_common.common.interp_utils import InterpolatedArray


@rfi_model_registry(template='lwa_cell_tower')
class LWACellTower(AbstractRFIEmitterData):

    def rfi_injection_model(self) -> str:
        return os.path.join(*self.content_path, 'rfi_injection_model.mat')

    def plot_acf(self):
        mat = loadmat(self.rfi_injection_model())
        delays = jnp.asarray(mat['t_acf'][0])
        auto_correlation_function = jnp.asarray(mat['acf']).T
        plt.plot(delays * 1e6, np.abs(auto_correlation_function))
        plt.xlabel('Delay [us]')
        plt.ylabel('ACF')
        plt.show()

    def make_source_params(self, freqs: au.Quantity, central_freq: au.Quantity | None = None,
                           bandwidth: au.Quantity | None = None,
                           full_stokes: bool = False) -> RFIEmitterSourceModelParams:
        # E=1
        mat = loadmat(self.rfi_injection_model())
        delays = jnp.asarray(mat['t_acf'][0])
        auto_correlation_function = jnp.asarray(mat['acf']).T  # [n_delays, E=1]
        if np.allclose(np.diff(delays), delays[1] - delays[0], atol=1e-8):
            regular_grid = True
        else:
            regular_grid = False

        if central_freq is None:
            central_freq = np.mean(freqs)
        if bandwidth is None:
            bandwidth = 5 * au.MHz

        rfi_band_mask = np.logical_and(freqs >= central_freq - bandwidth / 2, freqs <= central_freq + bandwidth / 2)

        nominal_spectral_flux_density = (100 * au.Jy) * (1 * au.km) ** 2
        spectral_flux_density = rfi_band_mask[None].astype(np.float32) * nominal_spectral_flux_density.to(
            'Jy*m^2')  # [E=1, num_chans]
        if full_stokes:
            spectral_flux_density = 0.5 * au.Quantity(
                np.stack(
                    [
                        np.stack([spectral_flux_density, 0 * spectral_flux_density], axis=-1),
                        np.stack([0 * spectral_flux_density, spectral_flux_density], axis=-1)
                    ],
                    axis=-2
                )
            )  # [E=1, num_chans, 2, 2]
            auto_correlation_function = auto_correlation_function[:, :, None, None,
                                        None] * spectral_flux_density  # [n_delays, E=1, num_chans, 2, 2]
        else:
            auto_correlation_function = auto_correlation_function[:, :,
                                        None] * spectral_flux_density  # [n_delays, E=1, num_chans]

        # ENU coords
        # Far field limit would be around
        far_field_limit = fraunhofer_far_field_limit(diameter=2.7 * au.km, freq=central_freq)
        print(f"Far field limit: {far_field_limit} at {central_freq}")
        position_enu = au.Quantity([[1e3, 1e3, 120]], unit='m')  # [1, 3]

        delay_acf = InterpolatedArray(
            x=delays, values=auto_correlation_function, axis=0, regular_grid=regular_grid
        )  # [E=1,chan[2,2]]

        return RFIEmitterSourceModelParams(
            freqs=freqs,
            position_enu=position_enu,
            delay_acf=delay_acf
        )
