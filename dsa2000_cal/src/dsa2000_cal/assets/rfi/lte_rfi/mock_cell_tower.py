import jax.numpy as jnp
import numpy as np
from astropy import units as au

from dsa2000_cal.assets.registries import rfi_model_registry
from dsa2000_cal.assets.rfi.rfi_emitter_model import RFIEmitterSourceModelParams, AbstractRFIEmitterData
from dsa2000_cal.common.astropy_utils import fraunhofer_far_field_limit
from dsa2000_common.common.interp_utils import InterpolatedArray
from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_common.visibility_model.source_models.rfi.parametric_rfi_emitter import ParametricDelayACF


@rfi_model_registry(template='mock_cell_tower')
class MockCellTower(AbstractRFIEmitterData):

    def make_source_params(self, freqs: au.Quantity, central_freq: au.Quantity | None = None,
                           bandwidth: au.Quantity | None = None,
                           full_stokes: bool = False) -> RFIEmitterSourceModelParams:
        # E=1
        delays = jnp.linspace(-1., 1., 100)  # [n_delays]
        auto_correlation_function = jnp.sin(delays)[:, None]  # [n_delays, 1]
        if np.allclose(np.diff(delays), delays[1] - delays[0], atol=1e-8):
            regular_grid = True
        else:
            regular_grid = False

        if central_freq is None:
            central_freq = np.mean(freqs)
        if bandwidth is None:
            bandwidth = 5 * au.MHz

        rfi_band_mask = np.logical_and(freqs >= central_freq - bandwidth / 2, freqs <= central_freq + bandwidth / 2)

        nominal_spectral_flux_density = (100 * au.Jy) * (1 * au.km) ** 2 * ((55 * au.MHz) / central_freq)
        spectral_flux_density = rfi_band_mask[None].astype(np.float32) * nominal_spectral_flux_density.to(
            'Jy*m^2')  # [1, num_chans]
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
        )  # [E,chan,[2,2]]

        return RFIEmitterSourceModelParams(
            freqs=freqs,
            position_enu=position_enu,
            delay_acf=delay_acf
        )


@rfi_model_registry(template='parametric_mock_cell_tower')
class MockCellTower(AbstractRFIEmitterData):

    def make_source_params(self, freqs: au.Quantity, central_freq: au.Quantity | None = None,
                           bandwidth: au.Quantity | None = None,
                           full_stokes: bool = False) -> RFIEmitterSourceModelParams:
        # E=1
        if central_freq is None:
            central_freq = np.mean(freqs)
        if bandwidth is None:
            bandwidth = 5 * au.MHz

        nominal_spectral_flux_density = (100 * au.Jy) * (1 * au.km) ** 2 / bandwidth
        nominal_spectral_flux_density = nominal_spectral_flux_density.to(
            'Jy*m^2/Hz')  # [1]
        if full_stokes:
            spectral_power = jnp.ones((1, 2, 2)) * quantity_to_jnp(nominal_spectral_flux_density,
                                                                   'Jy*m^2/Hz')  # [E=1, 2, 2]
        else:
            spectral_power = jnp.ones((1,)) * quantity_to_jnp(nominal_spectral_flux_density, 'Jy*m^2/Hz')

        # ENU coords
        # Far field limit would be around
        far_field_limit = fraunhofer_far_field_limit(diameter=2.7 * au.km, freq=central_freq)
        print(f"Far field limit: {far_field_limit} at {central_freq}")
        position_enu = au.Quantity([[1e3, 1e3, 120]], unit='m')  # [1, 3]

        delay_acf = ParametricDelayACF(
            mu=quantity_to_jnp(central_freq)[None],
            fwhp=quantity_to_jnp(bandwidth)[None],
            spectral_power=spectral_power,
            channel_lower=quantity_to_jnp(freqs - bandwidth / 2),
            channel_upper=quantity_to_jnp(freqs + bandwidth / 2)
        )  # [E,chan,[2,2]]

        return RFIEmitterSourceModelParams(
            freqs=freqs,
            position_enu=position_enu,
            delay_acf=delay_acf
        )
