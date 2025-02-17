import jax.numpy as jnp
import numpy as np
from astropy import units as au

from dsa2000_assets.base_content import BaseContent
from dsa2000_assets.registries import rfi_model_registry
from dsa2000_common.common.astropy_utils import fraunhofer_far_field_limit
from dsa2000_common.abc import AbstractRFIEmitterData
from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_common.visibility_model.source_models.rfi.abc import AbstractRFIAutoCorrelationFunction
from dsa2000_common.visibility_model.source_models.rfi.parametric_rfi_emitter import ParametricDelayACF


@rfi_model_registry(template='parametric_mock_cell_tower')
class MockCellTower(BaseContent, AbstractRFIEmitterData):

    def make_rfi_acf(self, freqs: au.Quantity, central_freq: au.Quantity | None = None,
                     bandwidth: au.Quantity | None = None,
                     full_stokes: bool = False) -> AbstractRFIAutoCorrelationFunction:
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

        delay_acf = ParametricDelayACF(
            mu=quantity_to_jnp(central_freq)[None],
            fwhp=quantity_to_jnp(bandwidth)[None],
            spectral_power=spectral_power,
            channel_lower=quantity_to_jnp(freqs - bandwidth / 2),
            channel_upper=quantity_to_jnp(freqs + bandwidth / 2)
        )  # [E,chan,[2,2]]

        return delay_acf
