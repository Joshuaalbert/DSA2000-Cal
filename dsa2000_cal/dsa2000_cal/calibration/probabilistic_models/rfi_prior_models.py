import dataclasses
from abc import ABC, abstractmethod

import astropy.units as au
import jax
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jaxns import PriorModelType, Prior

from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.types import mp_policy
from dsa2000_cal.gain_models.beam_gain_model import BeamGainModel
from dsa2000_cal.geodesics.geodesic_model import GeodesicModel
from dsa2000_cal.visibility_model.source_models.rfi.rfi_emitter_source_model import RFIEmitterModelData
from dsa2000_cal.visibility_model.source_models.rfi.parametric_rfi_emitter import ParametricDelayACF

tfpd = tfp.distributions


class AbstractRFIPriorModel(ABC):
    @abstractmethod
    def build_prior_model(self, freqs: jax.Array, times: jax.Array) -> PriorModelType:
        """
        Define the prior model for the gains.

        Args:
            freqs: [num_chan] the frequencies
            times: [num_time] the times to compute the model data, in TT since start of observation

        Returns:
            rfi_emitter_data: .
        """
        ...


@dataclasses.dataclass(eq=False)
class FullyParameterisedRFIHorizonEmitter(AbstractRFIPriorModel):
    """
    A RFI model with fully parameterised spectral and spatial properties.
    """
    geodesic_model: GeodesicModel
    beam_gain_model: BeamGainModel
    num_emitters: int = 1
    acf_resolution: int = 10
    distance_min: au.Quantity = 0. * au.km
    distance_max: au.Quantity = 10. * au.km
    azimuth_min: au.Quantity = 0. * au.deg
    azimuth_max: au.Quantity = 360. * au.deg
    height_min: au.Quantity = 0. * au.m
    height_max: au.Quantity = 120. * au.m
    luminosity_min: au.Quantity = 10 * (au.Jy * au.km ** 2)  # Jy * m^2
    luminosity_max: au.Quantity = 100 * (au.Jy * au.km ** 2)  # Jy * m^2
    full_stokes: bool = True

    def __post_init__(self):
        if not self.distance_min.unit.is_equivalent("km"):
            raise ValueError("distance_min must be in km")
        if not self.distance_max.unit.is_equivalent("km"):
            raise ValueError("distance_max must be in km")
        if not self.azimuth_min.unit.is_equivalent("deg"):
            raise ValueError("azimuth_min must be in deg")
        if not self.azimuth_max.unit.is_equivalent("deg"):
            raise ValueError("azimuth_max must be in deg")
        if not self.height_min.unit.is_equivalent("m"):
            raise ValueError("height_min must be in m")
        if not self.height_max.unit.is_equivalent("m"):
            raise ValueError("height_max must be in m")
        if not self.luminosity_min.unit.is_equivalent("W / MHz"):
            raise ValueError("luminosity_min must be in W / MHz")
        if not self.luminosity_max.unit.is_equivalent("W / MHz"):
            raise ValueError("luminosity_max must be in W / MHz")

    def get_source_enu(self):
        ones = jnp.ones((self.num_emitters,), dtype=mp_policy.length_dtype)
        distance = yield Prior(
            tfpd.Uniform(
                low=quantity_to_jnp(self.distance_min) * ones,
                high=quantity_to_jnp(self.distance_max) * ones
            ),
            name='distance'
        ).parametrised()
        azimuth = yield Prior(
            tfpd.Uniform(
                low=quantity_to_jnp(self.azimuth_min) * ones,
                high=quantity_to_jnp(self.azimuth_max) * ones
            ),
            name='azimuth'
        ).parametrised()
        height = yield Prior(
            tfpd.Uniform(
                low=quantity_to_jnp(self.height_min) * ones,
                high=quantity_to_jnp(self.height_max) * ones
            ),
            name='height'
        ).parametrised()
        # Convert to ENU, azimuth is measured East of North
        east = distance * jnp.sin(azimuth)
        north = distance * jnp.cos(azimuth)
        up = height
        source_positions_enu = jnp.stack([east, north, up], axis=-1)  # [E, 3]
        return source_positions_enu

    def get_acf(self):
        max_delay = 1e-5  # seconds
        delay_acf_x = jnp.linspace(0., max_delay, self.acf_resolution)
        delay_acf_x = jnp.concatenate([-delay_acf_x[::-1], delay_acf_x[1:]])
        delay_acf_values_real = yield Prior(
            tfpd.Uniform(
                low=-0.5 * jnp.ones((self.acf_resolution, self.num_emitters)),
                high=1. * jnp.ones((self.acf_resolution, self.num_emitters))
            ),
            name='delay_acf_values_real'
        ).parametrised()
        delay_acf_values_imag = yield Prior(
            tfpd.Uniform(
                low=-1. * jnp.ones((self.acf_resolution, self.num_emitters)),
                high=1. * jnp.ones((self.acf_resolution, self.num_emitters))
            ),
            name='delay_acf_values_imag'
        ).parametrised()  # [num_delays, num_emitters]
        delay_acf_values = jax.lax.complex(delay_acf_values_real, delay_acf_values_imag)
        delay_acf_values /= delay_acf_values[0:1, :]  # normalise central value to 1
        delay_acf_values = jnp.concatenate([delay_acf_values[::-1], delay_acf_values[1:]], axis=0)
        delay_acf = InterpolatedArray(
            x=delay_acf_x,
            values=delay_acf_values,
            axis=0,
            regular_grid=True
        )  # [ E]
        return delay_acf

    def get_spectral_power(self, freqs: jax.Array):
        if self.full_stokes:
            luminosity = yield Prior(
                tfpd.Uniform(
                    low=quantity_to_jnp(self.luminosity_min, 'Jy*m^2') * jnp.ones((self.num_emitters, 2, 2)),
                    high=quantity_to_jnp(self.luminosity_max, 'Jy*m^2') * jnp.ones((self.num_emitters, 2, 2))
                ),
                name='luminosity'
            ).parametrised()
            luminosity = jnp.tile(luminosity[:, None, :, :], (1, len(freqs), 1, 1))  # [num_source, num_chan, 2, 2]
        else:
            luminosity = yield Prior(
                tfpd.Uniform(
                    low=quantity_to_jnp(self.luminosity_min, 'Jy*m^2') * jnp.ones((self.num_emitters)),
                    high=quantity_to_jnp(self.luminosity_max, 'Jy*m^2') * jnp.ones((self.num_emitters))
                ),
                name='luminosity'
            ).parametrised()
            luminosity = jnp.tile(luminosity[:, None], (1, len(freqs)))  # [num_source, num_chan]
        return luminosity

    def build_prior_model(self, freqs: jax.Array, times: jax.Array) -> PriorModelType:
        def prior_model():
            source_positions_enu = yield from self.get_source_enu()
            delay_acf = yield from self.get_acf()
            luminosity = yield from self.get_spectral_power(freqs)

            geodesics = self.geodesic_model.compute_near_field_geodesics(
                times=times,
                source_positions_enu=source_positions_enu
            )
            gains = self.beam_gain_model.compute_gain(
                freqs=freqs,
                times=times,
                geodesics=geodesics
            )
            return RFIEmitterModelData(
                freqs=freqs,
                position_enu=source_positions_enu,
                luminosity=luminosity,
                delay_acf=delay_acf,
                gains=gains
            )

        return prior_model





@dataclasses.dataclass(eq=False)
class ParametricRFIHorizonEmitter(FullyParameterisedRFIHorizonEmitter):
    """
    A RFI model with spectral parametrisation.
    """
    mu_low: au.Quantity = 700 * au.MHz
    mu_high: au.Quantity = 900 * au.MHz
    fwhm_low: au.Quantity = 10 * au.kHz
    channel_width: au.Quantity = 130 * au.kHz
    min_channel_power: au.Quantity = 0. * au.Jy * au.km ** 2
    max_channel_power: au.Quantity = 100 * au.Jy * au.km ** 2
    convention: str = 'physical'

    def __post_init__(self):
        super().__post_init__()
        self.fwhm_high = self.channel_width

    def get_acf(self):
        ones = jnp.ones((self.num_emitters,), dtype=mp_policy.freq_dtype)
        mu = yield Prior(
            tfpd.Uniform(
                low=quantity_to_jnp(self.mu_low) * ones,
                high=quantity_to_jnp(self.mu_high) * ones
            ),
            name='mu'
        ).parametrised()
        fwhp = yield Prior(
            tfpd.Uniform(
                low=quantity_to_jnp(self.fwhm_low) * ones,
                high=quantity_to_jnp(self.fwhm_high) * ones
            ),
            name='fwhp'
        ).parametrised()
        if self.full_stokes:
            eye_diag = jnp.ones((self.num_emitters, 2), dtype=mp_policy.length_dtype)
            spectral_power = yield Prior(
                tfpd.Uniform(
                    low=quantity_to_jnp(self.min_channel_power / self.channel_width, 'Jy*m^2/Hz') * eye_diag,
                    high=quantity_to_jnp(self.max_channel_power / self.channel_width, 'Jy*m^2/Hz') * eye_diag
                ),
                name='spectral_power'
            ).parametrised()
            spectral_power = jax.vmap(jnp.diag)(spectral_power)  # [E, 2, 2]
            pol_angle = yield Prior(
                tfpd.Uniform(
                    low=0. * ones,
                    high=2. * jnp.pi * ones
                ),
                name='pol_angle'
            ).parametrised()  # [E]
            # Rotate the spectral power
            R = jnp.stack(
                [
                    jnp.stack([jnp.cos(pol_angle), -jnp.sin(pol_angle)], axis=-1),
                    jnp.stack([jnp.sin(pol_angle), jnp.cos(pol_angle)], axis=-1)
                ],
                axis=-2
            )
            spectral_power = R.T @ spectral_power @ R  # [E, 2, 2]
        else:
            spectral_power = yield Prior(
                tfpd.Uniform(
                    low=quantity_to_jnp(self.min_channel_power/self.channel_width, 'Jy*m^2/Hz') * ones,
                    high=quantity_to_jnp(self.max_channel_power/self.channel_width, 'Jy*m^2/Hz') * ones
                ),
                name='spectral_power'
            ).parametrised()

        return ParametricDelayACF(
            mu=mu,
            fwhp=fwhp,
            spectral_power=spectral_power,
            resolution=self.acf_resolution,
            convention=self.convention
        )
