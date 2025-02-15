import jax
from astropy import units as au
from jax import numpy as jnp

from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_common.visibility_model.source_models.rfi.parametric_rfi_emitter import ParametricDelayACF


def test_parametric_delay_acf():
    mu = jnp.asarray([700e6, 700e6, 699e6, 699e6])
    fwhp = jnp.asarray([1e6, 100e3, 1e6, 5e6])
    spectral_power = quantity_to_jnp([10, 10, 10, 10] * au.Jy * (1 * au.km) ** 2 / (130 * au.kHz), 'Jy*km^2/Hz')
    channel_lower = jnp.asarray([700e6])
    channel_upper = jnp.asarray([700e6 + 130e3])
    delay_acf = ParametricDelayACF(mu, fwhp,
                                   spectral_power=spectral_power,
                                   channel_lower=channel_lower,
                                   channel_upper=channel_upper,
                                   convention='physical', resolution=128)
    taus = jnp.linspace(-1e-4, 1e-4, 1000)

    acf_vals = jax.vmap(delay_acf)(taus)
    import pylab as plt

    plt.plot(taus * 1e6, jnp.abs(acf_vals)[:, 0], label='mu=700MHz,sigma=1MHz')
    plt.plot(taus * 1e6, jnp.abs(acf_vals)[:, 1], label='mu=700MHz,sigma=100kHz')
    plt.plot(taus * 1e6, jnp.abs(acf_vals)[:, 2], label='mu=699MHz,sigma=1MHz')
    plt.plot(taus * 1e6, jnp.abs(acf_vals)[:, 3], label='mu=699MHz,sigma=5MHz')
    plt.legend()
    plt.title('Parametric Delay ACF, Channel 700MHz to 700.130MHz')
    plt.xlabel(r'Delay ($\mu$s)')
    plt.ylabel('ACF (Jy km^2)')
    plt.show()
