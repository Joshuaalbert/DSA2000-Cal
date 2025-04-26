import jax
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from dsa2000_fm.array_layout.fast_psf_evaluation import compute_psf

tfpd = tfp.distributions


def test_compute_psf():
    N = 1650
    D = 10000
    antennas_gcrs = tfpd.Normal(0., 10e3).sample((N, 3), seed=jax.random.PRNGKey(0)).at[..., 2].set(0.)
    lmn = tfpd.Normal(0., 0.001).sample((D, 3), seed=jax.random.PRNGKey(1))
    lmn = lmn.at[..., 2].set(jnp.sqrt(1 - lmn[..., 0] ** 2 - lmn[..., 1] ** 2))
    freqs = jnp.linspace(700e6, 2000e6, 2)
    latitude = jnp.pi / 4
    transit_dec = 0.

    psf = compute_psf(antennas=antennas_gcrs, lmn=lmn, freqs=freqs, time=latitude, ra0=0., dec0=transit_dec,
                      with_autocorr=True, accumulate_dtype=jnp.float64)
    plt.scatter(
        lmn[..., 0],
        lmn[..., 1],
        c=10 * jnp.log10(psf),
        s=1,
        cmap='jet',
        vmin=-60,
        vmax=10 * np.log10(0.5)
    )

    plt.show()