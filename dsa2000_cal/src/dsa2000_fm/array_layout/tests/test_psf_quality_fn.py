import astropy.units as au
import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from dsa2000_fm.array_layout.psf_quality_fn import create_target, sparse_annulus, dense_annulus
import pylab as plt
tfpd = tfp.distributions


def test_create_target():
    D = 10000
    M = 2
    lmn = tfpd.Normal(0., 0.001).sample((D, 3), seed=jax.random.PRNGKey(1))
    lmn = lmn.at[..., 2].set(jnp.sqrt(1 - lmn[..., 0] ** 2 - lmn[..., 1] ** 2))
    freqs = np.linspace(700e6, 2000e6, 2) * au.Hz
    transit_decs = np.linspace(0, np.pi / 4, M) * au.rad
    psf_dB_mean, psf_dB_std = create_target(key=jax.random.PRNGKey(0), target_array_name='dsa2000W', lmn=lmn,
                                            freqs=freqs, ra0=None, dec0s=transit_decs, num_samples=10,
                                            accumulate_dtype=jnp.float32, num_antennas=None)

    # Plot the results
    plt.scatter(
        lmn[..., 0],
        lmn[..., 1],
        c=psf_dB_mean[0,...],
        s=1,
        cmap='jet',
        vmin=-60,
        vmax=10 * np.log10(0.5)
    )
    plt.colorbar(label='PSF Mean (dB)')
    plt.title('PSF Mean')
    plt.xlabel('l')
    plt.ylabel('m')
    plt.show()

    # Plot the results
    plt.scatter(
        lmn[..., 0],
        lmn[..., 1],
        c=psf_dB_std[0, ...],
        s=1,
        cmap='jet',
        vmin=0,
        vmax=10
    )
    plt.colorbar(label='PSF std (dB)')
    plt.title('PSF std')
    plt.xlabel('l')
    plt.ylabel('m')
    plt.show()


def test_sample_annulus():
    num_samples = 1000
    inner_radius = 0.5
    outer_radius = 1.0
    key = jax.random.PRNGKey(0)
    dtype = jnp.float32
    samples = sparse_annulus(key, inner_radius, outer_radius, dtype, num_samples)
    import pylab as plt
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.show()
    assert np.all(samples[:, 0] ** 2 + samples[:, 1] ** 2 >= inner_radius ** 2)
    assert np.all(samples[:, 0] ** 2 + samples[:, 1] ** 2 <= outer_radius ** 2)


def test_dense_annulus():
    inner_radius = 0.5
    outer_radius = 1.0
    dl=0.01
    frac = 1.
    dtype = jnp.float32
    samples = dense_annulus(inner_radius, outer_radius, dl, frac, dtype)
    print(samples)
    import pylab as plt
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.show()
    assert np.all(samples[:, 0] ** 2 + samples[:, 1] ** 2 >= inner_radius ** 2)
    assert np.all(samples[:, 0] ** 2 + samples[:, 1] ** 2 <= outer_radius ** 2)