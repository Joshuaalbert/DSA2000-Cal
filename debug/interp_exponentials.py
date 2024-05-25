import jax.numpy as jnp
import numpy as np


def main():
    def exponential(x, nu):
        return np.exp(-2j * jnp.pi * (x * nu / constants.c))

    def approximation(x, nu, nu0, dnu):
        alpha = (nu - nu0) / dnu
        return (1 - alpha) * exponential(x, nu0) * alpha * exponential(x, nu0 + dnu)

    def error(x, nu, nu0, dnu):
        return np.abs(exponential(x, nu) - approximation(x, nu, nu0, dnu))

    import astropy.units as au
    import astropy.constants as constants

    x = 3 * au.km
    nu0 = 50e6 * au.Hz
    dnu = 24e3 * au.Hz
    nu = jnp.linspace(nu0, nu0 + dnu, 100) * au.Hz

    errors = error(x, nu, nu0, dnu)
    import pylab as plt

    plt.plot(nu, errors)
    plt.show()


if __name__ == '__main__':
    main()
