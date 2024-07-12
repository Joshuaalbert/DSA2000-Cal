import dataclasses

import numpy as np
from jax import numpy as jnp


@dataclasses.dataclass(eq=False)
class ApertureTransform:
    """
    A class to transform between aperture and image planes.

    For fourier convention, the transform is defined as:

    .. math::

            f_image = int f_aperture(x) e^{2i pi x nu} dx
            f_aperture = int f_image(nu) e^{-2i pi x nu} dnu

    For casa convention, the transform is defined as:

    .. math::

            f_image = int f_aperture(x) e^{-2i pi x nu} dx
            f_aperture = int f_image(nu) e^{2i pi x nu} dnu

    """
    convention: str = 'physical'

    def to_image(self, f_aperture, axes, dx):
        if self.convention == 'physical':
            return self._to_image_fourier(f_aperture, axes, dx)
        elif self.convention == 'casa':
            return self._to_image_casa(f_aperture, axes, dx)
        else:
            raise ValueError(f"Unknown convention {self.convention}")

    def to_aperture(self, f_image, axes, dnu):
        if self.convention == 'physical':
            return self._to_aperture_fourier(f_image, axes, dnu)
        elif self.convention == 'casa':
            return self._to_aperture_casa(f_image, axes, dnu)
        else:
            raise ValueError(f"Unknown convention {self.convention}")

    def _to_aperture_fourier(self, f_image, axes, dnu):
        # undo uses -2pi convention so fft is used
        return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(f_image, axes=axes), axes=axes), axes=axes) * dnu

    def _to_image_fourier(self, f_aperture, axes, dx):
        factor = np.prod([f_aperture.shape[axis] for axis in axes])
        # uses -2pi convention so ifft is used
        return jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(f_aperture, axes=axes), axes=axes),
                                axes=axes) * dx * factor

    def _to_aperture_casa(self, f_image, axes, dnu):
        # uses +2pi convention so ifft is used
        factor = np.prod([f_image.shape[axis] for axis in axes])
        return jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(f_image, axes=axes), axes=axes),
                                axes=axes) * dnu * factor

    def _to_image_casa(self, f_aperture, axes, dx):
        # uses +2pi convention so ifft is used
        return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(f_aperture, axes=axes), axes=axes), axes=axes) * dx


def _find_optimal_fft_size(N, required_radix=[2]):
    # Allowed radix bases in the order of efficiency
    allowed_radix = [4, 2, 3, 5, 7]

    def is_optimal_size(x):
        """ Check if x can be entirely factorized using the allowed radix bases """
        for radix in allowed_radix:
            while x % radix == 0:
                x //= radix
        return x == 1

    def includes_required_radix(x):
        """ Check if x includes all required radix factors at least once """
        for radix in required_radix:
            temp = x
            if temp % radix != 0:
                return False
        return True

    # Ensure N includes all required radix factors
    for radix in required_radix:
        while N % radix != 0:
            N *= radix

    # Start checking from N, increment until an optimal size is found
    while not (is_optimal_size(N) and includes_required_radix(N)):
        N += 1

    return N


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def find_optimal_fft_size(N, required_radix=None,
                          allowed_radix=None) -> int:
    """
    Find the optimal size for FFT given the input size N and required radix factors.

    Args:
        N: the input size
        required_radix: the required radix factors
        allowed_radix: the allowed radix factors

    Returns:
        int: the optimal size for FFT
    """
    if required_radix is None:
        required_radix = {4}  # Most efficienct radix
    if allowed_radix is None:
        allowed_radix = {4, 2, 3, 5, 7}

    if not required_radix.issubset(allowed_radix):
        raise ValueError("required_radix must be a subset of allowed_radix")

    def satisfy(x):
        factors = prime_factors(x)
        # Since 4 is good, replace 2 2's with 4
        if factors.count(2) == 2:
            factors.remove(2)
            factors.remove(2)
            factors.append(4)
        factors = set(factors)
        # only contains allows, and must contain required
        return factors.issubset(allowed_radix) and required_radix.issubset(factors)

    while not satisfy(N):
        N += 1
    return N
