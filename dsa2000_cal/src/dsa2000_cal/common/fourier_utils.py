import dataclasses
from typing import Tuple, Callable

import jax
import numpy as np
from jax import numpy as jnp


def move_axes_to_end(arr: jax.Array, axes: Tuple[int, int]) -> Tuple[jax.Array, Callable[[jax.Array], jax.Array]]:
    # Move axes to end so that axes=(-2,-1) can be used for fft.
    arr_end = jnp.moveaxis(arr, axes, (-2, -1))

    def move_axes_back(arr_end: jax.Array):
        # Move both axes to original spot
        return jnp.moveaxis(arr_end, (-2, -1), axes)

    return arr_end, move_axes_back


@dataclasses.dataclass(eq=False)
class ApertureTransform:
    """
    A class to transform between aperture and image planes.

    For physical convention, the transform is defined as:

    .. math::

            f_image = int f_aperture(x) e^{2i pi x l} dx
            f_aperture = int f_image(l) e^{-2i pi x l} dl

    For engineering convention, the transform is defined as:

    .. math::

            f_image = int f_aperture(x) e^{-2i pi x l} dm
            f_aperture = int f_image(l) e^{2i pi x l} dl

    Where x is in dimensionless units (e.g. aperture position divided by wavelength),
    and l is direction cosine of wave vector.
    """
    convention: str = 'physical'

    def to_image(self, f_aperture, axes, dx, dy):
        """
        Transform from aperture to image plane.

        Args:
            f_aperture: [..., num_l, num_m, ...]
            axes: the axes of the l and m dimensions
            dx: the pixel size in the l direction = 1 / (num_l * dl)
            dy: the pixel size in the m direction = 1 / (num_m * dm)

        Returns:
            f_image: [..., num_x, num_y, ...]
        """
        transform = self._to_image_physical_with_shifts
        if self.convention == 'engineering':
            transform = lambda f_aperture, axes, dx, dy, _transform=transform: _transform(f_aperture.conj(), axes, dx, dy).conj()
        if axes != (-2, -1):
            f_aperture, move_back = move_axes_to_end(f_aperture, axes)
            result = transform(f_aperture, axes, dx, dy)
            return move_back(result)
        return transform(f_aperture, axes, dx, dy)

    def to_aperture(self, f_image, axes, dl, dm):
        transform = self._to_aperture_physical_with_shifts
        if self.convention == 'engineering':
            transform = lambda f_image, axes, dl, dm, _transform=transform: _transform(f_image.conj(), axes, dl,
                                                                                       dm).conj()
        if axes != (-2, -1):
            f_image, move_back = move_axes_to_end(f_image, axes)
            result = transform(f_image, axes, dl, dm)
            return move_back(result)
        return transform(f_image, axes, dl, dm)

    def _to_aperture_physical_with_shifts(self, f_image, axes, dl, dm):
        # uses -2pi convention, do shifts with axis rolling for efficiency
        f_image_scaled = f_image * dl * dm
        f_aperture = jnp.fft.fft2(jnp.fft.ifftshift(f_image_scaled, axes=axes), axes=axes)
        f_aperture_shifted = jnp.fft.fftshift(f_aperture, axes=axes)
        return f_aperture_shifted

    def _to_image_physical_with_shifts(self, f_aperture, axes, dx, dy):
        x_axis, y_axis = axes
        num_x = np.shape(f_aperture)[x_axis]
        num_y = np.shape(f_aperture)[y_axis]
        # uses 2pi convention, do shifts with axis rolling for efficiency
        f_aperture_shifted = jnp.fft.ifftshift(f_aperture, axes=axes)
        f_aperture_scaled = f_aperture_shifted * dx * dy * num_x * num_y
        f_image = jnp.fft.ifft2(f_aperture_scaled, axes=axes)
        f_image_shifted = jnp.fft.fftshift(f_image, axes=axes)
        return f_image_shifted


def _find_optimal_fft_size(N, required_radix=None):
    # Allowed radix bases in the order of efficiency
    if required_radix is None:
        required_radix = [2]
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
