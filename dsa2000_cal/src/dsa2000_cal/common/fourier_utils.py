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
    use_shifts: bool = True

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

        if self.convention == 'physical':
            if axes != (-2, -1):
                f_aperture, move_back = move_axes_to_end(f_aperture, axes)
                result = self._to_image_physical_with_shifts(f_aperture, axes, dx, dy)
                return move_back(result)
            return self._to_image_physical_with_shifts(f_aperture, axes, dx, dy)
        elif self.convention == 'engineering':
            if axes != (-2, -1):
                f_aperture, move_back = move_axes_to_end(f_aperture, axes)
                result = self._to_image_engineering(f_aperture, axes, dx, dy)
                return move_back(result)
            return self._to_image_engineering(f_aperture, axes, dx, dy)
        else:
            raise ValueError(f"Unknown convention {self.convention}")

    def to_aperture(self, f_image, axes, dl, dm):
        if self.convention == 'physical':
            if axes != (-2, -1):
                f_image, move_back = move_axes_to_end(f_image, axes)
                result = self._to_aperture_physical_with_shifts(f_image, axes, dl, dm)
                return move_back(result)
            return self._to_aperture_physical_with_shifts(f_image, axes, dl, dm)
        elif self.convention == 'engineering':
            if axes != (-2, -1):
                f_image, move_back = move_axes_to_end(f_image, axes)
                result = self._to_aperture_engineering(f_image, axes, dl, dm)
                return move_back(result)
            return self._to_aperture_engineering(f_image, axes, dl, dm)
        else:
            raise ValueError(f"Unknown convention {self.convention}")

    def _to_aperture_physical(self, f_image, axes, dl, dm):
        # No shift needed if we compute in this form.
        l_axis, m_axis = axes
        num_l = np.shape(f_image)[l_axis]
        num_m = np.shape(f_image)[m_axis]

        # uses -2pi convention, f[m] = f(lmin + m dl), dl * dx = 1/N
        # F(x[n]) = int f(l) e^{-2 pi i l x[n]} dl = sum_m f[m] e^{-2 pi i l[m] x[n]} dl
        # = sum_m f[m] e^{-2 pi i (lmin + m dl) (xmin + n dx)} dl
        # = sum_m f[m] e^{-2 pi i (lmin * xmin + lmin * n dx + m dl * xmin + m dl * n dx)} dl
        # = e^{-2 pi i (lmin * xmin + lmin * n dx)} sum_m f[m] e^{-2 pi i (m dl * xmin)} e^{-2 pi i (m * n)/N} dl
        # = e^{-2 pi i (lmin * xmin + lmin * n dx)} FFT{f[m] e^{-2 pi i (m dl * xmin)} dl}

        # e^{-2 pi i (m dl * xmin)}
        dx = 1 / (num_l * dl)
        dy = 1 / (num_m * dm)
        xmin = -0.5 * num_l * dx
        ymin = -0.5 * num_m * dy
        Ml, Mm = jnp.meshgrid(jnp.arange(num_l), jnp.arange(num_m), indexing='ij')
        pre_phase_factor = jnp.exp(-2j * jnp.pi * (Ml * dl * xmin + Mm * dm * ymin))

        f_image_shifted = pre_phase_factor * f_image
        f_image_scaled = f_image_shifted * dl * dm

        f_aperture = jnp.fft.fftshift(jnp.fft.fft2(f_image_scaled, axes=axes), axes=axes)

        # e^{-2 pi i (lmin * (xmin + * n dx))}
        Nx, Ny = Ml, Mm
        lmin = -0.5 * num_l * dl
        mmin = -0.5 * num_m * dm
        post_phase_factor = jnp.exp(-2j * jnp.pi * (lmin * (xmin + Nx * dx) + mmin * (ymin + Ny * dy)))
        f_aperture_shifted = post_phase_factor * f_aperture

        return f_aperture_shifted

    def _to_aperture_physical_with_shifts(self, f_image, axes, dl, dm):
        if not self.use_shifts:
            return self._to_aperture_physical(f_image, axes, dl, dm)
        # uses -2pi convention, do shifts with axis rolling
        f_image_scaled = f_image * dl * dm
        f_aperture = jnp.fft.fft2(f_image_scaled, axes=axes)
        f_aperture_shifted = jnp.fft.fftshift(f_aperture, axes=axes)
        return f_aperture_shifted

    def _to_image_physical(self, f_aperture, axes, dx, dy):
        # No shift needed if we compute in this form.
        x_axis, y_axis = axes
        num_x = np.shape(f_aperture)[x_axis]
        num_y = np.shape(f_aperture)[y_axis]

        # uses 2pi convention, F[n] = F(xmin + n dx), dx * dl = 1/N
        # f(l[m]) = int F(x[n]) e^{2 pi i x[n] l[m]} dx = sum_n F[n] e^{2 pi i x[n] l[m]} dx
        # = sum_n F[n] e^{2 pi i (xmin + n dx) (lmin + m dl)} dx
        # = sum_n F[n] e^{2 pi i (xmin lmin + xmin m dl + n dx lmin + n dx m dl)} dx
        # = e^{2 pi i (xmin lmin + xmin m dl)} sum_n F[n] e^{2 pi i (n dx lmin)} e^{2 pi i (n m)/N} dx
        # = e^{2 pi i (xmin lmin + xmin m dl)} (N * IFFT{F[n] e^{2 pi i (n dx lmin)} dx})

        # e^{2 pi i (n dx lmin)}
        dl = 1 / (num_x * dx)
        dm = 1 / (num_y * dy)
        lmin = -0.5 * num_x * dl
        mmin = -0.5 * num_y * dm
        Nx, Ny = jnp.meshgrid(jnp.arange(num_x), jnp.arange(num_y), indexing='ij')
        pre_phase_factor = jnp.exp(2j * jnp.pi * (Nx * dx * lmin + Ny * dy * mmin))

        f_aperture_shifted = pre_phase_factor * f_aperture

        f_aperture_scaled = f_aperture_shifted * dx * dy * num_x * num_y

        f_image = jnp.fft.ifft2(jnp.fft.ifftshift(f_aperture_scaled, axes=axes), axes=axes)

        # e^{2 pi i (xmin (lmin + m dl))}

        Ml, Mm = Nx, Ny
        xmin = -0.5 * num_x * dx
        ymin = -0.5 * num_y * dy
        post_phase_factor = jnp.exp(2j * jnp.pi * (xmin * (lmin + Ml * dl) + ymin * (mmin + Mm * dm)))
        f_image_shifted = post_phase_factor * f_image

        return f_image_shifted

    def _to_image_physical_with_shifts(self, f_aperture, axes, dx, dy):
        if not self.use_shifts:
            return self._to_image_physical(f_aperture, axes, dx, dy)
        x_axis, y_axis = axes
        num_x = np.shape(f_aperture)[x_axis]
        num_y = np.shape(f_aperture)[y_axis]
        # uses 2pi convention, do shifts with axis rolling
        f_aperture_shifted = jnp.fft.ifftshift(f_aperture, axes=axes)
        f_aperture_scaled = f_aperture_shifted * dx * dy * num_x * num_y
        f_image = jnp.fft.ifft2(f_aperture_scaled, axes=axes)
        f_image_shifted = f_image
        return f_image_shifted

    def _to_aperture_engineering(self, f_image, axes, dl, dm):
        # uses +2pi convention so ifft is used
        return self._to_aperture_physical_with_shifts(f_image.conj(), axes, dl, dm).conj()

    def _to_image_engineering(self, f_aperture, axes, dx, dy):
        # uses +2pi convention so ifft is used
        return self._to_image_physical_with_shifts(f_aperture.conj(), axes, dx, dy).conj()


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
