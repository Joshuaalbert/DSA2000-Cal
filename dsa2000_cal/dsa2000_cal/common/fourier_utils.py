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
    convention: str = 'fourier'

    def to_image(self, f_aperture, axes, dx):
        if self.convention == 'fourier':
            return self._to_image_fourier(f_aperture, axes, dx)
        elif self.convention == 'casa':
            return self._to_image_casa(f_aperture, axes, dx)
        else:
            raise ValueError(f"Unknown convention {self.convention}")

    def to_aperture(self, f_image, axes, dnu):
        if self.convention == 'fourier':
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
