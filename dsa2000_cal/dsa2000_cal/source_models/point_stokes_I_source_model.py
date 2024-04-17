import dataclasses

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax.numpy as jnp
import numpy as np
import pylab as plt
from astropy import constants
from jax._src.typing import SupportsDType

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.source_models.wsclean_util import parse_and_process_wsclean_source_line


@dataclasses.dataclass(eq=False)
class PointSourceModel(AbstractSourceModel):
    """
    Predict vis for point source.
    """
    freqs: au.Quantity  # [num_freqs] Frequencies
    l0: au.Quantity  # [num_sources] l coordinate of the source
    m0: au.Quantity  # [num_sources] m coordinate of the source
    A: au.Quantity  # [num_sources, num_freqs] Flex amplitude of the source

    dtype: SupportsDType = jnp.complex64

    def __post_init__(self):
        if not self.freqs.unit.is_equivalent(au.Hz):
            raise ValueError(f"Expected freqs to be in Hz, got {self.freqs.unit}")
        if not self.l0.unit.is_equivalent(au.dimensionless_unscaled):
            raise ValueError(f"Expected l0 to be dimensionless, got {self.l0.unit}")
        if not self.m0.unit.is_equivalent(au.dimensionless_unscaled):
            raise ValueError(f"Expected m0 to be dimensionless, got {self.m0.unit}")
        if not self.A.unit.is_equivalent(au.Jy):
            raise ValueError(f"Expected A to be in Jy, got {self.A.unit}")

        # Ensure all are 1D vectors
        if self.freqs.isscalar:
            self.freqs = self.freqs.reshape((-1,))
        if self.l0.isscalar:
            self.l0 = self.l0.reshape((-1,))
        if self.m0.isscalar:
            self.m0 = self.m0.reshape((-1,))
        if self.A.isscalar:
            self.A = self.A.reshape((-1, 1))

        self.num_sources = self.l0.shape[0]
        self.num_freqs = self.freqs.shape[0]

        if self.A.shape != (self.num_sources, self.num_freqs):
            raise ValueError(f"A must have shape ({self.num_sources},{self.num_freqs}) got {self.A.shape}")

        if not all([x.shape == (self.num_sources,) for x in [self.l0, self.m0]]):
            raise ValueError("All inputs must have the same shape")

        self.n0 = np.sqrt(1 - self.l0 ** 2 - self.m0 ** 2)  # [num_sources]
        self.wavelengths = constants.c / self.freqs  # [num_freqs]

    def flux_weighted_lmn(self) -> au.Quantity:
        A_avg = np.mean(self.A, axis=1)  # [num_sources]
        m_avg = np.sum(self.m0 * A_avg) / np.sum(A_avg)
        l_avg = np.sum(self.l0 * A_avg) / np.sum(A_avg)
        lmn = np.stack([l_avg, m_avg, np.sqrt(1 - l_avg ** 2 - m_avg ** 2)]) * au.dimensionless_unscaled
        return lmn

    @staticmethod
    def from_wsclean_model(wsclean_clean_component_file: str, time: at.Time, phase_tracking: ac.ICRS,
                           freqs: au.Quantity, **kwargs) -> 'PointSourceModel':
        """
        Create a GaussianSourceModel from a wsclean model file.

        Args:
            wsclean_clean_component_file: the wsclean model file
            time: the time of the observation
            phase_tracking: the phase tracking center
            freqs: the frequencies to use
            **kwargs:

        Returns:

        """
        # Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency='125584411.621094', MajorAxis, MinorAxis, Orientation
        # Example: s0c0,POINT,08:28:05.152,39.35.08.511,0.000748810650400475,[-0.00695379313004673,-0.0849693907803257],false,125584411.621094,,,
        # RA and dec are the central coordinates of the component, in notation of "hh:mm:ss.sss" and "dd.mm.ss.sss".
        # The MajorAxis, MinorAxis and Orientation columns define the shape of the Gaussian.
        # The axes are given in units of arcseconds, and orientation is in degrees.

        source_directions = []
        spectrum = []
        with open(wsclean_clean_component_file, 'r') as fp:
            for line in fp:
                line = line.strip()
                if line == '':
                    continue
                if line.startswith('#'):
                    continue
                if line.startswith('Format'):
                    continue
                parsed_results = parse_and_process_wsclean_source_line(line, freqs)
                if parsed_results is None:
                    continue
                if parsed_results.type_ != 'POINT':
                    continue
                source_directions.append(parsed_results.direction)
                spectrum.append(parsed_results.spectrum)

        source_directions = ac.concatenate(source_directions).transform_to(ac.ICRS)
        lmn0 = icrs_to_lmn(source_directions, time, phase_tracking)
        l0 = lmn0[:, 0]
        m0 = lmn0[:, 1]
        A = jnp.stack(spectrum, axis=0) * au.Jy

        return PointSourceModel(
            freqs=freqs,
            l0=l0,
            m0=m0,
            A=A,
            **kwargs
        )

    def get_flux_model(self, lvec=None, mvec=None):
        # Use imshow to plot the sky model evaluated over a LM grid

        if lvec is None or mvec is None:
            # Use imshow to plot the sky model evaluated over a LM grid
            l_min = np.min(self.l0) - 0.01
            m_min = np.min(self.m0) - 0.01
            l_max = np.max(self.l0) + 0.01
            m_max = np.max(self.m0) + 0.01
            lvec = np.linspace(l_min.value, l_max.value, 100)
            mvec = np.linspace(m_min.value, m_max.value, 100)

        # Evaluate over LM
        flux_model = np.zeros((mvec.size, lvec.size)) * au.Jy

        dl = lvec[1] - lvec[0]
        dm = mvec[1] - mvec[0]

        for i in range(self.num_sources):
            l_idx = int((self.l0[i] - lvec[0]) / dl)
            m_idx = int((self.m0[i] - mvec[0]) / dm)
            if l_idx >= 0 and l_idx < lvec.size and m_idx >= 0 and m_idx < mvec.size:
                flux_model[m_idx, l_idx] += self.A[i, 0]
        return lvec, mvec, flux_model

    def plot(self):
        lvec, mvec, flux_model = self.get_flux_model()
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

        im = axs.imshow(flux_model, origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]))
        # colorbar
        plt.colorbar(im, ax=axs)
        axs.set_xlabel('l')
        axs.set_ylabel('m')
        plt.show()

    def __add__(self, other: 'PointSourceModel') -> 'PointSourceModel':
        if not np.all(self.freqs == other.freqs):
            raise ValueError("Frequencies must match")
        return PointSourceModel(
            freqs=self.freqs,
            l0=au.Quantity(np.concatenate([self.l0, other.l0])),
            m0=au.Quantity(np.concatenate([self.m0, other.m0])),
            A=au.Quantity(np.concatenate([self.A, other.A], axis=0)),
            dtype=self.dtype
        )
