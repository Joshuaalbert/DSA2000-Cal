import dataclasses
from typing import List

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np
import pylab as plt

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.source_models.gaussian_source_model import GaussianSourceModel
from dsa2000_cal.source_models.point_source_model import PointSourceModel


@dataclasses.dataclass(eq=False)
class WSCleanSourceModel(AbstractSourceModel):
    """
    Predict vis for Gaussian + point sources.
    """
    point_source_model: PointSourceModel
    gaussian_source_model: GaussianSourceModel

    def __post_init__(self):
        # Ensure same frequencies
        if np.any(self.point_source_model.freqs != self.gaussian_source_model.freqs):
            raise ValueError("Point and Gaussian source models must have the same frequencies")

    @staticmethod
    def from_wsclean_model(wsclean_file: str, time: at.Time, phase_tracking: ac.ICRS,
                           freqs: au.Quantity, lmn_transform_params: bool = True, **kwargs) -> 'WSCleanSourceModel':
        """
        Create a GaussianSourceModel from a wsclean model file.

        Args:
            wsclean_file: the wsclean model file
            time: the time of the observation
            phase_tracking: the phase tracking center
            freqs: the frequencies to use
            lmn_transform_params: whether to transform the ellipsoidal parameters to the plane of the sky
            **kwargs:

        Returns:
            WSCleanSourceModel
        """
        return WSCleanSourceModel(
            gaussian_source_model=GaussianSourceModel.from_wsclean_model(
                wsclean_file=wsclean_file,
                time=time,
                phase_tracking=phase_tracking,
                freqs=freqs,
                lmn_transform_params=lmn_transform_params,
                **kwargs
            ),
            point_source_model=PointSourceModel.from_wsclean_model(
                wsclean_file=wsclean_file,
                time=time,
                phase_tracking=phase_tracking,
                freqs=freqs,
                **kwargs
            )
        )

    def predict(self, uvw: au.Quantity) -> jax.Array:
        return self.point_source_model.predict(uvw) + self.gaussian_source_model.predict(uvw)

    def get_flux_model(self, lvec=None, mvec=None):
        lvec_point, mvec_point, flux_model_point = self.point_source_model.get_flux_model(lvec=lvec, mvec=mvec)
        lvec_gaussian, mvec_gaussian, flux_model_gaussian = self.gaussian_source_model.get_flux_model(lvec=lvec,
                                                                                                      mvec=mvec)
        if lvec is None or mvec is None:
            lvec_min = min(lvec_point[0], lvec_gaussian[0])
            lvec_max = max(lvec_point[-1], lvec_gaussian[-1])
            mvec_min = min(mvec_point[0], mvec_gaussian[0])
            mvec_max = max(mvec_point[-1], mvec_gaussian[-1])
            lvec = np.linspace(lvec_min, lvec_max, 100)
            mvec = np.linspace(mvec_min, mvec_max, 100)

            # Get on common grid
            _, _, flux_model_point = self.point_source_model.get_flux_model(lvec=lvec, mvec=mvec)
            _, _, flux_model_gaussian = self.gaussian_source_model.get_flux_model(lvec=lvec, mvec=mvec)
            flux_model = flux_model_point + flux_model_gaussian
            return lvec, mvec, flux_model
        else:
            flux_model = flux_model_point + flux_model_gaussian
            return lvec, mvec, flux_model

    def plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), squeeze=False,
                                sharex=True, sharey=True)

        lvec_point, mvec_point, flux_model_point = self.point_source_model.get_flux_model()
        axs[0][0].imshow(flux_model_point, origin='lower',
                         extent=(lvec_point[0], lvec_point[-1], mvec_point[0], mvec_point[-1]))
        axs[0][0].set_title('Point Source')

        lvec_gaussian, mvec_gaussian, flux_model_gaussian = self.gaussian_source_model.get_flux_model()
        axs[1][0].imshow(flux_model_gaussian, origin='lower',
                         extent=(lvec_gaussian[0], lvec_gaussian[-1], mvec_gaussian[0], mvec_gaussian[-1]))
        axs[1][0].set_title('Gaussian Source')

        # Grid onto common grid

        lvec_min = min(lvec_point[0], lvec_gaussian[0])
        lvec_max = max(lvec_point[-1], lvec_gaussian[-1])
        mvec_min = min(mvec_point[0], mvec_gaussian[0])
        mvec_max = max(mvec_point[-1], mvec_gaussian[-1])
        lvec = np.linspace(lvec_min, lvec_max, 100)
        mvec = np.linspace(mvec_min, mvec_max, 100)

        # Get on common grid
        _, _, flux_model_point = self.point_source_model.get_flux_model(lvec=lvec, mvec=mvec)
        _, _, flux_model_gaussian = self.gaussian_source_model.get_flux_model(lvec=lvec, mvec=mvec)
        flux_model = flux_model_point + flux_model_gaussian

        axs[2][0].imshow(flux_model, origin='lower', extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]))
        axs[2][0].set_title('Combined Source')

        # 0 hspce
        fig.subplots_adjust(hspace=0, wspace=0)

        fig.tight_layout()
        plt.show()

    def __add__(self, other: 'WSCleanSourceModel') -> 'WSCleanSourceModel':
        return WSCleanSourceModel(
            point_source_model=self.point_source_model + other.point_source_model,
            gaussian_source_model=self.gaussian_source_model + other.gaussian_source_model
        )

    def __or__(self, other: 'AbstractSourceModel') -> 'ConcatenatedSourceModel':
        if isinstance(other, ConcatenatedSourceModel):
            return ConcatenatedSourceModel(source_models=[self, *other.source_models])
        return ConcatenatedSourceModel(source_models=[self, other])


@dataclasses.dataclass(eq=False)
class ConcatenatedSourceModel(AbstractSourceModel):
    source_models: List[WSCleanSourceModel]

    def predict(self, uvw: au.Quantity) -> jax.Array:
        vis = self.source_models[0].predict(uvw)
        for source_model in self.source_models[1:]:
            vis += source_model.predict(uvw)
        return vis
