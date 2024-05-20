import dataclasses

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import numpy as np
import pylab as plt

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.source_models.gaussian_stokes_I_source_model import GaussianSourceModel
from dsa2000_cal.source_models.point_stokes_I_source_model import PointSourceModel


@dataclasses.dataclass(eq=False)
class WSCleanSourceModel(AbstractSourceModel):
    """
    Predict vis for Gaussian + point sources.
    """
    point_source_model: PointSourceModel | None
    gaussian_source_model: GaussianSourceModel | None
    freqs: au.Quantity

    def __post_init__(self):
        if self.point_source_model is None and self.gaussian_source_model is None:
            raise ValueError("At least one of point_source_model or gaussian_source_model must be provided")
        # Ensure same frequencies
        if self.point_source_model is not None and self.gaussian_source_model is not None:
            if np.any(self.point_source_model.freqs != self.gaussian_source_model.freqs):
                raise ValueError("Point and Gaussian source models must have the same frequencies")
        if self.point_source_model is not None and np.any(self.freqs != self.point_source_model.freqs):
            raise ValueError("Frequencies must match point source frequencies")
        if self.gaussian_source_model is not None and np.any(self.freqs != self.gaussian_source_model.freqs):
            raise ValueError("Frequencies must match Gaussian source frequencies")

    def flux_weighted_lmn(self) -> au.Quantity:
        denom = 0.
        l_avg = 0.
        m_avg = 0.
        if self.point_source_model is not None:
            A_avg_points = np.mean(self.point_source_model.A, axis=1)  # [num_sources]
            denom += np.sum(A_avg_points)
            l_avg += np.sum(A_avg_points * self.point_source_model.l0)
            m_avg += np.sum(A_avg_points * self.point_source_model.m0)

        if self.gaussian_source_model is not None:
            A_avg_gaussians = np.mean(self.gaussian_source_model.A, axis=1)  # [num_sources]
            denom += np.sum(A_avg_gaussians)
            l_avg += np.sum(A_avg_gaussians * self.gaussian_source_model.l0)
            m_avg += np.sum(A_avg_gaussians * self.gaussian_source_model.m0)

        l_avg /= denom
        m_avg /= denom
        lmn = np.asarray([l_avg, m_avg, np.sqrt(1 - l_avg ** 2 - m_avg ** 2)]) * au.dimensionless_unscaled
        return lmn

    @staticmethod
    def from_wsclean_model(wsclean_clean_component_file: str,
                           time: at.Time, phase_tracking: ac.ICRS,
                           freqs: au.Quantity, lmn_transform_params: bool = True, **kwargs) -> 'WSCleanSourceModel':
        """
        Create a GaussianSourceModel from a wsclean model file.

        Args:
            wsclean_clean_component_file: the wsclean model file
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
                wsclean_clean_component_file=wsclean_clean_component_file,
                time=time,
                phase_tracking=phase_tracking,
                freqs=freqs,
                lmn_transform_params=lmn_transform_params,
                **kwargs
            ),
            point_source_model=PointSourceModel.from_wsclean_model(
                wsclean_clean_component_file=wsclean_clean_component_file,
                time=time,
                phase_tracking=phase_tracking,
                freqs=freqs,
                **kwargs
            ),
            freqs=freqs
        )

    def get_flux_model(self, lvec=None, mvec=None):
        lvec_point, mvec_point, flux_model_point = self.point_source_model.get_flux_model(lvec=lvec, mvec=mvec)
        lvec_gaussian, mvec_gaussian, flux_model_gaussian = self.gaussian_source_model.get_flux_model(lvec=lvec,
                                                                                                      mvec=mvec)
        if lvec is None or mvec is None:
            lvec_min = min(lvec_point[0], lvec_gaussian[0])
            lvec_max = max(lvec_point[-1], lvec_gaussian[-1])
            mvec_min = min(mvec_point[0], mvec_gaussian[0])
            mvec_max = max(mvec_point[-1], mvec_gaussian[-1])
            lvec = np.linspace(lvec_min, lvec_max, 256)
            mvec = np.linspace(mvec_min, mvec_max, 256)

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

        lvec_min = mvec_min = np.inf
        lvec_max = mvec_max = -np.inf

        if self.point_source_model is not None:
            lvec_point, mvec_point, flux_model_point = self.point_source_model.get_flux_model()
            im = axs[0][0].imshow(
                flux_model_point,
                origin='lower',
                extent=(lvec_point[0], lvec_point[-1], mvec_point[0], mvec_point[-1]),
                interpolation='none'
            )
            axs[0][0].set_title('Point Source')
            # colorbar
            plt.colorbar(im, ax=axs[0][0])
            lvec_min = min(lvec_point[0], lvec_min)
            lvec_max = max(lvec_point[-1], lvec_max)
            mvec_min = min(mvec_point[0], mvec_min)
            mvec_max = max(mvec_point[-1], mvec_max)

        if self.gaussian_source_model is not None:
            lvec_gaussian, mvec_gaussian, flux_model_gaussian = self.gaussian_source_model.get_flux_model()
            im = axs[1][0].imshow(
                flux_model_gaussian,
                origin='lower',
                extent=(lvec_gaussian[0], lvec_gaussian[-1], mvec_gaussian[0], mvec_gaussian[-1]),
                interpolation='none'
            )
            axs[1][0].set_title('Gaussian Source')
            plt.colorbar(im, ax=axs[1][0])
            lvec_min = min(lvec_gaussian[0], lvec_min)
            lvec_max = max(lvec_gaussian[-1], lvec_max)
            mvec_min = min(mvec_gaussian[0], mvec_min)
            mvec_max = max(mvec_gaussian[-1], mvec_max)

        # Grid onto common grid
        lvec = np.linspace(lvec_min, lvec_max, 512)
        mvec = np.linspace(mvec_min, mvec_max, 512)

        # Get on common grid
        flux_model = 0.
        if self.point_source_model is not None:
            _, _, flux_model_point = self.point_source_model.get_flux_model(lvec=lvec, mvec=mvec)
            flux_model = flux_model + flux_model_point
        if self.gaussian_source_model is not None:
            _, _, flux_model_gaussian = self.gaussian_source_model.get_flux_model(lvec=lvec, mvec=mvec)
            flux_model = flux_model + flux_model_gaussian

        im = axs[2][0].imshow(
            flux_model,
            origin='lower',
            extent=(lvec[0], lvec[-1], mvec[0], mvec[-1]),
            interpolation='none'
        )
        axs[2][0].set_title('Combined Source(s)')
        plt.colorbar(im, ax=axs[2][0])
        axs[0][0].set_ylabel('m')
        axs[1][0].set_ylabel('m')
        axs[2][0].set_ylabel('m')
        axs[2][0].set_xlabel('l')
        # 0 hspce
        fig.subplots_adjust(hspace=0, wspace=0)

        fig.tight_layout()
        fig.savefig('synthetic_source_model.png', dpi=300)
        plt.show()
