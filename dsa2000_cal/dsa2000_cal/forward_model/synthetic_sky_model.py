import dataclasses
from typing import List

import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
import jax
import numpy as np

from dsa2000_cal.common.astropy_utils import create_spherical_grid
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.source_models.gaussian_stokes_I_source_model import GaussianSourceModelParams, \
    transform_ellipsoidal_params_to_plane_of_sky, GaussianSourceModel
from dsa2000_cal.source_models.point_stokes_I_source_model import PointSourceModelParams, PointSourceModel
from dsa2000_cal.source_models.wsclean_stokes_I_source_model import WSCleanSourceModel


class SkyModel(SerialisableBaseModel):
    bright_sources: List[PointSourceModelParams]
    faint_sources: List[GaussianSourceModelParams]

    def to_wsclean_source_models(self) -> List[WSCleanSourceModel]:
        source_models = []
        for source_params in self.bright_sources:
            source = PointSourceModel.from_point_source_params(source_params)
            source_models.append(
                WSCleanSourceModel(
                    point_source_model=source,
                    gaussian_source_model=None,
                    freqs=source.freqs
                )
            )
        for source_params in self.faint_sources:
            source = GaussianSourceModel.from_gaussian_source_params(source_params)
            source_models.append(
                WSCleanSourceModel(
                    point_source_model=None,
                    gaussian_source_model=source,
                    freqs=source.freqs
                )
            )
        return source_models


@dataclasses.dataclass(eq=False)
class SyntheticSkyModelProducer:
    phase_tracking: ac.ICRS
    obs_time: at.Time
    freqs: au.Quantity
    num_bright_sources: int
    num_faint_sources: int
    field_of_view: au.Quantity
    mean_major: au.Quantity = 1 * au.arcmin
    mean_minor: au.Quantity = 0.5 * au.arcmin

    seed: int = 42

    def __post_init__(self):
        if not self.field_of_view.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected field_of_view to be in degrees, got {self.field_of_view.unit}")
        if not self.phase_tracking.isscalar:
            raise ValueError(f"Expected phase_tracking to be a scalar, got {self.phase_tracking}")
        if not self.obs_time.isscalar:
            raise ValueError(f"Expected obs_time to be a scalar, got {self.obs_time}")

        if self.num_bright_sources < 0:
            raise ValueError(f"Expected num_bright_sources to be greater than equal 0, got {self.num_bright_sources}")
        if self.num_faint_sources < 0:
            raise ValueError(f"Expected num_faint_sources to be greater than equal 0, got {self.num_faint_sources}")

        if not self.mean_minor.unit.is_equivalent(au.arcmin):
            raise ValueError(f"Expected mean_minor to be in arcmin, got {self.mean_minor.unit}")
        if not self.mean_major.unit.is_equivalent(au.arcmin):
            raise ValueError(f"Expected mean_major to be in arcmin, got {self.mean_major.unit}")

        self.key = jax.random.PRNGKey(self.seed)

    def create_sky_model(self) -> SkyModel:
        """
        Create a sky model with bright and faint sources.

        Returns:
            SkyModel: A sky model with bright and faint sources
        """

        self.key, key1, key2 = jax.random.split(self.key, 3)

        bright_source_params = []
        if self.num_bright_sources > 0:
            dr_bright = choose_dr(field_of_view=self.field_of_view, total_n=self.num_bright_sources)
            bright_rotation = float(jax.random.uniform(key1, (), minval=0, maxval=60)) * au.deg
            bright_sources = create_spherical_grid(
                pointing=self.phase_tracking,
                angular_radius=0.5 * self.field_of_view,
                dr=dr_bright,
                sky_rotation=bright_rotation
            )
            for source in bright_sources:
                lmn0 = icrs_to_lmn(sources=source, phase_tracking=self.phase_tracking, time=self.obs_time)
                A = np.ones((1, len(self.freqs))) * au.Jy
                bright_source_params.append(
                    PointSourceModelParams(
                        freqs=self.freqs,
                        l0=lmn0[0],
                        m0=lmn0[1],
                        A=A
                    )
                )
        faint_source_params = []
        if self.num_faint_sources > 0:
            dr_faint = choose_dr(field_of_view=self.field_of_view, total_n=self.num_faint_sources)
            faint_rotation = float(jax.random.uniform(key2, (), minval=0, maxval=60)) * au.deg
            faint_sources = create_spherical_grid(
                pointing=self.phase_tracking,
                angular_radius=0.5 * self.field_of_view,
                dr=dr_faint,
                sky_rotation=faint_rotation
            )
            for source in faint_sources:
                self.key, key_major, key_minor, key_theta = jax.random.split(self.key, 4)
                # Sampling turned off for now...
                # major = float(jax.random.laplace(key_major, ())) * self.mean_major
                # minor = float(jax.random.laplace(key_minor, ())) * self.mean_minor
                major = self.mean_major
                minor = self.mean_minor
                if major < minor:
                    major, minor = minor, major
                theta = float(jax.random.uniform(key_theta, (), minval=0, maxval=180)) * au.deg
                l0, m0, major_tangent, minor_tangent, theta_tangent = transform_ellipsoidal_params_to_plane_of_sky(
                    major=major[None],
                    minor=minor[None],
                    theta=theta[None],
                    source_directions=source[None],
                    phase_tracking=self.phase_tracking,
                    obs_time=self.obs_time,
                    lmn_transform_params=True
                )
                A = np.ones((1, len(self.freqs))) * au.Jy
                faint_source_params.append(
                    GaussianSourceModelParams(
                        freqs=self.freqs,
                        l0=l0,
                        m0=m0,
                        major=major_tangent,
                        minor=minor_tangent,
                        theta=theta_tangent,
                        A=A
                    )
                )
        return SkyModel(
            bright_sources=bright_source_params,
            faint_sources=faint_source_params
        )


def choose_dr(field_of_view: au.Quantity, total_n: int) -> au.Quantity:
    """
    Choose the dr for a given field of view and total number of sources.
    Approximate number of sources result.

    Args:
        field_of_view:
        total_n:

    Returns:
        the sky spacing
    """
    possible_n = [1, 7, 19, 37, 62, 93, 130, 173, 223, 279, 341, 410, 485,
                  566, 653, 747, 847, 953, 1066, 1185]
    if total_n not in possible_n:
        print(f"total_n {total_n} not exactly achievable. Will be rounded to nearest achievable value. "
              f"This is based on a uniform dithering of the sky. Possible values are {possible_n}.")
    if total_n == 1:
        return field_of_view
    dr = 0.5 * field_of_view / np.arange(1, total_n + 1)
    N = np.floor(0.5 * field_of_view / dr)
    num_n = [1 + np.sum(np.floor(2 * np.pi * np.arange(1, n + 1))) for n in N]
    return au.Quantity(np.interp(total_n, num_n, dr))
