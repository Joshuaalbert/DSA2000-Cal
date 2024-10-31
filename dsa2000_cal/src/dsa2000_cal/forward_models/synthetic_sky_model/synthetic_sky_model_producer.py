import dataclasses
from typing import List

import astropy.coordinates as ac
import astropy.units as au
import jax
import numpy as np

from src.dsa2000_cal.assets import fill_registries
from src.dsa2000_cal.assets import source_model_registry, rfi_model_registry
from dsa2000_cal.common.astropy_utils import create_spherical_grid_old, create_random_spherical_layout, choose_dr
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from src.dsa2000_cal.visibility_model.source_models.celestial.fits_source_model import FITSSourceModel
from src.dsa2000_cal.visibility_model.source_models.celestial.gaussian_source_model import \
    GaussianSourceModel
from dsa2000_cal.visibility_model.source_models.celestial.point_source_model import PointSourceModel
from src.dsa2000_cal.visibility_model.source_models.rfi.rfi_emitter_source_model import \
    RFIEmitterSourceModel


@dataclasses.dataclass(eq=False)
class SyntheticSkyModelProducer:
    phase_tracking: ac.ICRS
    freqs: au.Quantity
    field_of_view: au.Quantity

    seed: int = 42

    def __post_init__(self):
        if not self.field_of_view.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected field_of_view to be in degrees, got {self.field_of_view.unit}")
        if not self.phase_tracking.isscalar:
            raise ValueError(f"Expected phase_tracking to be a scalar, got {self.phase_tracking}")

        np.random.seed(self.seed)
        self.keys = jax.random.split(jax.random.PRNGKey(self.seed), 5)

    def create_sources_outside_fov(self, key=None, num_bright_sources: int = 100,
                                   full_stokes: bool = True) -> PointSourceModel:
        """
        Create 10Jy sources outside over sky.

        Args:
            key: the random key
            num_bright_sources: the number of bright sources to create, about 1 per 400 sq deg

        Returns:
            sources: a single facet model with all the bright sources
        """
        if key is None:
            key = self.keys[0]

        # Based on 100 10Jy sources per 400 sq deg
        coords = create_random_spherical_layout(
            num_sources=num_bright_sources,
            key=key
        )
        lmn = icrs_to_lmn(sources=coords, phase_tracking=self.phase_tracking)

        num_sources = len(lmn)
        source_flux = 10 * au.Jy * ((700 * au.MHz) / np.mean(self.freqs))
        if full_stokes:
            A = np.zeros((num_sources, len(self.freqs), 2, 2)) * au.Jy
            A[..., 0, 0] = 0.5 * source_flux
            A[..., 1, 1] = 0.5 * source_flux
        else:
            A = np.ones((num_sources, len(self.freqs))) * source_flux

        return PointSourceModel(
            freqs=self.freqs,
            l0=lmn[:, 0],
            m0=lmn[:, 1],
            A=A
        )

    def create_sources_inside_fov(self, key=None, num_sources: int = 100, full_stokes: bool = True) -> PointSourceModel:
        """
        Create sources inside the field

        Args:
            key: the random key
            num_sources: the number of sources to create

        Returns:
            sources: a single facet model with all the bright sources
        """
        if key is None:
            key = self.keys[1]
        shrink_factor = 0.75  # Shrink the field of view to avoid sources at the edge
        dr_bright = choose_dr(field_of_view=self.field_of_view, total_n=num_sources)
        bright_rotation = float(jax.random.uniform(key, (), minval=0, maxval=60)) * au.deg
        bright_sources = create_spherical_grid_old(
            pointing=self.phase_tracking,
            angular_radius=0.5 * self.field_of_view * shrink_factor,
            dr=dr_bright,
            sky_rotation=bright_rotation
        )
        num_sources = len(bright_sources)
        lmn = icrs_to_lmn(sources=bright_sources, phase_tracking=self.phase_tracking)
        source_flux = 1 * au.Jy * ((700 * au.MHz) / np.mean(self.freqs))
        if full_stokes:
            A = np.zeros((num_sources, len(self.freqs), 2, 2)) * au.Jy
            A[..., 0, 0] = 0.5 * source_flux
            A[..., 1, 1] = 0.5 * source_flux
        else:
            A = np.ones((num_sources, len(self.freqs))) * source_flux
        return PointSourceModel(
            freqs=self.freqs,
            l0=lmn[:, 0],
            m0=lmn[:, 1],
            A=A
        )

    def create_diffuse_sources_inside_fov(self, key=None, mean_major: au.Quantity = 1 * au.arcmin,
                                          mean_minor: au.Quantity = 0.5 * au.arcmin,
                                          num_sources: int = 100, full_stokes: bool = True) -> GaussianSourceModel:
        """
        Create faint sources inside the field.

        Args:
            key: the random key
            mean_major: the mean major axis of the sources
            mean_minor: the mean minor axis of the sources
            num_sources: the number of sources to create

        Returns:
            sources: a single facet model with all the faint sources
        """
        if key is None:
            key = self.keys[2]
        if not mean_major.unit.is_equivalent(au.arcmin):
            raise ValueError(f"Expected mean_major to be in arcmin, got {mean_major.unit}")
        if not mean_minor.unit.is_equivalent(au.arcmin):
            raise ValueError(f"Expected mean_minor to be in arcmin, got {mean_minor.unit}")
        shrink_factor = 0.75  # Shrink the field of view to avoid sources at the edge
        key1, key2 = jax.random.split(key, 2)

        dr_faint = choose_dr(field_of_view=self.field_of_view, total_n=num_sources)
        faint_rotation = float(jax.random.uniform(key1, (), minval=0, maxval=60)) * au.deg
        faint_sources = create_spherical_grid_old(
            pointing=self.phase_tracking,
            angular_radius=0.5 * self.field_of_view * shrink_factor,
            dr=dr_faint,
            sky_rotation=faint_rotation
        )
        num_sources = len(faint_sources)

        lmn = icrs_to_lmn(sources=faint_sources, phase_tracking=self.phase_tracking)
        source_flux = 0.1 * au.Jy * ((700 * au.MHz) / np.mean(self.freqs))
        major = mean_major.to('rad').value * np.ones((num_sources,)) * au.dimensionless_unscaled
        minor = mean_minor.to('rad').value * np.ones((num_sources,)) * au.dimensionless_unscaled
        theta = np.asarray(jax.random.uniform(key2, (num_sources,), minval=0, maxval=180)) * au.deg
        if full_stokes:
            A = np.zeros((num_sources, len(self.freqs), 2, 2)) * au.Jy
            A[..., 0, 0] = 0.5 * source_flux
            A[..., 1, 1] = 0.5 * source_flux
        else:
            A = np.ones((num_sources, len(self.freqs))) * source_flux
        return GaussianSourceModel(
            freqs=self.freqs,
            l0=lmn[:, 0],
            m0=lmn[:, 1],
            major=major,
            minor=minor,
            theta=theta,
            A=A
        )

    def create_a_team_sources(self, key=None, a_team_sources: List[str] | None = None, full_stokes: bool = True) -> \
            List[FITSSourceModel]:
        """
        Create the A-Team sources as FITS models.

        Args:
            key: the random key
            a_team_sources: the A-Team sources to create

        Returns:
            sources: the A-Team sources
        """
        fill_registries()
        if key is None:
            key = self.keys[3]
        if a_team_sources is None:
            a_team_sources = ["cas_a", "cyg_a", "vir_a", "tau_a"]
        source_models = []
        for source in a_team_sources:
            source_model_asset = source_model_registry.get_instance(source_model_registry.get_match(source))
            # To repoint th image
            # new_centre = ac.ICRS(
            #     *offset_by(self.phase_tracking.ra, self.phase_tracking.dec,
            #                posang=np.random.uniform(0., 360.) * au.deg,
            #                distance=np.random.uniform(0.5, 1.) * au.deg)
            # )
            source_model = FITSSourceModel.from_wsclean_model(
                wsclean_fits_files=source_model_asset.get_wsclean_fits_files(),
                phase_tracking=self.phase_tracking, freqs=self.freqs, ignore_out_of_bounds=True,
                full_stokes=full_stokes,
                # repoint_centre=new_centre
            )
            source_models.append(source_model)
        return source_models

    def create_rfi_emitter_sources(self, key=None, rfi_sources: List[str] | None = None, full_stokes: bool = True) -> [
        RFIEmitterSourceModel]:
        """
        Create RFI emitter sources inside the field.

        Args:
            key: the random key
            rfi_sources: the LTE sources

        Returns:
            sources: a single facet model with all the RFI emitter sources
        """
        if key is None:
            key = self.keys[4]
        if rfi_sources is None:
            rfi_sources = ["lte_cell_tower"]

        fill_registries()
        source_models = []
        for rfi_source in rfi_sources:
            rfi_model = rfi_model_registry.get_instance(rfi_model_registry.get_match(rfi_source))
            rfi_model_params = rfi_model.make_source_params(freqs=self.freqs, full_stokes=full_stokes)
            source_model = RFIEmitterSourceModel(rfi_model_params)
            source_models.append(source_model)
        return source_models


