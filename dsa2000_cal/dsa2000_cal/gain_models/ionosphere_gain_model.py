import dataclasses
import logging
import os
from datetime import timedelta
from functools import partial
from typing import Tuple

from astropy.wcs import WCS
from jax.config import config

from dsa2000_cal.assets.content_registry import fill_registries
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.astropy_utils import mean_itrs, create_spherical_grid, create_spherical_earth_grid, \
    plot_icrs_points

config.update("jax_enable_x64", True)
# Set num jax devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
from astropy import units as au, coordinates as ac, time as at
from jax import lax
from jax._src.typing import SupportsDType
from tomographic_kernel.models.cannonical_models import SPECIFICATION, build_ionosphere_tomographic_kernel
from tomographic_kernel.tomographic_kernel import GeodesicTuple
from tomographic_kernel.utils import make_coord_array

from dsa2000_cal.common.coord_utils import earth_location_to_enu, icrs_to_enu
from dsa2000_cal.common.interp_utils import get_interp_indices_and_weights, batched_convolved_interp
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.src.common.jax_utils import chunked_pmap, pad_to_chunksize

logger = logging.getLogger(__name__)


@dataclasses.dataclass(eq=False)
class IonosphereGainModel(GainModel):
    """
    Uses nearest neighbour interpolation to compute the gain model.
    """
    freqs: au.Quantity  # [num_freqs]
    antennas: ac.EarthLocation  # [num_ant]

    # Simulation parameters
    array_location: ac.EarthLocation
    phase_tracking: ac.ICRS
    model_directions: ac.ICRS  # [num_model_dir]
    model_times: at.Time  # [num_model_time]
    model_antennas: ac.EarthLocation  # [num_model_ant]

    specification: SPECIFICATION
    plot_folder: str

    ref_ant: ac.EarthLocation | None = None
    ref_time: at.Time | None = None

    compute_tec: bool = True  # Faster to compute TEC only and differentiate later
    S_marg: int = 25
    jitter: float = 0.05  # Adds 0.05 mTECU noise to the covariance matrix

    dtype: SupportsDType = jnp.complex64
    seed: int = 42

    TEC_CONV: float = -8.4479745  # MHz/mTECU
    convention: str = 'fourier'

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        # make sure all 1D
        if self.freqs.isscalar:
            self.freqs = self.freqs.reshape((1,))
        if self.model_directions.isscalar:
            self.model_directions = self.model_directions.reshape((1,))
        if self.model_times.isscalar:
            self.model_times = self.model_times.reshape((1,))
        if self.model_antennas.isscalar:
            self.model_antennas = self.model_antennas.reshape((1,))
        if self.antennas.isscalar:
            self.antennas = self.antennas.reshape((1,))

        self.num_freqs = len(self.freqs)
        self.num_antenna = len(self.antennas)

        # dtec: au.Quantity  # [num_time, num_dir, num_ant]
        if self.ref_time is None:
            self.ref_time = self.model_times[0]
        if self.ref_ant is None:
            self.ref_ant = self.array_location

        self.enu_geodesics_data, self.dtec = self.simulate_ionosphere()

        if self.enu_geodesics_data.shape[:-1] != (
                len(self.model_times), len(self.model_directions), len(self.model_antennas)
        ):
            raise ValueError(
                f"enu_geodesics_data shape {self.enu_geodesics_data.shape} "
                f"does not match {(len(self.model_times), len(self.model_directions), len(self.model_antennas), -1)}."
            )

        if self.dtec.shape != (len(self.model_times), len(self.model_directions), len(self.model_antennas)):
            raise ValueError(
                f"dtec shape {self.dtec.shape} "
                f"does not match shape {(len(self.model_times), len(self.model_directions), len(self.model_antennas))}."
            )

    @partial(jax.jit, static_argnames=['self', 'northern_hemisphere'])
    def _simulate_ionosphere_jax(self, x0, earth_center_enu, X: GeodesicTuple, northern_hemisphere: bool):

        tomo_kernel = build_ionosphere_tomographic_kernel(
            x0=x0,
            earth_centre=earth_center_enu,
            specification=self.specification,
            S_marg=self.S_marg,
            compute_tec=self.compute_tec,
            northern_hemisphere=northern_hemisphere
        )

        def compute_covariance_row(X1: GeodesicTuple):
            X1 = jax.tree_map(lambda x: x.reshape((1, -1)), X1)
            K = tomo_kernel.cov_func(X1, X)
            return K[0, :]

        mean = tomo_kernel.mean_func(X)

        def pmap_batched_fn(X):
            chunk_size = len(jax.devices())
            X_padded, remove_extra_fn = pad_to_chunksize(X, chunk_size=chunk_size)
            chunked_pmap_fn = chunked_pmap(compute_covariance_row, chunk_size=chunk_size)
            return remove_extra_fn(chunked_pmap_fn(X_padded))

        cov = pmap_batched_fn(X)

        # Simulate from it
        key = jax.random.PRNGKey(self.seed)
        Z = jax.random.normal(key, (cov.shape[-1],), dtype=cov.dtype)

        def cholesky_simulate(jitter):
            cov_plus_jitter = cov + jitter * jnp.eye(cov.shape[0])
            L = jnp.linalg.cholesky(cov_plus_jitter)
            dtec = L @ Z + mean
            return dtec

        def svd_simulate(jitter):
            cov_plus_jitter = cov + jitter * jnp.eye(cov.shape[0])
            max_eig, min_eig, L = msqrt(cov_plus_jitter)
            dtec = L @ Z + mean
            return dtec

        dtec = cholesky_simulate(self.jitter)

        is_nan = jnp.any(jnp.isnan(dtec))

        dtec = lax.cond(
            is_nan,
            lambda: cholesky_simulate(10. * self.jitter),
            lambda: dtec
        )

        is_nan = jnp.any(jnp.isnan(dtec))

        dtec = lax.cond(
            is_nan,
            lambda: svd_simulate(10. * self.jitter),
            lambda: dtec
        )

        return dtec, mean, cov

    def simulate_ionosphere(self) -> Tuple[jax.Array, jax.Array]:
        """
        Compute the tomographic Gaussian representation of the ionosphere.

        Returns:
            mean, covariance
        """
        # Plot Antenna Layout in East North Up frame
        model_antennas_enu = earth_location_to_enu(
            antennas=self.model_antennas,
            array_location=self.array_location,
            time=self.ref_time
        )

        x0 = earth_location_to_enu(
            self.ref_ant,
            array_location=self.array_location,
            time=self.ref_time
        )

        earth_center = ac.GCRS(0 * au.deg, 0 * au.deg, 0 * au.km).transform_to(ac.ITRS).earth_location
        earth_center_enu = earth_location_to_enu(
            antennas=earth_center,
            array_location=self.array_location,
            time=self.ref_time
        )

        northern_hemisphere = self.ref_ant.geodetic.lat > 0 * au.deg

        # Plot model antennas
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))
        ax[0][0].scatter(model_antennas_enu[:, 0].to('m'), model_antennas_enu[:, 1].to('m'), marker='+')
        ax[0][0].set_xlabel(f"East (m)")
        ax[0][0].set_ylabel(f"North (m)")
        ax[0][0].set_title(f"Model Antenna Locations")

        ax[0][0].scatter(x0[0].to('m'), x0[1].to('m'), marker='o', color='red',
                         label="Reference Antenna")

        ax[0][0].scatter(earth_center_enu[0].to('m'), earth_center_enu[1].to('m'), marker='o', color='green',
                         label="Earth Centre")
        ax[0][0].legend()
        fig.savefig(os.path.join(self.plot_folder, "model_antenna_locations.png"))
        plt.close(fig)

        max_baseline = np.max(np.linalg.norm(model_antennas_enu[:, None, :] - model_antennas_enu[None, :, :], axis=-1))
        logger.info(f"Maximum antenna baseline: {max_baseline} km")

        # Plot model directions
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---AIT', 'DEC--AIT']  # AITOFF projection
        wcs.wcs.crval = [0, 0]  # Center of the projection
        wcs.wcs.crpix = [0, 0]
        wcs.wcs.cdelt = [-1, 1]

        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 10), subplot_kw=dict(projection=wcs))
        ax[0][0].scatter(self.model_directions.ra.deg, self.model_directions.dec.deg, marker='o',
                         transform=ax[0][0].get_transform('world'))
        ax[0][0].set_xlabel('Right Ascension')
        ax[0][0].set_ylabel('Declination')
        ax[0][0].set_title("Model Directions")
        fig.savefig(os.path.join(self.plot_folder, "model_directions.png"))
        plt.close(fig)

        enu_geodesics_data = []
        for time in self.model_times:
            model_antennas = earth_location_to_enu(
                antennas=self.model_antennas,
                array_location=self.array_location,
                time=time
            )
            model_directions = icrs_to_enu(
                sources=self.model_directions,
                array_location=self.array_location,
                time=time
            )
            ref_ant = earth_location_to_enu(
                self.ref_ant,
                array_location=self.array_location,
                time=time
            )

            model_time_s = (time.mjd - self.ref_time.mjd) * 86400.

            X = make_coord_array(
                model_time_s[None, None],
                quantity_to_jnp(model_directions),
                quantity_to_jnp(model_antennas, 'km'),
                quantity_to_jnp(ref_ant, 'km')[None, :],
                flat=False
            )  # [1, num_model_dir, num_model_ant, 1, 10]

            enu_geodesics_data.append(X)

        enu_geodesics_data = jnp.concatenate(enu_geodesics_data,
                                             axis=0)  # [num_model_time, num_model_dir, num_model_ant, 1, 10]
        enu_geodesics_data = enu_geodesics_data[..., 0, :]  # [num_model_time, num_model_dir, num_model_ant, 10]
        enu_geodesics_data_flat = jnp.reshape(enu_geodesics_data, (-1, enu_geodesics_data.shape[-1]))  # [-1, 10]

        X1 = GeodesicTuple(
            t=enu_geodesics_data_flat[:, 0:1],
            k=enu_geodesics_data_flat[:, 1:4],
            x=enu_geodesics_data_flat[:, 4:7],
            ref_x=enu_geodesics_data_flat[:, 7:10]
        )

        # Stacking in time, gives shape of data (Nt, Na, Nd)
        print(f"Total number of coordinates: {X1.x.shape[0]}")

        dtec, mean, cov = self._simulate_ionosphere_jax(
            x0=quantity_to_jnp(x0, 'km'),
            earth_center_enu=quantity_to_jnp(earth_center_enu, 'km'),
            X=X1,
            northern_hemisphere=northern_hemisphere
        )

        # plot covariance
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))
        ax[0][0].imshow(cov, origin='lower', aspect='auto')
        ax[0][0].set_title("Covariance")
        fig.savefig(os.path.join(self.plot_folder, "covariance.png"))
        plt.close(fig)

        # plot mean
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))
        ax[0][0].plot(mean)
        ax[0][0].set_title("Mean")
        fig.savefig(os.path.join(self.plot_folder, "mean.png"))
        plt.close(fig)

        dtec = jnp.reshape(dtec, (len(self.model_times), len(self.model_directions), len(self.model_antennas)))
        dtec -= dtec[..., 0:1]  # Subtract arbitrary reference antenna

        for i, time in enumerate(self.model_times):
            for j, direction in enumerate(self.model_directions):
                # Plot mean and covariance
                fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))
                sc = ax[0][0].scatter(model_antennas_enu[:, 0].to('m'), model_antennas_enu[:, 1].to('m'),
                                      c=dtec[i, j, :], marker='o')
                fig.colorbar(sc, ax=ax[0][0])
                ax[0][0].set_xlabel(f"East (m)")
                ax[0][0].set_ylabel(f"North (m)")
                ax[0][0].set_title(f"Sampled dtec {time} {direction}")
                fig.savefig(os.path.join(self.plot_folder, f"dtec_t{i:02d}_d{j:02d}.png"))
                plt.close(fig)

        return enu_geodesics_data, dtec

    @partial(jax.jit, static_argnums=(0,))
    def _compute_beam_jax(self, time_mjd: jax.Array, enu_geodesics_sources: jax.Array) -> jax.Array:
        """
        Compute the beam for a given time and set of sources.

        Args:
            time_mjd: [num_time] The time in JD.
            enu_geodesics_sources: (source_shape) + [num_ant, 10] The source coordinates in the ENU frame.

        Returns:

        """

        shape = np.shape(enu_geodesics_sources)[:-2]
        enu_geodesics_sources = jnp.reshape(
            enu_geodesics_sources, (-1, enu_geodesics_sources.shape[-2], enu_geodesics_sources.shape[-1])
        )  # [num_sources, num_ant, 6]

        freqs = quantity_to_jnp(self.freqs, 'MHz')
        phase_factor = jnp.asarray(self.TEC_CONV) / freqs  # [num_freqs] rad / mTECU

        # Interpolate in time
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(time_mjd, jnp.asarray(self.model_times.mjd))
        dtec = self.dtec[i0] * alpha0 + self.dtec[i1] * alpha1  # [num_model_sources, num_model_ant]
        dtec = jnp.reshape(dtec, (-1,))

        enu_geodesics_data = self.enu_geodesics_data[i0] * alpha0 + self.enu_geodesics_data[
            i1] * alpha1  # [num_model_sources, num_model_ant, 3]
        enu_geodesics_data = jnp.reshape(enu_geodesics_data, (-1, enu_geodesics_data.shape[-1]))

        k = 27

        dtec_interp = batched_convolved_interp(
            enu_geodesics_sources, enu_geodesics_data, dtec, k=k
        )  # [num_sources, num_ant]

        dtec_interp = jnp.reshape(dtec_interp, shape + dtec_interp.shape[1:])

        phase = dtec_interp[..., None] * phase_factor  # (source_shape) + [num_ant, num_freq]

        if self.convention == 'casa':
            constant = jnp.asarray(-1j, self.dtype)
        elif self.convention == 'fourier':
            constant = jnp.asarray(1j, self.dtype)
        else:
            raise ValueError(f"Unknown convention {self.convention}")

        scalar_gain = jnp.exp(constant * phase)
        # set diagonal
        gains = jnp.zeros(phase.shape + (2, 2), self.dtype)
        gains = gains.at[..., 0, 0].set(scalar_gain)
        gains = gains.at[..., 1, 1].set(scalar_gain)
        return gains

    def compute_beam(self, sources: ac.ICRS, phase_tracking: ac.ICRS, array_location: ac.EarthLocation, time: at.Time,
                     **kwargs):

        antennas = earth_location_to_enu(
            antennas=self.antennas,
            array_location=self.array_location,
            time=time
        )
        directions = icrs_to_enu(
            sources=sources,
            array_location=self.array_location,
            time=time
        )  # (sources_shape)

        shape = directions.shape
        directions = directions.reshape((-1,))

        if array_location != self.array_location:
            raise ValueError(f"Array location {array_location} does not match {self.array_location}")

        ref_ant = earth_location_to_enu(
            self.ref_ant,
            array_location=self.array_location,
            time=time
        )

        time_s = (time.mjd - self.ref_time.mjd) * 86400.

        enu_geodesics_sources = make_coord_array(
            time_s[None, None],
            quantity_to_jnp(directions),
            quantity_to_jnp(antennas, 'km'),
            quantity_to_jnp(ref_ant, 'km')[None, :],
            flat=False
        )  # [1, num_dir, num_ant, 1, 10]

        enu_geodesics_sources = enu_geodesics_sources[0, ..., 0, :]  # [num_dir, num_ant, 10]
        enu_geodesics_sources = jnp.reshape(
            enu_geodesics_sources, shape + enu_geodesics_sources.shape[-1:]
        )  # (source_shape) + [num_ant, 10]

        gains = self._compute_beam_jax(
            time_mjd=time.mjd,
            enu_geodesics_sources=enu_geodesics_sources
        )  # (source_shape) + [num_ant, num_freq]

        return gains


def ionosphere_gain_model_factory(phase_tracking: ac.ICRS,
                                  field_of_view: au.Quantity,
                                  angular_separation: au.Quantity,
                                  spatial_separation: au.Quantity,
                                  observation_start_time: at.Time,
                                  observation_duration: timedelta,
                                  temporal_resolution: timedelta,
                                  freqs: au.Quantity,
                                  specification: SPECIFICATION,
                                  array_name: str,
                                  plot_folder: str
                                  ):
    if not field_of_view.unit.is_equivalent(au.deg):
        raise ValueError("Field of view should be in degrees")
    if not angular_separation.unit.is_equivalent(au.deg):
        raise ValueError("Angular separation should be in degrees")
    if not spatial_separation.unit.is_equivalent(au.m):
        raise ValueError("Spatial separation should be in meters")

    fill_registries()
    array = array_registry.get_instance(array_registry.get_match(array_name))

    antennas_itrs = array.get_antennas()
    antennas = antennas_itrs.earth_location
    array_location = mean_itrs(antennas_itrs).earth_location

    if observation_duration == timedelta(0):
        model_times = observation_start_time.reshape((-1,))
    else:
        if temporal_resolution <= timedelta(seconds=0):
            raise ValueError("Temporal resolution should be positive")
        num_times = int(observation_duration.total_seconds() / temporal_resolution.total_seconds()) + 1
        model_times = at.Time(
            (np.arange(num_times) * temporal_resolution.total_seconds() / 86400.) + observation_start_time.mjd,
            format='mjd'
        )
    model_directions = create_spherical_grid(
        pointing=phase_tracking,
        angular_width=0.5 * field_of_view,
        dr=angular_separation
    )

    max_baseline = np.max(
        np.linalg.norm(
            antennas_itrs.cartesian.xyz.T[:, None, :] - antennas_itrs.cartesian.xyz.T[None, :, :],
            axis=-1
        )
    )
    radius = 0.5 * max_baseline

    model_antennas = create_spherical_earth_grid(
        center=array_location,
        radius=radius,
        dr=spatial_separation
    )

    # filter out model antennas that are too far from an actual antenna
    def keep(model_antenna: ac.EarthLocation):
        dist = np.linalg.norm(
            model_antenna.get_itrs().cartesian.xyz - antennas_itrs.cartesian.xyz.T,
            axis=-1
        )
        return np.any(dist < 2. * spatial_separation)

    # List of EarthLocation
    model_antennas = list(filter(keep, model_antennas))
    # Via ITRS then back to EarthLocation
    model_antennas = ac.concatenate(list(map(lambda x: x.get_itrs(), model_antennas))).earth_location

    ionosphere_gain_model = IonosphereGainModel(
        freqs=freqs,
        antennas=antennas,
        array_location=array_location,
        phase_tracking=phase_tracking,
        model_directions=model_directions,
        model_times=model_times,
        model_antennas=model_antennas,
        specification=specification,
        plot_folder=plot_folder
    )
    return ionosphere_gain_model


def test_real_ionosphere_gain_model():

    phase_tracking = ac.ICRS(ra=0 * au.deg, dec=0 * au.deg)
    field_of_view = 4 * au.deg
    angular_separation = 32 * au.arcmin
    spatial_separation = 1000 * au.m
    observation_start_time = at.Time('2021-01-01T00:00:00', scale='utc')
    observation_duration = timedelta(minutes=0)
    temporal_resolution = timedelta(seconds=0)
    freqs = [700e6, 2000e6] * au.Hz
    ionosphere_gain_model = ionosphere_gain_model_factory(
        phase_tracking=phase_tracking,
        field_of_view=field_of_view,
        angular_separation=angular_separation,
        spatial_separation=spatial_separation,
        observation_start_time=observation_start_time,
        observation_duration=observation_duration,
        temporal_resolution=temporal_resolution,
        freqs=freqs,
        specification='light_dawn',
        array_name='dsa2000W',
        plot_folder='plot_ionosphere'
    )


def test_ionosphere_gain_model():
    freqs = [700e6, 2000e6] * au.Hz
    antennas = ac.EarthLocation.from_geodetic([0, 0.01, 0.02, 0.03] * au.deg, [0, 0, 0, 0] * au.deg,
                                              [0, 0, 0, 0] * au.m)
    model_antennas = ac.EarthLocation.from_geodetic(
        [0.005, 0.015, 0.025] * au.deg,
        [0, 0.001, 0.002] * au.deg,
        [0, 0, 0] * au.m
    )

    array_location = ac.EarthLocation(lat=0 * au.deg, lon=0 * au.deg, height=0 * au.m)
    phase_tracking = ac.ICRS(0 * au.deg, 0 * au.deg)
    model_directions = ac.ICRS(ra=[0, 0.] * au.deg, dec=[0., 1.] * au.deg)
    model_times = at.Time(['2021-01-01T00:00:00', '2021-01-01T00:00:30'], scale='utc')

    ionosphere_gain_model = IonosphereGainModel(
        freqs=freqs,
        antennas=antennas,
        array_location=array_location,
        phase_tracking=phase_tracking,
        model_directions=model_directions,
        model_times=model_times,
        model_antennas=model_antennas,
        specification='light_dawn',
        plot_folder='plot_ionosphere'
    )

    assert ionosphere_gain_model.num_freqs == len(freqs)
    assert ionosphere_gain_model.num_antenna == len(antennas)
    assert ionosphere_gain_model.ref_ant == array_location
    assert ionosphere_gain_model.ref_time == model_times[0]
    assert np.all(np.isfinite(ionosphere_gain_model.dtec))
    assert np.all(np.isfinite(ionosphere_gain_model.enu_geodesics_data))


def msqrt(A):
    """
    Computes the matrix square-root using SVD, which is robust to poorly conditioned covariance matrices.
    Computes, M such that M @ M.T = A

    Args:
    A: [N,N] Square matrix to take square root of.

    Returns: [N,N] matrix.
    """
    U, s, Vh = jnp.linalg.svd(A)
    L = U * jnp.sqrt(s)
    max_eig = jnp.max(s)
    min_eig = jnp.min(s)
    return max_eig, min_eig, L
