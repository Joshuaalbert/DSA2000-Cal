import dataclasses
import os
import time as time_mod

import jax
import numpy as np
import pylab as plt
from astropy import units as au, coordinates as ac, time as at
from jax import numpy as jnp, lax
from jax._src.typing import SupportsDType
from tensorflow_probability.substrates import jax as tfp
from tomographic_kernel.models.cannonical_models import SPECIFICATION, build_ionosphere_tomographic_kernel
from tomographic_kernel.tomographic_kernel import GeodesicTuple, TomographicKernel
from tomographic_kernel.utils import make_coord_array

from dsa2000_cal.common.cache_utils import check_cache
from dsa2000_common.common.coord_utils import earth_location_to_enu, lmn_to_enu
from dsa2000_common.common.jax_utils import pad_to_chunksize, chunked_pmap
from dsa2000_cal.common.linalg_utils import msqrt
from dsa2000_common.common.mixed_precision_utils import complex_type
from dsa2000_common.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel

TEC_CONV: float = -8.4479745 * au.rad * au.MHz  # rad MHz / mTECU


class IonosphereSimulationCache(SerialisableBaseModel):
    # Simulation parameters
    specification: SPECIFICATION

    compute_tec: bool
    S_marg: int
    jitter: float
    seed: int

    array_location: ac.EarthLocation
    pointing: ac.ICRS
    ref_ant: ac.EarthLocation
    ref_time: at.Time

    # Model coords
    model_times: at.Time  # [num_model_time]
    model_lmn: au.Quantity  # [num_model_dir, 3]
    model_antennas: ac.EarthLocation  # [num_model_ant]

    # Data
    dtec: np.ndarray  # [num_model_time, num_model_dir, num_model_ant]

    def __init__(self, **data) -> None:
        # Call the superclass __init__ to perform the standard validation
        super(IonosphereSimulationCache, self).__init__(**data)


@dataclasses.dataclass(eq=False)
class IonosphereSimulation:
    """
    Uses nearest neighbour interpolation to compute the gain model.
    """

    # Simulation parameters
    array_location: ac.EarthLocation
    pointing: ac.ICRS
    model_times: at.Time  # [num_model_time]
    model_lmn: au.Quantity  # [num_model_dir, 3]
    model_antennas: ac.EarthLocation  # [num_model_ant]

    specification: SPECIFICATION
    plot_folder: str
    cache_folder: str

    ref_ant: ac.EarthLocation | None = None
    ref_time: at.Time | None = None

    compute_tec: bool = True  # Faster to compute TEC only and differentiate later
    S_marg: int = 25
    jitter: float = 0.05  # Adds 0.05 mTECU noise to the covariance matrix

    dtype: SupportsDType = complex_type
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)

        self.cache_file = os.path.join(self.cache_folder, f"cache_{self.specification}_{self.seed}.json")

        # make sure all 1D
        if self.model_lmn.isscalar:
            raise ValueError("Model directions must be [num_model_dirs, 3]")
        if self.model_times.isscalar:
            raise ValueError("Model times must be a list of Time objects.")
        if self.model_antennas.isscalar:
            raise ValueError("Model antennas must be a list of EarthLocations.")

        if self.ref_time is None:
            self.ref_time = self.model_times[0]
        if self.ref_ant is None:
            self.ref_ant = self.array_location
        self.earth_center = ac.EarthLocation.from_geocentric(0 * au.m, 0 * au.m, 0 * au.m)

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
            X1 = jax.tree.map(lambda x: x.reshape((1, -1)), X1)
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

    def simulate_ionosphere(self) -> IonosphereSimulationCache:
        """
        Compute the tomographic Gaussian representation of the ionosphere.

        Returns:
            mean, covariance
        """

        if os.path.exists(self.cache_file):
            cache = IonosphereSimulationCache.parse_file(self.cache_file)

            check_cache(
                cache_model=cache,
                pointing=self.pointing,
                model_antennas=self.model_antennas,
                array_location=self.array_location,
                model_times=self.model_times,
                ref_ant=self.ref_ant,
                ref_time=self.ref_time,
                specification=self.specification,
                compute_tec=self.compute_tec,
                S_marg=self.S_marg,
                jitter=self.jitter,
                seed=self.seed
            )
            print(f"Successfully loaded cache {self.cache_file}.")
            return cache

        # Plot Antenna Layout in East North Up frame
        model_antennas_enu = earth_location_to_enu(
            antennas=self.model_antennas,
            array_location=self.array_location,
            time=self.ref_time
        ).cartesian.xyz.T

        x0 = earth_location_to_enu(
            self.array_location,
            array_location=self.array_location,
            time=self.ref_time
        ).cartesian.xyz

        earth_center_enu = earth_location_to_enu(
            antennas=self.earth_center,
            array_location=self.array_location,
            time=self.ref_time
        ).cartesian.xyz

        northern_hemisphere = self.ref_ant.geodetic.lat > 0 * au.deg

        max_baseline = np.max(np.linalg.norm(
            model_antennas_enu.to('km').value[:, None, :] - model_antennas_enu.to('km').value[None, :, :],
            axis=-1)) * au.km
        print(f"Maximum antenna baseline: {max_baseline}")

        enu_geodesics_data = []
        for time in self.model_times:
            model_antennas = earth_location_to_enu(
                antennas=self.model_antennas,
                array_location=self.array_location,
                time=time
            ).cartesian.xyz.T
            model_directions = lmn_to_enu(
                lmn=self.model_lmn,
                array_location=self.array_location,
                time=time,
                phase_center=self.pointing
            ).cartesian.xyz.T
            ref_ant = earth_location_to_enu(
                self.ref_ant,
                array_location=self.array_location,
                time=time
            ).cartesian.xyz

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

        run_fn = jax.jit(lambda: self._simulate_ionosphere_jax(
            x0=quantity_to_jnp(x0, 'km'),
            earth_center_enu=quantity_to_jnp(earth_center_enu, 'km'),
            X=X1,
            northern_hemisphere=northern_hemisphere
        )).lower().compile()

        t0 = time_mod.time()
        dtec, mean, cov = run_fn()
        t1 = time_mod.time()

        print(f"Sucessfully completed ionosphere simulation in {t1 - t0:.2f} seconds")

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

        dtec = jnp.reshape(dtec, (len(self.model_times), len(self.model_lmn), len(self.model_antennas)))
        dtec -= dtec[..., 0:1]  # Subtract arbitrary reference antenna

        # for i, time in enumerate(self.model_times):
        #     for j, direction in enumerate(self.model_lmn):
        #         # Plot mean and covariance
        #         fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))
        #         sc = ax[0][0].scatter(model_antennas_enu[:, 0].to('m'), model_antennas_enu[:, 1].to('m'),
        #                               c=dtec[i, j, :], marker='o')
        #         fig.colorbar(sc, ax=ax[0][0])
        #         ax[0][0].set_xlabel(f"East (m)")
        #         ax[0][0].set_ylabel(f"North (m)")
        #         ax[0][0].set_title(f"Sampled dtec {time} {direction}")
        #         fig.savefig(os.path.join(self.plot_folder, f"dtec_t{i:02d}_d{j:02d}.png"))
        #         plt.close(fig)

        cache = IonosphereSimulationCache(
            specification=self.specification,
            compute_tec=self.compute_tec,
            S_marg=self.S_marg,
            jitter=self.jitter,
            seed=self.seed,
            array_location=self.array_location,
            pointing=self.pointing,
            ref_ant=self.ref_ant,
            ref_time=self.ref_time,
            model_times=self.model_times,
            model_lmn=self.model_lmn,
            model_antennas=self.model_antennas,
            dtec=np.asarray(dtec)
        )

        with open(self.cache_file, 'w') as fp:
            fp.write(cache.json(indent=2))

        return cache


@dataclasses.dataclass(eq=False)
class ToTFPKernel(tfp.math.psd_kernels.PositiveSemidefiniteKernel):
    """
    Wrapper for a TFP kernel.

    Args:
        tomo_kernel: the TomographicKernel kernel to wrap.
    """
    tomo_kernel: TomographicKernel

    def __post_init__(self):
        tfp.math.psd_kernels.PositiveSemidefiniteKernel.__init__(self, feature_ndims=1)

    def matrix(self, x1, x2, name='matrix'):
        X1 = GeodesicTuple(
            t=x1[:, 0:1],
            k=x1[:, 1:4],
            x=x1[:, 4:7],
            ref_x=x1[:, 7:10]
        )

        X2 = GeodesicTuple(
            t=x2[:, 0:1],
            k=x2[:, 1:4],
            x=x2[:, 4:7],
            ref_x=x2[:, 7:10]
        )

        return self.tomo_kernel.cov_func(X1, X2)
