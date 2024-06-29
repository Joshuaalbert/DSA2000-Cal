import dataclasses
import os
import time as time_mod
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from astropy import units as au, coordinates as ac, time as at
from jax import lax
from jax._src.typing import SupportsDType
from tomographic_kernel.frames import ENU
from tomographic_kernel.models.cannonical_models import SPECIFICATION, build_ionosphere_tomographic_kernel
from tomographic_kernel.tomographic_kernel import GeodesicTuple, TomographicKernel
from tomographic_kernel.utils import make_coord_array

from dsa2000_cal.assets.content_registry import fill_registries, NoMatchFound
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.astropy_utils import create_spherical_grid, create_spherical_earth_grid
from dsa2000_cal.common.coord_utils import earth_location_to_enu, icrs_to_lmn, lmn_to_enu
from dsa2000_cal.common.interp_utils import convolved_interp
from dsa2000_cal.common.jax_utils import chunked_pmap, pad_to_chunksize, multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.gain_models.spherical_interpolator import SphericalInterpolatorGainModel, phi_theta_from_lmn

tfpd = tfp.distributions

TEC_CONV: float = -8.4479745 * au.rad * au.MHz  # rad MHz / mTECU


class CachedIonosphereSimulation(SerialisableBaseModel):
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
        print(data)
        super(CachedIonosphereSimulation, self).__init__(**data)
        _check_ionosphere_cache(self)


def _check_ionosphere_cache(params: CachedIonosphereSimulation):
    if np.shape(params.dtec) != (len(params.model_times), len(params.model_lmn), len(params.model_antennas)):
        raise ValueError(f"Invalid shape for dtec {np.shape(params.dtec)}, should be (time, dir, ant).")


def compare_directions(icrs1: ac.ICRS, icrs2: ac.ICRS, atol=1 * au.arcsec):
    if icrs1.shape != icrs2.shape:
        return False
    return np.all(icrs1.separation(icrs2) <= atol)


def compare_earth_locations(earth_location1: ac.EarthLocation, earth_location2: ac.EarthLocation, atol=1e-3 * au.m):
    if earth_location1.shape != earth_location2.shape:
        return False
    itrs1 = earth_location1.get_itrs()
    itrs2 = earth_location2.get_itrs()
    return np.all(itrs1.separation_3d(itrs2) <= atol)


def compare_times(time1: at.Time, time2: at.Time, atol=1e-3 * au.s):
    if time1.shape != time2.shape:
        return False
    return np.all(np.abs((time1 - time2).sec) * au.s <= atol)


@dataclasses.dataclass(eq=False)
class IonosphereGainModel(SphericalInterpolatorGainModel):
    ...


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

    dtype: SupportsDType = jnp.complex64
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.plot_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)

        self.cache_file = os.path.join(self.cache_folder, f"cached_{self.specification}_{self.seed}.json")

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

    def simulate_ionosphere(self) -> CachedIonosphereSimulation:
        """
        Compute the tomographic Gaussian representation of the ionosphere.

        Returns:
            mean, covariance
        """

        if os.path.exists(self.cache_file):
            cache = CachedIonosphereSimulation.parse_file(self.cache_file)

            if cache.pointing != self.pointing:
                raise ValueError(f"Model pointing does not match {cache.pointing} != {self.pointing}")
            if not compare_earth_locations(cache.model_antennas, self.model_antennas):
                raise ValueError(f"Model antennas do not match {cache.model_antennas} != {self.model_antennas}")
            if not compare_earth_locations(cache.array_location, self.array_location):
                raise ValueError(f"Array location does not match {cache.array_location} != {self.array_location}")
            if not compare_times(cache.model_times, self.model_times):
                raise ValueError(f"Model times do not match {cache.model_times} != {self.model_times}")
            if cache.ref_ant != self.ref_ant:
                raise ValueError(f"Reference antenna does not match {cache.ref_ant} != {self.ref_ant}")
            if cache.ref_time != self.ref_time:
                raise ValueError(f"Reference time does not match {cache.ref_time} != {self.ref_time}")
            if cache.specification != self.specification:
                raise ValueError(f"Specification does not match {cache.specification} != {self.specification}")
            if cache.compute_tec != self.compute_tec:
                raise ValueError(f"Compute TEC does not match {cache.compute_tec} != {self.compute_tec}")
            if cache.S_marg != self.S_marg:
                raise ValueError(f"S_marg does not match {cache.S_marg} != {self.S_marg}")
            if cache.jitter != self.jitter:
                raise ValueError(f"Jitter does not match {cache.jitter} != {self.jitter}")
            if cache.seed != self.seed:
                raise ValueError(f"Seed does not match {cache.seed} != {self.seed}")

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
                phase_tracking=self.pointing
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

        cache = CachedIonosphereSimulation(
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


def interpolate_antennas(antennas_enu: jax.Array, model_antennas_enu: jax.Array, dtec: jax.Array,
                         k: int = 3) -> jax.Array:
    """
    Interpolate from model antennas to antennas.

    Args:
        antennas_enu: [N, 3] antenna coords
        model_antennas_enu: [M, 3] model antenna coords on given spatial resolution
        dtec: [num_time, num_dir, M] dtec to interpolate in mTECU

    Returns:
        dtec: [num_time, num_dir, N]
    """

    if np.shape(model_antennas_enu)[0] < k:
        raise ValueError(f"Too few model antennas, need at least {k}.")

    @partial(multi_vmap, in_mapping="[T,D,M]", out_mapping="[T,D,...]")
    def interp(dtec):
        return convolved_interp(antennas_enu, model_antennas_enu, dtec, k, mode='euclidean')

    dtec_interp = interp(dtec)  # [num_time, num_dir, N]
    return dtec_interp


def create_model_gains(antennas_enu: jax.Array, model_antennas_enu: jax.Array, dtec: jax.Array,
                       model_freqs: au.Quantity, k: int = 3, dtype='complex64') -> jax.Array:
    """
    Compute gains.

    Args:
        antennas_enu: [N, 3] antenna coords
        model_antennas_enu: [M, 3] model antenna coords on given spatial resolution
        dtec: [num_time, num_dir, M] dtec to interpolate in mTECU
        model_freqs: [num_freqs] quantity for freqs.
        k: interp order
        dtype: dtype

    Returns:
        gains: [num_time, num_dir, N, num_freqs, 2, 2] gains
    """
    phase_factor = quantity_to_jnp(TEC_CONV / model_freqs)  # [num_model_freqs] rad / mTECU

    dtec = interpolate_antennas(
        antennas_enu=antennas_enu,
        model_antennas_enu=model_antennas_enu,
        dtec=dtec,
        k=k
    )  # [num_model_times, num_model_dir, num_ant]

    model_phase = dtec[..., None] * phase_factor  # [num_model_times, num_model_dir, num_ant, num_model_freqs]
    model_gains = jnp.zeros(np.shape(model_phase) + (2, 2),
                            dtype=dtype)  # [num_model_times, num_model_dir, num_ant, num_model_freqs, 2, 2]
    scalar_gain = jnp.exp(1j * model_phase)
    model_gains = model_gains.at[..., 0, 0].set(scalar_gain)
    model_gains = model_gains.at[..., 1, 1].set(scalar_gain)
    return model_gains


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


def ionosphere_gain_model_factory(pointing: ac.ICRS | ENU,
                                  model_freqs: au.Quantity,
                                  field_of_view: au.Quantity,
                                  spatial_resolution: au.Quantity,
                                  observation_start_time: at.Time,
                                  observation_duration: au.Quantity,
                                  temporal_resolution: au.Quantity,
                                  specification: SPECIFICATION,
                                  array_name: str,
                                  plot_folder: str,
                                  cache_folder: str,
                                  seed: int
                                  ) -> IonosphereGainModel:
    """
    Simulates ionosphere then crates gain model from it.

    Args:
        pointing: the pointing of antennas. Only scalars for now.
        model_freqs: [num_model_freqs] the frequencies to compute gains at.
        field_of_view: the size of primary beam, FWHM.
        spatial_resolution: the spatial resolution of simulation, should be less than ionosphere spatial scale.
        observation_start_time: when to start simulation.
        observation_duration: How long observation lasts.
        temporal_resolution: Time resolution of simulation, should be less than dynamical time of ionosphere.
        specification: where kind of ionosphere to simulate. See tomographic_kernel.
        array_name: the name of array.
        plot_folder: the place to plot things.
        cache_folder: the place to cache things.
        seed: the random number seed of simulation

    Returns:
        ionosphere gain model
    """
    os.makedirs(plot_folder, exist_ok=True)
    if not model_freqs.unit.is_equivalent(au.Hz):
        raise ValueError("Model frequencies should be in Hz")

    if not spatial_resolution.unit.is_equivalent(au.m):
        raise ValueError("Spatial separation should be in meters")

    if not field_of_view.unit.is_equivalent(au.deg):
        raise ValueError("Field of view should be in degrees")

    if not observation_duration.unit.is_equivalent(au.s):
        raise ValueError("Observation duration should be in seconds")

    if not temporal_resolution.unit.is_equivalent(au.s):
        raise ValueError("Temporal resolution should be in seconds")

    nominal_height = 200 * au.km
    angular_resolution = (spatial_resolution / nominal_height) * au.rad
    print(f"Angular resolution: {angular_resolution.to(au.arcmin)}")

    model_directions = create_spherical_grid(
        pointing=pointing,
        angular_radius=0.5 * field_of_view,
        dr=angular_resolution
    )
    print(f"Number of model directions: {len(model_directions)}")
    # Convert to lmn
    model_lmn = icrs_to_lmn(sources=model_directions, phase_tracking=pointing)  # [num_model_dir, 3]

    # Plot model directions
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))
    ax[0][0].scatter(model_lmn[:, 0], model_lmn[:, 1], marker='o')
    ax[0][0].set_xlabel('l')
    ax[0][0].set_ylabel('m')
    ax[0][0].set_title("Model Directions")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_folder, "model_directions.png"))
    plt.close(fig)

    if observation_duration == 0 * au.s:
        model_times = observation_start_time.reshape((-1,))
    else:
        if temporal_resolution <= 0 * au.s:
            raise ValueError("Temporal resolution should be positive")
        num_times = int(observation_duration / temporal_resolution) + 1
        model_times = observation_start_time + np.arange(num_times) * temporal_resolution
    ref_time = model_times[0]
    print(f"Number of model times: {len(model_times)}")

    fill_registries()
    try:
        array = array_registry.get_instance(array_registry.get_match(array_name))
    except NoMatchFound as e:
        raise ValueError(
            f"Array {array_name} not found in registry. Add it to use the IonosphereGainModel factory."
        ) from e

    antennas = array.get_antennas()
    antennas_itrs = antennas.get_itrs()
    array_location = array.get_array_location()

    radius = np.max(np.linalg.norm(
        antennas.get_itrs().cartesian.xyz.T - array_location.get_itrs().cartesian.xyz,
        axis=-1
    ))
    print(f"Array radius: {radius}")

    model_antennas = create_spherical_earth_grid(
        center=array_location,
        radius=radius,
        dr=spatial_resolution
    )

    # filter out model antennas that are too far from any actual antenna
    def keep(model_antenna: ac.EarthLocation):
        dist = np.linalg.norm(
            model_antenna.get_itrs().cartesian.xyz - antennas_itrs.cartesian.xyz.T,
            axis=-1
        )
        return np.any(dist < spatial_resolution)

    # List of EarthLocation
    model_antennas = list(filter(keep, model_antennas))
    # Via ITRS then back to EarthLocation
    model_antennas = ac.concatenate(list(map(lambda x: x.get_itrs(), model_antennas))).earth_location

    # Plot Antenna Layout in East North Up frame
    model_antennas_enu = earth_location_to_enu(
        antennas=model_antennas,
        array_location=array_location,
        time=ref_time
    ).cartesian.xyz.T

    x0 = earth_location_to_enu(
        array_location,
        array_location=array_location,
        time=ref_time
    ).cartesian.xyz

    antennas_enu = earth_location_to_enu(
        antennas=antennas,
        array_location=array_location,
        time=ref_time
    ).cartesian.xyz.T

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))
    ax[0][0].scatter(antennas_enu[:, 0].to('m'), antennas_enu[:, 1].to('m'), marker='*', c='grey', alpha=0.5,
                     label="Array Antennas")
    ax[0][0].scatter(model_antennas_enu[:, 0].to('m'), model_antennas_enu[:, 1].to('m'), marker='+',
                     label='Model Antennas')
    ax[0][0].set_xlabel(f"East (m)")
    ax[0][0].set_ylabel(f"North (m)")
    ax[0][0].set_title(f"Model Antenna Locations")

    ax[0][0].scatter(x0[0].to('m'), x0[1].to('m'), marker='o', color='red',
                     label="Reference Antenna")

    ax[0][0].legend()
    fig.savefig(os.path.join(plot_folder, "model_antenna_locations.png"))
    plt.close(fig)

    ionosphere_simulation = IonosphereSimulation(
        array_location=array_location,
        pointing=pointing,
        model_lmn=model_lmn,
        model_times=model_times,
        model_antennas=model_antennas,
        specification=specification,
        plot_folder=plot_folder,
        cache_folder=cache_folder,
        seed=seed
    )

    simulation_results = ionosphere_simulation.simulate_ionosphere()

    create_model_gains_jit = jax.jit(lambda antennas_enu, model_antennas_enu, dtec: create_model_gains(
        antennas_enu=antennas_enu, model_antennas_enu=model_antennas_enu, dtec=dtec,
        model_freqs=model_freqs, k=3
    ))

    model_gains = np.asarray(create_model_gains_jit(
        antennas_enu=quantity_to_jnp(earth_location_to_enu(
            antennas=antennas,
            array_location=array_location,
            time=observation_start_time
        ).cartesian.xyz.T),
        model_antennas_enu=quantity_to_jnp(earth_location_to_enu(
            antennas=model_antennas,
            array_location=array_location,
            time=observation_start_time
        ).cartesian.xyz.T),
        dtec=jnp.asarray(simulation_results.dtec)
    )) * au.dimensionless_unscaled  # [num_model_times, num_model_dir, num_ant, num_model_freqs, 2, 2]

    model_phi, model_theta = phi_theta_from_lmn(
        simulation_results.model_lmn[:, 0], simulation_results.model_lmn[:, 1], simulation_results.model_lmn[:, 2]
    )  # [num_model_dir, 3]

    model_phi = model_phi * au.rad
    model_theta = model_theta * au.rad

    return IonosphereGainModel(
        antennas=antennas,
        model_times=simulation_results.model_times,
        model_phi=model_phi,
        model_theta=model_theta,
        model_freqs=model_freqs,
        model_gains=model_gains,
        tile_antennas=False
    )


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
