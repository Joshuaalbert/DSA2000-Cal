import dataclasses
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from astropy import units as au, coordinates as ac, time as at
from tomographic_kernel.frames import ENU
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.assets.content_registry import fill_registries, NoMatchFound
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.common.astropy_utils import create_spherical_grid, create_spherical_earth_grid
from dsa2000_cal.common.coord_utils import earth_location_to_enu, icrs_to_lmn
from dsa2000_cal.common.interp_utils import convolved_interp
from dsa2000_cal.common.jax_utils import multi_vmap
from dsa2000_cal.common.quantity_utils import quantity_to_jnp
from dsa2000_cal.forward_models.systematics.ionosphere_simulation import TEC_CONV, IonosphereSimulation
from dsa2000_cal.gain_models.spherical_interpolator import SphericalInterpolatorGainModel, phi_theta_from_lmn

tfpd = tfp.distributions


@dataclasses.dataclass(eq=False)
class IonosphereGainModel(SphericalInterpolatorGainModel):
    ...


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
