import dataclasses
import logging
import os
from functools import partial
from typing import NamedTuple

import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
from astropy import constants
from ray import serve

import dsa2000_cal.common.context as ctx
from dsa2000_cal.common.array_types import FloatArray, ComplexArray, IntArray
from dsa2000_cal.common.fourier_utils import ApertureTransform
from dsa2000_cal.common.interp_utils import InterpolatedArray
from dsa2000_cal.common.jax_utils import block_until_ready
from dsa2000_cal.common.mixed_precision_utils import mp_policy
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np, time_to_jnp
from dsa2000_cal.common.ray_utils import TimerLog
from dsa2000_cal.common.serialise_utils import SerialisableBaseModel
from dsa2000_cal.common.types import DishEffectsParams
from dsa2000_cal.forward_models.streaming.distributed.common import ForwardModellingRunParams
from dsa2000_cal.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel
from dsa2000_cal.gain_models.beam_gain_model import build_beam_gain_model
from dsa2000_cal.gain_models.gain_model import GainModel
from dsa2000_cal.geodesics.base_geodesic_model import BaseGeodesicModel

logger = logging.getLogger('ray')


class SimulationParams(NamedTuple):
    dish_diameter: jax.Array
    focal_length: jax.Array
    elevation_pointing_error_stddev: jax.Array
    cross_elevation_pointing_error_stddev: jax.Array
    axial_focus_error_stddev: jax.Array
    elevation_feed_offset_stddev: jax.Array
    cross_elevation_feed_offset_stddev: jax.Array
    horizon_peak_astigmatism_stddev: jax.Array
    surface_error_mean: jax.Array
    surface_error_stddev: jax.Array


class StaticDishRealisationParams(NamedTuple):
    elevation_feed_offset: jax.Array
    cross_elevation_feed_offset: jax.Array
    horizon_peak_astigmatism: jax.Array
    surface_error: jax.Array


class DynamicDishRealisationParams(NamedTuple):
    elevation_point_error: jax.Array  # [num_time, num_ant]
    cross_elevation_point_error: jax.Array  # [num_time, num_ant]
    axial_focus_error: jax.Array  # [num_time, num_ant]


class SimulateDishState(NamedTuple):
    beam_aperture: jax.Array

    # Transition parameters
    dish_effect_params: SimulationParams
    static_system_params: StaticDishRealisationParams

    # Static parameters
    L: jax.Array
    M: jax.Array
    dl: jax.Array
    dm: jax.Array
    X: jax.Array
    Y: jax.Array
    dx: jax.Array
    dy: jax.Array
    model_freqs: jax.Array
    model_times: FloatArray
    lvec: FloatArray
    mvec: FloatArray
    lmn_image: jax.Array  # [Nl, Nm, 3]


class SystemGainSimulatorParams(SerialisableBaseModel):
    geodesic_model: BaseGeodesicModel
    init_key: IntArray


class SystemGainSimulatorResponse(NamedTuple):
    gain_model: GainModel


@serve.deployment
class SystemGainSimulator:

    def __init__(self, params: ForwardModellingRunParams, system_gain_simulator_params: SystemGainSimulatorParams):
        self.params = params
        self.system_gain_simulator_params = system_gain_simulator_params
        self.params.plot_folder = os.path.join(self.params.plot_folder, 'system_gain_simulator')
        os.makedirs(self.params.plot_folder, exist_ok=True)

        beam_model = build_beam_gain_model(
            array_name=self.params.ms_meta.array_name,
            times=self.params.ms_meta.times,
            ref_time=self.params.ms_meta.ref_time,
            freqs=self.params.ms_meta.freqs,
            full_stokes=self.params.full_stokes
        )
        dish_gain_model = BaseDishGainModel(
            geodesic_model=system_gain_simulator_params.geodesic_model,
            full_stokes=self.params.full_stokes,
            num_antennas=len(self.params.ms_meta.antennas),
            freqs=self.params.ms_meta.freqs,
            dish_effects_params=self.params.dish_effects_params,
            convention=self.params.ms_meta.convention
        )

        self.state = dish_gain_model.get_state(beam_model=beam_model, key=system_gain_simulator_params.init_key)
        plot_aperture_model_host(
            beam_aperture=self.state.beam_aperture,
            dl=self.state.dl,
            dm=self.state.dm,
            plot_folder=self.params.plot_folder,
            name='beam_model_aperture'
        )

        self.compute_dish_model_jit = jax.jit(dish_gain_model.step)

    async def __call__(self, key, time_idx: int, freq_idx: int) -> SystemGainSimulatorResponse:
        logger.info(f"Simulating dish gains for time {time_idx} and freq {freq_idx}")
        time = time_to_jnp(self.params.ms_meta.times[time_idx], self.params.ms_meta.ref_time)
        freq = quantity_to_jnp(self.params.ms_meta.freqs[freq_idx], 'Hz')
        with TimerLog("Simulating dish gains..."):
            gain_model, model_gains_aperture = block_until_ready(
                self.compute_dish_model_jit(key, time[None], freq[None], self.state))
        gain_model = jax.tree.map(np.asarray, gain_model)
        model_gains_aperture = jax.tree_map(np.asarray, model_gains_aperture)
        plot_aperture_model_host(
            beam_aperture=model_gains_aperture,
            dl=self.state.dl,
            dm=self.state.dm,
            plot_folder=self.params.plot_folder,
            name=f"dish_gains_aperture_T{time_idx}_F{freq_idx}"
        )
        return SystemGainSimulatorResponse(
            gain_model=gain_model
        )


@dataclasses.dataclass(eq=False)
class BaseDishGainModel:
    geodesic_model: BaseGeodesicModel
    full_stokes: bool
    num_antennas: int
    freqs: au.Quantity
    dish_effects_params: DishEffectsParams

    convention: str = 'physical'

    def get_state(self, beam_model: BaseSphericalInterpolatorGainModel, key) -> SimulateDishState:

        # Compute the "perfect" beam aperture pattern
        beam_aperture = self.compute_beam_aperture(
            beam_model=beam_model
        )  # [num_model_times, lres, mres, num_ant/1, num_model_freqs, 2, 2]

        # Compute aperture coordinates
        model_times = beam_model.model_times
        model_freqs = beam_model.model_freqs

        # Check sampling rate
        wavelengths = quantity_to_np(constants.c) / quantity_to_np(self.freqs)
        aperture_sampling_interval = np.min(wavelengths)
        minimal_n = 2 * int(
            quantity_to_np(self.dish_effects_params.dish_diameter) / aperture_sampling_interval) + 1
        n = np.size(beam_model.lvec)
        if n < minimal_n:
            raise ValueError(f"Beam model resolution {np.shape(beam_model.lvec)} is too low for the dish diameter.")

        L, M = jnp.meshgrid(beam_model.lvec, beam_model.mvec, indexing='ij')
        N = jnp.sqrt(1. - (jnp.square(L) + jnp.square(M)))
        lmn_image = jnp.stack([L, M, N], axis=-1)  # [Nl, Nm, 3]
        dl = beam_model.lvec[1] - beam_model.lvec[0]
        dm = beam_model.mvec[1] - beam_model.mvec[0]
        dx = 1. / (dm * n)  # units of wavelength
        dy = 1. / (dl * n)  # units of wavelength
        xvec = (-0.5 * n + jnp.arange(n)) * dx
        yvec = (-0.5 * n + jnp.arange(n)) * dy
        yvec = -yvec  # L = -Y
        Y, X = jnp.meshgrid(yvec, xvec, indexing='ij')

        # Get the dish parameters and sample static ones
        dish_effect_params = SimulationParams(
            **dict((k, quantity_to_jnp(v)) for k, v in self.dish_effects_params.dict().items())
        )
        static_system_params_fn = ctx.transform(lambda: self._get_static_system_params(
            dish_effect_params=dish_effect_params,
            num_antennas=self.num_antennas
        ))
        static_system_params = static_system_params_fn.init(key).fn_val

        return SimulateDishState(
            # Could change depending if beam is static model or not
            beam_aperture=beam_aperture,
            # Static parameters
            model_times=model_times,
            model_freqs=model_freqs,
            lvec=beam_model.lvec,
            mvec=beam_model.mvec,
            dish_effect_params=dish_effect_params,
            L=L,
            M=M,
            dl=dl,
            dm=dm,
            X=X,
            Y=Y,
            dx=dx,
            dy=dy,
            lmn_image=lmn_image,
            static_system_params=static_system_params,
        )

    def _get_static_system_params(self, dish_effect_params: SimulationParams,
                                  num_antennas: int) -> StaticDishRealisationParams:
        """
        Get the system parameters for the dish.

        Args:
            dish_effect_params: the dish effect parameters
            num_antennas: the number of antennas

        Returns:
            the system parameters for the dish
        """
        elevation_feed_offset = dish_effect_params.elevation_feed_offset_stddev * jax.random.normal(
            key=ctx.next_rng_key(),
            shape=(num_antennas,)
        )

        cross_elevation_feed_offset = dish_effect_params.cross_elevation_feed_offset_stddev * jax.random.normal(
            key=ctx.next_rng_key(),
            shape=(num_antennas,)
        )

        horizon_peak_astigmatism = dish_effect_params.horizon_peak_astigmatism_stddev * jax.random.normal(
            key=ctx.next_rng_key(),
            shape=(num_antennas,)
        )

        surface_error = dish_effect_params.surface_error_mean + dish_effect_params.surface_error_stddev * jax.random.normal(
            key=ctx.next_rng_key(),
            shape=(num_antennas,)
        )

        return StaticDishRealisationParams(
            elevation_feed_offset=elevation_feed_offset,
            cross_elevation_feed_offset=cross_elevation_feed_offset,
            horizon_peak_astigmatism=horizon_peak_astigmatism,
            surface_error=surface_error
        )

    def _get_dynamic_system_params(self, dish_effect_params: SimulationParams, num_antennas: int,
                                   num_times: int) -> DynamicDishRealisationParams:
        elevation_point_error = dish_effect_params.elevation_pointing_error_stddev * jax.random.normal(
            key=ctx.next_rng_key(),
            shape=(num_times, num_antennas,)
        )

        cross_elevation_point_error = dish_effect_params.cross_elevation_pointing_error_stddev * jax.random.normal(
            key=ctx.next_rng_key(),
            shape=(num_times, num_antennas,)
        )

        axial_focus_error = dish_effect_params.axial_focus_error_stddev * jax.random.normal(
            key=ctx.next_rng_key(),
            shape=(num_times, num_antennas,)
        )
        return DynamicDishRealisationParams(
            elevation_point_error=elevation_point_error,
            cross_elevation_point_error=cross_elevation_point_error,
            axial_focus_error=axial_focus_error
        )

    def step(self, key, times: FloatArray, freqs: FloatArray, state: SimulateDishState):
        # Compute the aperture field
        _, elevation_rad = self.geodesic_model.compute_far_field_geodesic(
            times=times,
            lmn_sources=jnp.asarray([[0., 0., 1.]], mp_policy.angle_dtype),
            return_elevation=True
        )  # [num_sources, num_time, num_ant]
        elevation_rad = elevation_rad[0, :, :]  # [num_time, num_ant]
        elevation_rad = jnp.mean(elevation_rad, axis=0)  # [num_ant]
        interp = InterpolatedArray(state.model_freqs, state.beam_aperture, axis=4, regular_grid=True,
                                   check_spacing=False)
        beam_aperture = interp(freqs)  # [num_freqs, num_model_times, lres, mres, num_ant/1, 2, 2]
        interp = InterpolatedArray(state.model_times, beam_aperture, axis=1, regular_grid=True,
                                   check_spacing=False)
        beam_aperture = interp(times)  # [num_times, num_freqs, lres, mres, num_ant/1, 2, 2]
        beam_aperture = jnp.moveaxis(beam_aperture, 1, 4)  # [num_times, lres, mres, num_ant/1, num_freqs, 2, 2]

        model_wavelengths = quantity_to_jnp(constants.c) / freqs
        compute_dish_aperture_fn = ctx.transform(self.compute_dish_aperture)
        model_gains_aperture = compute_dish_aperture_fn.init(key,
                                                             beam_aperture=beam_aperture,
                                                             elevation_rad=elevation_rad,
                                                             dish_effect_params=state.dish_effect_params,
                                                             full_stokes=self.full_stokes,
                                                             X=state.X,
                                                             Y=state.Y,
                                                             model_wavelengths=model_wavelengths,
                                                             static_system_params=state.static_system_params
                                                             ).fn_val  # [num_times, lres, mres, num_ant, num_freqs[, 2, 2]]

        model_gains_image = self.compute_dish_image(
            model_gains_aperture=model_gains_aperture,
            dx=state.dx,
            dy=state.dy
        )  # [num_model_times, lres, mres, num_ant, num_model_freqs[, 2, 2]]

        gain_model = BaseSphericalInterpolatorGainModel(
            model_freqs=freqs,
            model_times=times,
            lvec=state.lvec,
            mvec=state.mvec,
            model_gains=model_gains_image,
            tile_antennas=False,
            full_stokes=self.full_stokes
        )

        print(gain_model)

        return gain_model, model_gains_aperture

    def compute_dish_image(self, model_gains_aperture: jax.Array, dx: FloatArray, dy: FloatArray) -> jax.Array:
        """
        Compute the model field in the image plane.

        Args:
            model_gains_aperture: [num_model_times, lres, mres, num_ant, num_model_freqs[, 2, 2]]

        Returns:
            [num_model_times, lres, mres, num_ant, num_model_freqs[, 2, 2]]
        """
        am = ApertureTransform(convention=self.convention)
        model_gains_image = am.to_image(
            f_aperture=model_gains_aperture, axes=(1, 2), dx=dy, dy=dx  # l and m switched
        )  # [num_model_times, lres, mres, num_ant, num_model_freqs[, 2, 2]]
        return model_gains_image

    def compute_dish_aperture(self, beam_aperture: jax.Array,
                              elevation_rad: jax.Array,
                              X: jax.Array,
                              Y: jax.Array,
                              model_wavelengths: jax.Array,
                              dish_effect_params: SimulationParams,
                              static_system_params: StaticDishRealisationParams,
                              full_stokes: bool
                              ) -> jax.Array:
        """
        Computes the E-field at the aperture of the dish.

        Args:
            beam_aperture: [num_times, lres, mres, num_ant/1, num_freqs[, 2, 2]]
            elevation_rad: [num_ant] the elevation
            X: [lres, mres] x position in units of wavelength
            Y: [lres, mres] y position in units of wavelength
            model_wavelengths: [num_freqs] the model wavelengths
            dish_effect_params: the dish effect parameters
            static_system_params: the static system parameters
            full_stokes: if True, return full stokes

        Returns:
            [num_times, lres, mres, num_ant, num_freqs, 2, 2]
        """

        num_model_times = np.shape(beam_aperture)[0]
        dynamic_system_params = self._get_dynamic_system_params(
            dish_effect_params=dish_effect_params,
            num_antennas=self.num_antennas,
            num_times=num_model_times
        )

        focal_length = dish_effect_params.focal_length
        dish_diameter = dish_effect_params.dish_diameter  # []
        R = 0.5 * dish_diameter  # []

        X = X[None, :, :, None, None] * model_wavelengths  # [1, lres, mres, 1, num_freqs]
        Y = Y[None, :, :, None, None] * model_wavelengths  # [1, lres, mres, 1, num_freqs]
        r = jnp.sqrt(X ** 2 + Y ** 2)
        diameter_mask = r <= R  # [1, lres, mres, 1, num_freqs]
        focal_ratio = r / focal_length  # [1, lres, mres, 1, num_freqs]

        elevation_point_error = dynamic_system_params.elevation_point_error[:, None, None, :,
                                None]  # [num_times, 1, 1, num_ant, 1]
        cross_elevation_point_error = dynamic_system_params.cross_elevation_point_error[:, None, None, :,
                                      None]  # [num_times, 1, 1, num_ant, 1]
        axial_focus_error = dynamic_system_params.axial_focus_error[:, None, None, :,
                            None]  # [num_times, 1, 1, num_ant, 1]
        elevation_feed_offset = static_system_params.elevation_feed_offset[None, None, None, :,
                                None]  # [1, 1, 1, num_ant, 1]
        cross_elevation_feed_offset = static_system_params.cross_elevation_feed_offset[None, None, None, :,
                                      None]  # [1, 1, 1, num_ant, 1]
        horizon_peak_astigmatism = static_system_params.horizon_peak_astigmatism[None, None, None, :,
                                   None]  # [1, 1, 1, num_ant, 1]
        surface_error = static_system_params.surface_error[None, None, None, :, None]  # [1, 1, 1, num_ant, 1]
        elevation_rad = elevation_rad[None, None, None, :, None]  # [1, 1, 1, num_ant, 1]

        pointing_error = elevation_point_error * X - cross_elevation_point_error * Y

        sin_theta_p = focal_ratio / (1. + 0.25 * focal_ratio ** 2)
        cos_theta_p = (1. - 0.25 * focal_ratio ** 2) / (1. + 0.25 * focal_ratio ** 2)

        cos_phi = jnp.where(r == 0., 1., X / r)
        sin_phi = jnp.where(r == 0., 0., Y / r)

        feed_shift_error = (
                axial_focus_error * cos_theta_p
                - elevation_feed_offset * sin_theta_p * cos_phi
                - cross_elevation_feed_offset * sin_theta_p * sin_phi
        )
        cos_2phi = 2. * cos_phi ** 2 - 1.
        cos_elevation = jnp.cos(elevation_rad)  # [num_times, 1, 1, num_ant, 1]
        peak_astigmatism = horizon_peak_astigmatism * cos_elevation
        astigmatism_error = peak_astigmatism * (r / R) ** 2 * cos_2phi

        total_path_length_error = pointing_error + feed_shift_error + astigmatism_error + surface_error  # [num_times, lres, mres, num_ant, num_freqs]

        if self.convention == 'engineering':
            phase = (2 * jnp.pi) * (
                    total_path_length_error / model_wavelengths)  # [num_times, lres, mres, num_ant, num_freqs]
        elif self.convention == 'physical':
            phase = (-2 * jnp.pi) * (
                    total_path_length_error / model_wavelengths)  # [num_times, lres, mres, num_ant, num_freqs]
        else:
            raise ValueError(f"Unknown convention {self.convention}")

        # Multiple in aperature
        aperture_field = mp_policy.cast_to_vis(jax.lax.complex(jnp.cos(phase), jnp.sin(phase)))  #
        if full_stokes:
            diameter_mask = diameter_mask[..., None, None]  # [1, lres, mres, 1, num_freqs, 1, 1]
            aperture_field = aperture_field[
                ..., None, None]  # [num_times, lres, mres, num_ant, num_model_freqs, 1, 1]
        # Zeros outside the dish
        model_gains_aperture = jnp.where(
            diameter_mask,
            aperture_field * beam_aperture,
            jnp.zeros((), mp_policy.vis_dtype)
        )  # [num_times, lres, mres, num_ant, num_freqs[, 2, 2]]
        return model_gains_aperture

    def compute_beam_aperture(self, beam_model: BaseSphericalInterpolatorGainModel) -> ComplexArray:
        """
        Compute the beam aperture from the beam model.

        Args:
            beam_model: the beam model

        Returns:
            the beam aperture [num_model_times, lres, mres, num_ant/1, num_model_freqs, 2, 2]
        """
        dl = beam_model.lvec[1] - beam_model.lvec[0]
        dm = beam_model.mvec[1] - beam_model.mvec[0]
        beam_image = beam_model.model_gains  # [num_model_times, lres, mres, [num_ant,] num_model_freqs[, 2, 2]]
        if beam_model.tile_antennas:
            beam_image = beam_image[:, :, :, None,
                         ...]  # [num_model_times, lres, mres, num_ant/1, num_model_freqs, 2, 2]
        am = ApertureTransform(convention=self.convention)
        beam_aperture = am.to_aperture(
            f_image=beam_image,
            axes=(1, 2),
            dl=dl,
            dm=dm
        )  # [num_model_times, lres, mres, [num_ant,] num_model_freqs[, 2, 2]]
        if np.shape(beam_aperture) != np.shape(beam_image):
            raise ValueError(f"Expected {np.shape(beam_image)}, got {np.shape(beam_aperture)}")
        # use callback to plot the aperture model
        return mp_policy.cast_to_vis(beam_aperture)


def plot_aperture_model_host(beam_aperture: jax.Array, dl, dm, plot_folder: str, name: str) -> jnp.bool_:
    output = os.path.join(plot_folder, f'{name}.png')
    if os.path.exists(output):
        return np.asarray(False)
    beam_aperture = beam_aperture[0]  # [lres, mres, [num_ant,] num_model_freqs[, 2, 2]]
    length = len(np.shape(beam_aperture))
    select = [slice(None)] * 2 + [0] * (length - 2)
    beam_aperture = beam_aperture[tuple(select)]  # [lres, mres]
    # L=-Y, M=X
    beam_aperture = beam_aperture[::-1, :]  # [lres, mres]
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    # plot amp and phase over X Y
    dy = 1. / (dl * np.shape(beam_aperture)[0])
    ymin = -0.5 * np.shape(beam_aperture)[0] * dy
    ymax = ymin + np.shape(beam_aperture)[0] * dy
    dx = 1. / (dm * np.shape(beam_aperture)[1])
    xmin = -0.5 * np.shape(beam_aperture)[1] * dx
    xmax = xmin + np.shape(beam_aperture)[1] * dx
    extent = [xmin, xmax, ymin, ymax]
    axs[0].imshow(np.abs(beam_aperture).T, extent=extent,
                  origin='lower', aspect='auto',
                  interpolation='nearest')
    axs[0].set_title('Amplitude')
    axs[0].set_ylabel('X [wavelengths]')
    axs[0].set_xlabel('Y [wavelengths]')
    axs[1].imshow(np.angle(beam_aperture).T, extent=extent,
                  origin='lower', aspect='auto', cmap='hsv',
                  interpolation='nearest')
    axs[1].set_title('Phase')
    axs[1].set_ylabel('X [wavelengths]')
    axs[1].set_xlabel('Y [wavelengths]')
    plt.savefig(output)
    plt.close(fig)
    return np.asarray(True)


def plot_aperture_model_callback(beam_aperture: ComplexArray, dl: FloatArray, dm: FloatArray,
                                 plot_folder: str, name: str) -> jnp.bool_:
    """
    Plot the aperture model.

    Args:
        beam_aperture: [num_model_times, lres, mres, num_ant/1, num_model_freqs, 2, 2]
        dl: the l resolution
        dm: the m resolution
    """

    return jax.experimental.io_callback(
        partial(plot_aperture_model_host, plot_folder=plot_folder, name=name),
        jax.ShapeDtypeStruct((), jnp.bool_), beam_aperture, dl, dm,
        ordered=False
    )


def test_aperture_transform():
    am = ApertureTransform(convention='physical')
    beam_gain_model = build_beam_gain_model(array_name='dsa2000W_small', resolution=513)
    beam_image = beam_gain_model.model_gains  # [num_model_times, lres, mres, num_model_freqs, 2, 2]
    beam_image = beam_image[0, :, :, 0, 0, 0]
    dl = beam_gain_model.lvec[1] - beam_gain_model.lvec[0]
    dm = beam_gain_model.mvec[1] - beam_gain_model.mvec[0]
    beam_aperture = am.to_aperture(beam_image, axes=(-1, -2), dl=dl, dm=dm)
    print(beam_aperture)
    import pylab as plt
    plt.imshow(np.abs(beam_image), origin='lower')
    plt.show()
    plt.imshow(np.angle(beam_image), origin='lower')
    plt.show()
    plt.imshow(np.abs(beam_aperture), origin='lower')
    plt.show()
    plt.imshow(np.angle(beam_aperture), origin='lower')
    plt.show()
    dx = 1. / (dm * np.shape(beam_aperture)[0])
    dy = 1. / (dl * np.shape(beam_aperture)[1])
    beam_image_rec = am.to_image(beam_aperture, axes=(-1, -2), dx=dx, dy=dy)
    plt.imshow(np.abs(beam_image_rec), origin='lower')
    plt.show()
    plt.imshow(np.angle(beam_image_rec), origin='lower')
    plt.show()
