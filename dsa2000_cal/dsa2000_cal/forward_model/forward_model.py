import dataclasses
import os
from functools import partial
from typing import Tuple, List

import astropy.coordinates as ac
import astropy.units as au
import jax
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import offset_by
from jax import lax
from tomographic_kernel.models.cannonical_models import SPECIFICATION

from dsa2000_cal.abc import AbstractSourceModel
from dsa2000_cal.calibration.calibration import Calibration, CalibrationData, CalibrationParams
from dsa2000_cal.common import wgridder
from dsa2000_cal.common.coord_utils import icrs_to_lmn
from dsa2000_cal.common.fits_utils import ImageModel
from dsa2000_cal.common.fourier_utils import find_optimal_fft_size
from dsa2000_cal.common.noise import calc_baseline_noise
from dsa2000_cal.common.quantity_utils import quantity_to_jnp, quantity_to_np
from dsa2000_cal.gain_models.beam_gain_model import beam_gain_model_factory
from dsa2000_cal.gain_models.dish_effects_gain_model import DishEffectsGainModel, DishEffectsGainModelParams
from dsa2000_cal.gain_models.gain_model import ProductGainModel
from dsa2000_cal.gain_models.ionosphere_gain_model import ionosphere_gain_model_factory
from dsa2000_cal.measurement_sets.measurement_set import VisibilityCoords, MeasurementSet, VisibilityData
from dsa2000_cal.predict.fft_stokes_I_predict import FFTStokesIPredict, FFTStokesIModelData
from dsa2000_cal.predict.point_predict import PointPredict, PointModelData
from dsa2000_cal.source_models.corr_translation import linear_to_stokes
from dsa2000_cal.source_models.fits_stokes_I_source_model import FitsStokesISourceModel


@dataclasses.dataclass(eq=False)
class ForwardModel:
    """
    Runs forward modelling using a sharded data structure over devices.
    """
    ms: MeasurementSet
    simulation_source_models: List[AbstractSourceModel]

    # Dish effect model parameters
    dish_effect_params: DishEffectsGainModelParams

    # Ionosphere model parameters
    ionosphere_specification: SPECIFICATION

    # Calibration parameters
    calibrator_source_models: List[AbstractSourceModel]
    time_solution_interval: int

    # Imaging parameters
    field_of_view: au.Quantity

    plot_folder: str
    cache_folder: str
    ionosphere_seed: int
    dish_effects_seed: int
    seed: int

    oversample_factor: float = 2.5
    epsilon: float = 1e-4
    convention: str = 'casa'

    def __post_init__(self):
        if not self.field_of_view.unit.is_equivalent(au.deg):
            raise ValueError(f"Expected field_of_view to be in degrees, got {self.field_of_view.unit}")
        self.beam_gain_model = beam_gain_model_factory(self.ms.meta.array_name)
        self.dft_predict = PointPredict()
        self.key = jax.random.PRNGKey(self.seed)

        self.calibration = Calibration(num_iterations=15)

    def _simulate_systematics(self) -> ProductGainModel:
        """
        Simulate systematics such as ionosphere and dish effects.

        Returns:
            system_gain_model: the system gain model
        """

        if len(self.ms.meta.times) > 1:
            # temporal_resolution = (self.ms.meta.times[1] - self.ms.meta.times[0]).sec * au.s
            observation_duration = (self.ms.meta.times[-1] - self.ms.meta.times[0]).sec * au.s
            # So that there are two time points, making it scalable but not realistic
            temporal_resolution = observation_duration
        else:
            temporal_resolution = 0 * au.s
            observation_duration = 0 * au.s

        # TODO: Improve performance -- Will take too long for realistic sized datasets
        ionosphere_gain_model = ionosphere_gain_model_factory(
            phase_tracking=self.ms.meta.phase_tracking,
            field_of_view=self.discrete_sky_model.get_angular_diameter() + 32 * au.arcmin,
            angular_separation=32 * au.arcmin,
            spatial_separation=1 * au.km,
            observation_start_time=self.ms.meta.times[0],
            observation_duration=observation_duration,
            temporal_resolution=temporal_resolution,
            specification=self.ionosphere_specification,
            array_name=self.ms.meta.array_name,
            plot_folder=os.path.join(self.plot_folder, 'plot_ionosphere'),
            cache_folder=os.path.join(self.cache_folder, 'cache_ionosphere'),
            seed=self.ionosphere_seed

        )
        dish_effect_gain_model = DishEffectsGainModel(
            beam_gain_model=self.beam_gain_model,
            model_times=self.ms.meta.times,
            dish_effect_params=self.dish_effect_params,
            seed=self.dish_effects_seed,
            cache_folder=os.path.join(self.cache_folder, 'cache_dish_effects_model'),
            plot_folder=os.path.join(self.plot_folder, 'plot_dish_effects_model'),
        )

        # Order is by right multiplication of systematics encountered by radiation from source to observer
        system_gain_model = dish_effect_gain_model @ ionosphere_gain_model

        return system_gain_model

    def _simulate_visibilties(self, system_gain_model: ProductGainModel):
        """
        Simulate visibilities using the system gain model, and store the results in the MS.

        Args:
            system_gain_model: the system gain model
        """
        # Build full gain model
        brightness = self.discrete_sky_model.get_source_model_linear(
            freqs=quantity_to_jnp(self.ms.meta.freqs)
        )  # [num_sources, num_freqs, 2, 2]

        gen = self.ms.create_block_generator(vis=False, weights=False, flags=False)
        gen_response = None
        while True:
            try:
                time, visibility_coords, _ = gen.send(gen_response)
            except StopIteration:
                break

            # Compute the lmn coordinates of the sources
            lmn = self.discrete_sky_model.compute_lmn(phase_tracking=self.ms.meta.phase_tracking,
                                                      time=time)  # [num_sources, 3]

            # Get gains
            gains = system_gain_model.compute_gain(
                freqs=self.ms.meta.freqs,
                sources=self.discrete_sky_model.coords_icrs,
                phase_tracking=self.ms.meta.phase_tracking,
                array_location=self.ms.meta.array_location,
                time=time,
                mode='fft'
            )  # [num_sources, num_ant, num_freq, 2, 2]

            # Simulate visibilities

            dft_model_data = PointModelData(
                image=brightness,
                gains=gains[:, None, ...],  # Add time dim [1]
                lmn=lmn
            )

            self.key, key = jax.random.split(self.key, 2)
            sim_vis_data = self._simulate_jax(
                key=key,
                dft_model_data=dft_model_data,
                visibility_coords=visibility_coords
            )

            # Save the results by pushing the response back to the generator
            gen_response = jax.tree_map(np.asarray, sim_vis_data)

    def forward(self):
        # Simulate systematics
        system_gain_model = self._simulate_systematics()
        # Simulate visibilities
        self._simulate_visibilties(system_gain_model=system_gain_model)
        # Calibrate visibilities
        self._calibrate_visibilities()
        # Subtract visibilities
        subtracted_ms = self._subtract_visibilities()
        # Image visibilities
        self._image_visibilities(image_name='dirty_image', ms=subtracted_ms)

    def _subtract_visibilities(self) -> MeasurementSet:
        subtracted_ms = self.ms.clone(ms_folder='subtracted_ms')
        # Predict model with calibrated gains
        gen = subtracted_ms.create_block_generator(vis=True, weights=True, flags=False)
        gen_response = None
        while True:
            try:
                time, visibility_coords, data = gen.send(gen_response)
            except StopIteration:
                break

            # Predict model with calibrated gains
            vis_model = ...
            residual = data.vis - vis_model
            residual_variance = ...

            # Assuming weights represent 1/variance, we should add to the variance to account for uncertainty in the calibration
            # weights = 1 / (1 / weights + 1 / calibration_variance)
            weights = data.weights * residual_variance / (data.weights + residual_variance)

            # Subtract model from data
            gen_response = VisibilityData(
                vis=residual,
                weights=weights
            )

    def _calibrate_visibilities(self):
        # Average

        if self.time_solution_interval > 1:
            # Create an averaged MS, or some on the fly averaged generator
            avg_ms: MeasurementSet = ...
        else:
            avg_ms = self.ms

        gen = avg_ms.create_block_generator(vis=True, weights=True, flags=False)
        gen_response = None

        init_params = self.calibration.get_init_params(
            num_source=self.discrete_sky_model.num_sources,
            num_time=1,
            num_ant=len(self.ms.meta.antennas),
            num_chan=len(self.ms.meta.freqs)
        )  # [num_source, 1, num_ant, num_freqs, 2, 2]

        while True:
            try:
                time, visibility_coords, data = gen.send(gen_response)
            except StopIteration:
                break

            lmn = self.discrete_sky_model.compute_lmn(phase_tracking=self.ms.meta.phase_tracking, time=time)
            self.key, key = jax.random.split(self.key)
            params, subtracted_data = self._calibrate_jax(
                key=key,
                init_params=init_params,
                lmn=lmn,
                visibility_coords=visibility_coords,
                data=data
            )

            # Store params
            init_params = params

    @partial(jax.jit, static_argnums=(0,))
    def _calibrate_jax(self, key, init_params: CalibrationParams, lmn: jax.Array,
                       visibility_coords: VisibilityCoords, data: VisibilityData) -> Tuple[
        CalibrationParams, VisibilityData]:
        """
        Performs averaging, calibration, and subtraction.

        Args:
            key: PRNGKey
            init_params: initial starting point of solve
            lmn: [num_source, 3] lmn coords
            visibility_coords: the visibilities coordinates
            data: the visibilities data

        Returns:
            the new solved params, and subtracted data
        """

        freqs = quantity_to_jnp(self.ms.meta.freqs)
        image = self.discrete_sky_model.get_source_model_linear(
            freqs=freqs
        )  # [num_source, num_freqs, 2, 2]

        for calibrator_source_model in self.calibrator_source_models:
            if isinstance(calibrator_source_model, FitsStokesISourceModel):
                faint_predict = FFTStokesIPredict(
                    convention=self.convention
                )
                image = calibrator_source_model.get_source_model_linear(
                    freqs=freqs
                )
                gains = self.beam_gain_model.compute_gain(
                    freqs=freqs,
                    sources=calibrator_source_model.coords_icrs,
                    phase_tracking=self.ms.meta.phase_tracking,
                    array_location=self.ms.meta.array_location,
                    time=time
                )  # (source_shape) + [num_ant, num_freq, 2, 2]
                faint_model_data = FFTStokesIModelData(
                    image=image,  # [chan, Nx, Ny] in Stokes I
                    gains=...,  # [time, ant, chan, 2, 2]
                    l0=...,  # [chan]
                    m0=...,  # [chan]
                    dl=...,  # [chan]
                    dm=...,  # [chan]
                )
                vis_source = faint_predict.predict(
                    freqs=freqs,
                    faint_model_data=faint_model_data,
                    visibility_coords=visibility_coords
                )

        calibration_data = CalibrationData(
            visibility_coords=visibility_coords,
            image=image,
            lmn=lmn,
            freqs=freqs,
            obs_vis=data.vis,
            obs_vis_weight=data.weights
        )

        # TODO: Apply UV cutoff to ignore galactic plane

        params, _ = self.calibration.solve(
            init_params=init_params,
            data=calibration_data
        )
        return params

    @partial(jax.jit, static_argnums=(0,))
    def _simulate_jax(self, key, dft_model_data: PointModelData, visibility_coords: VisibilityCoords):
        visibilities = self.dft_predict.predict(
            freqs=quantity_to_jnp(self.ms.meta.freqs),
            dft_model_data=dft_model_data,
            visibility_coords=visibility_coords
        )
        visibilities = jnp.reshape(visibilities, np.shape(visibilities)[:-2] + (4,))

        chan_width_hz = quantity_to_jnp(self.ms.meta.channel_width)
        noise_scale = calc_baseline_noise(
            system_equivalent_flux_density=quantity_to_jnp(
                self.ms.meta.system_equivalent_flux_density, 'Jy'),
            chan_width_hz=chan_width_hz,
            t_int_s=quantity_to_jnp(self.ms.meta.integration_time, 's'),
        )

        # TODO: Simulation RFI

        # TODO: Simulate faint sky with FFT

        # Simulate measurement noise

        key1, key2 = jax.random.split(key)
        # Divide by sqrt(2) to account for polarizations
        noise = (noise_scale / jnp.sqrt(2.)) * (
                jax.random.normal(key1, visibilities.shape) + 1j * jax.random.normal(key2, visibilities.shape)
        )

        visibilities += noise

        weights = jnp.full(visibilities.shape, 1. / noise_scale ** 2)
        return VisibilityData(vis=visibilities, weights=weights)

    def _image_visibilities(self, image_name: str, ms: MeasurementSet) -> ImageModel:
        gen = ms.create_block_generator(vis=True, weights=False, flags=True)
        gen_response = None
        uvw = []
        vis = []
        flags = []

        while True:
            try:
                time, visibility_coords, data = gen.send(gen_response)
            except StopIteration:
                break
            uvw.append(visibility_coords.uvw)
            vis.append(data.vis)
            flags.append(data.flags)

        uvw = jnp.concatenate(uvw, axis=0)  # [num_rows, 3]
        vis = jnp.concatenate(vis, axis=0)  # [num_rows, chan, 4]
        flags = jnp.concatenate(flags, axis=0)  # [num_rows, chan]
        freqs = quantity_to_jnp(self.ms.meta.freqs)

        # Get the maximum baseline length
        max_baseline = np.max(np.linalg.norm(uvw, axis=-1))
        # minimum wavelength
        min_wavelength = quantity_to_np(np.min(au.c / self.ms.meta.freqs))
        # Number of pixels
        diffraction_limit_resolution = 1.22 * min_wavelength / max_baseline
        num_pixel = find_optimal_fft_size(
            int(self.oversample_factor * self.field_of_view.to('rad').value / diffraction_limit_resolution)
        )
        lon_top, lat_top = offset_by(
            lon=self.ms.meta.phase_tracking.ra,
            lat=self.ms.meta.phase_tracking.dec,
            posang=0 * au.deg,  # North
            distance=self.field_of_view / 2.
        )
        source_top = ac.ICRS(lon_top, lat_top)

        lon_east, lat_east = offset_by(
            lon=self.ms.meta.phase_tracking.ra,
            lat=self.ms.meta.phase_tracking.dec,
            posang=90 * au.deg,  # East -- increasing RA
            distance=self.field_of_view / 2.
        )
        source_east = ac.ICRS(lon_east, lat_east)

        source_centre = ac.ICRS(self.ms.meta.phase_tracking.ra, self.ms.meta.phase_tracking.dec)

        lmn_ref_points = icrs_to_lmn(
            sources=ac.concatenate([source_centre, source_top, source_east]).transform_to(ac.ICRS),
            time=self.ms.meta.times[0],
            phase_tracking=self.ms.meta.phase_tracking
        )
        dl = (lmn_ref_points[2, 0] - lmn_ref_points[0, 0]) / (num_pixel / 2.)
        dm = (lmn_ref_points[1, 1] - lmn_ref_points[0, 1]) / (num_pixel / 2.)

        # TODO: check that x==l axis here, as it is -y in antenna frame
        center_x = lmn_ref_points[0, 0]
        center_y = lmn_ref_points[0, 1]

        print(f"Center x: {center_x}, Center y: {center_y}")
        print(f"Image size: {num_pixel} x {num_pixel}")
        print(f"Pixel size: {dl} x {dm}")

        dirty_image = self._image_visibilties_jax(
            uvw=uvw,
            vis=vis,
            flags=flags,
            freqs=freqs,
            num_pixel=num_pixel,
            dl=quantity_to_jnp(dl),
            dm=quantity_to_jnp(dm),
            center_l=quantity_to_jnp(center_x),
            center_m=quantity_to_jnp(center_y)
        )  # [num_pixel, num_pixel]
        image_model = ImageModel(
            phase_tracking=self.ms.meta.phase_tracking,
            obs_time=self.ms.meta.times[0],
            dl=dl,
            dm=dm,
            freqs=np.mean(self.ms.meta.freqs, keepdims=True),
            coherencies=['I'],
            beam_major=diffraction_limit_resolution * au.rad,
            beam_minor=diffraction_limit_resolution * au.rad,
            beam_pa=0 * au.deg,
            unit='JY/PIXEL',  # TODO: check this is correct.
            object_name='forward_model',
            image=au.Quantity(np.asarray(dirty_image)[:, :, None, None], 'Jy')
        )
        with open(f"{image_name}.json", 'w') as fp:
            fp.write(image_model.json(indent=2))
        image_model.save_image_to_fits(f"{image_name}.fits", overwrite=False)
        return image_model

    @partial(jax.jit, static_argnames=['self', 'num_pixel'])
    def _image_visibilties_jax(self, uvw: jax.Array, vis: jax.Array, flags: jax.Array, freqs: jax.Array, num_pixel: int,
                               dl: jax.Array, dm: jax.Array, center_l: jax.Array, center_m: jax.Array):
        """
        Multi-channel synthesis image stokes I using a simple w-gridding algorithm.

        Args:
            uvw: [num_rows, 3]
            vis: [num_rows, num_chan, 4] in linear, i.e. [XX, XY, YX, YY]
            flags: [num_rows, num_chan]
            freqs: [num_chan]
            num_pixel: int
            dl: dl pixel size
            dm: dm pixel size
            center_l: centre l
            center_m: centre m

        Returns:
            dirty_image: [num_pixel, num_pixel]
        """

        # Convert linear visibilities to stokes, and create stokes I image
        def _transform_to_stokes_I(vis):
            num_rows, num_chan, _ = np.shape(vis)
            vis = lax.reshape(vis, (num_rows * num_chan, 4))

            def _single(_vis):
                return linear_to_stokes(_vis, flat_output=True)[0]

            vis_I = jax.vmap(_single)(vis)  # [num_rows * num_chan]
            return lax.reshape(vis_I, (num_rows, num_chan))

        vis_I = _transform_to_stokes_I(vis)  # [num_rows, num_chan]

        dirty_image = wgridder.vis2dirty(
            uvw=uvw,
            freqs=freqs,
            vis=vis_I,
            npix_x=num_pixel,
            npix_y=num_pixel,
            pixsize_x=dl,
            pixsize_y=dm,
            center_x=center_l,
            center_y=center_m,
            epsilon=self.epsilon,
            do_wgridding=True,
            mask=flags,
            divide_by_n=True,
            verbosity=0
        )  # [num_pixel, num_pixel]

        return dirty_image
