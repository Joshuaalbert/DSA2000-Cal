"""
Domain: Gain Models
SimulateIonosphere :: () -> GainModel
SimulateDishDefects :: () -> GainModel

Domain: Predict Visibilities
DFTPredict :: (DiscreteSkyModel, GainModel) -> Visibilties
FFTPredict :: (SkyImage, GainModel) -> Visibilties
RFIPredict :: (EmitterModel, GainModel) -> Visibilities

Domain: Process Visibilties
Flag :: (Visibilities) -> Visibilities
Average :: (Visibilties) -> Visibilities

Domain: Calibration
Calibrate :: (Visibilties, DiscreteSkyModel, GainModel) -> GainModel
Subtract :: (Visibilties, DiscreteSkyModel, GainModel) -> Visibilties

Domain: Imaging
Image :: (Visibilties, GainModel) -> SkyImage
"""
import dataclasses
from typing import NamedTuple, Any

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
import jax
import jax.numpy as jnp

from dsa2000_cal.models.run_config import RunConfig
from dsa2000_cal.src.common.vec_ops import VisibilityCoords
from dsa2000_cal.src.dft_predict.op import DFTPredict, DFTModelData
from dsa2000_cal.src.fft_predict.op import FFTPredict, FFTModelData
from dsa2000_cal.src.rfi_predict.op import RFIPredict, RFIModelData


class DiscreteCoords(NamedTuple):
    time_mjs: jax.Array
    antennas_enu: jax.Array
    pointing_enu: jax.Array
    freq_hz: jax.Array


class GridCoords(NamedTuple):
    time_mjs: jax.Array
    antennas_enu: jax.Array
    pointing_enu: jax.Array
    uvw: jax.Array

class EmitterCoords(NamedTuple):
    time_mjs: jax.Array
    antennas_enu: jax.Array
    pointing_enu: jax.Array
    freq_hz: jax.Array
    uvw: jax.Array


@dataclasses.dataclass(eq=False)
class DiscreteSkyModel:
    def get_coords(self, time: at.Time, antennas: ac.EarthLocation, pointing: ac.ICRS, freq:au.Quantity) -> DiscreteCoords:
        ...

    def get_image(self, freq) -> jax.Array:
        ...

    def get_lm(self) -> jax.Array:
        ...


@dataclasses.dataclass(eq=False)
class FaintSkyModel:
    def get_coords(self, time: at.Time, antennas: ac.EarthLocation, pointing: ac.ICRS, freq:au.Quantity) -> GridCoords:
        ...

    def get_image(self, freq) -> jax.Array:
        ...

    def get_lm(self) -> jax.Array:
        ...

@dataclasses.dataclass
class RFIEmitterModel:
    def get_coords(self, time: at.Time, antennas: ac.EarthLocation, pointing: ac.ICRS, freq:au.Quantity) -> EmitterCoords:
        ...

    def get_image(self, freq) -> jax.Array:
        ...

    def get_lm(self) -> jax.Array:
        ...



class GainModel:
    def predict_discrete(self, state: Any, discrete_coords: DiscreteCoords) -> jax.Array:
        ...

    def predict_grid(self, state: Any, grid_coords: GridCoords) -> jax.Array:
        ...

@dataclasses.dataclass
class BeamModel(GainModel):
    dtype: jnp.dtype = jnp.complex64
    chunksize: int = 1
    unroll: int = 1

    def simulate(self):
        ...

    def predict_discrete(self, state: Any, discrete_coords: DiscreteCoords) -> jax.Array:
        ...

    def predict_grid(self, state: Any, grid_coords: GridCoords) -> jax.Array:
        ...


@dataclasses.dataclass(eq=False)
class IonosphereGainModel(GainModel):
    dtype: jnp.dtype = jnp.complex64
    chunksize: int = 1
    unroll: int = 1

    def simulate(self):
        ...

    def predict_discrete(self, state: Any, discrete_coords: DiscreteCoords) -> jax.Array:
        ...

    def predict_grid(self, state: Any, grid_coords: GridCoords) -> jax.Array:
        ...


@dataclasses.dataclass(eq=False)
class DishDefectGainModel(GainModel):
    dtype: jnp.dtype = jnp.complex64
    chunksize: int = 1
    unroll: int = 1

    def simulate(self):
        ...

    def predict_discrete(self, state: Any, discrete_coords: DiscreteCoords) -> jax.Array:
        ...

@dataclasses.dataclass(eq=False)
class NoiseModel:
    noise_sigma: jnp.Array
    def add_noise(self, key: jax.random.PRNGKeyArray, visibilities: jnp.Array) -> jnp.Array:
        ...

@dataclasses.dataclass(eq=False)
class FullModel:
    """
    Full end-to-end model
    """
    beam_model: BeamModel | None
    ionosphere_model: IonosphereGainModel | None
    dish_model: DishDefectGainModel | None
    discrete_sky_model: DiscreteSkyModel | None
    faint_sky_model: FaintSkyModel | None
    rfi_emitter_model: RFIEmitterModel | None
    dft_predict: DFTPredict
    fft_predict: FFTPredict
    rfi_predict: RFIPredict
    noise_model: NoiseModel | None
    dtype = jnp.complex64

    @property
    def do_beam(self):
        return self.beam_model is not None

    @property
    def do_ionosphere(self):
        return self.ionosphere_model is not None

    @property
    def do_dish_defects(self):
        return self.dish_model is not None

    @property
    def do_discrete_sky(self):
        return self.discrete_sky_model is not None

    @property
    def do_faint_sky(self):
        return self.faint_sky_model is not None

    @property
    def do_rfi(self):
        return self.rfi_emitter_model is not None

    @property
    def do_noise(self):
        return self.noise_model is not None


    def predict(self, key: jax.random.PRNGKeyArray, time: at.Time, antennas: ac.EarthLocation, pointing: ac.ICRS, freq: au.Quantity,
                visibility_coords: VisibilityCoords, beam_state: Any, ionosphere_state: Any, dish_state: Any):
        # Predict visibilities
        visibilities = jnp.asarray(0., dtype=self.dtype)

        if self.do_discrete_sky:

            discrete_coords = self.discrete_sky_model.get_coords(time, antennas, pointing, freq)

            beam_gains_discrete = self.beam_model.predict_discrete(
                state=beam_state,
                discrete_coords=discrete_coords
            )

            ionosphere_gains_discrete = self.ionosphere_model.predict_discrete(
                state=ionosphere_state,
                discrete_coords=discrete_coords
            )  # [source, time, ant, chan, 2, 2]

            dish_gains_discrete = self.dish_model.predict_discrete(
                state=dish_state,
                discrete_coords=discrete_coords
            )  # [source, time, ant, chan, 2, 2]


            # Take product of gains on last dimension
            # [source, time, ant, chan, 2, 2] * [source, time, ant, chan, 2, 2] -> [source, time, ant, chan, 2, 2]
            dot = lambda x, y, z: x @ y @ z
            dot = jax.vmap(dot, in_axes=[3, 3, 3]) # vectorise over channel
            dot = jax.vmap(dot, in_axes=[2, 2, 2]) # vectorise over antenna
            dot = jax.vmap(dot, in_axes=[1, 1, 1]) # vectorise over time
            dot = jax.vmap(dot, in_axes=[0, 0, 0]) # vectorise over source

            gains_discrete = dot(ionosphere_gains_discrete, dish_gains_discrete, beam_gains_discrete)

            discrete_model_data = DFTModelData(
                image=self.discrete_sky_model.get_image(freq),
                gains=gains_discrete,
                lm=self.discrete_sky_model.get_lm()
            )
            dft_visibilties = self.dft_predict.predict(
                model_data=discrete_model_data,
                visibility_coords=visibility_coords,
                freq=freq
            )  # [row, chan, 2, 2]

            visibilities += dft_visibilties

        if self.do_faint_sky:

            grid_coords = self.faint_sky_model.get_coords(time, antennas, pointing, freq)

            beam_gains_grid = self.beam_model.predict_grid(
                state=beam_state,
                grid_coords=grid_coords
            ) # [Nl, Nm, time, ant, chan, 2, 2]

            ionosphere_gains_grid = self.ionosphere_model.predict_grid(
                state=ionosphere_state,
                grid_coords=grid_coords
            )  # [Nl, Nm, time, ant, chan, 2, 2]

            dish_gains_grid = self.dish_model.predict_grid(
                state=dish_state,
                grid_coords=grid_coords
            )  # [Nl, Nm, time, ant, chan, 2, 2]

            # Take product of gains on last dimension
            # [source, time, ant, chan, 2, 2] * [source, time, ant, chan, 2, 2] -> [source, time, ant, chan, 2, 2]
            dot = lambda x, y, z: x @ y @ z
            dot = jax.vmap(dot, in_axes=[4, 4, 4]) # vectorise over channel
            dot = jax.vmap(dot, in_axes=[3, 3, 3]) # vectorise over antenna
            dot = jax.vmap(dot, in_axes=[2, 2, 2]) # vectorise over time
            dot = jax.vmap(dot, in_axes=[1, 1, 1]) # vectorise over Nm
            dot = jax.vmap(dot, in_axes=[0, 0, 0]) # vectorise over Nl

            gains_grid = dot(ionosphere_gains_grid, dish_gains_grid, beam_gains_grid) # [Nl, Nm, time, ant, chan, 2, 2]

            faint_model_data = FFTModelData(
                image=self.faint_sky_model.get_image(freq),
                image_lm_rad=self.faint_sky_model.get_lm(),
                gains=gains_grid
            )
            fft_visibilties = self.fft_predict.predict(
                model_data=faint_model_data,
                visibility_coords=visibility_coords,
                freq=freq
            )

            visibilities += fft_visibilties

        if self.do_rfi:
            rfi_model_data = RFIModelData(
                image=self.rfi_emitter_model.get_image(freq),
                gains=self.rfi_emitter_model.get_coords(time, antennas, pointing, freq)
            )

            rfi_visibilities = self.rfi_predict.predict(
                model_data=rfi_model_data,
                visibility_coords=visibility_coords,
                freq=freq
            )

            visibilities += rfi_visibilities

        if self.do_noise:
            visibilities = self.noise_model.add_noise(key, visibilities)

        return visibilities

    def run(self, run_config: RunConfig):
        """
        Run the full model.
        """
        if self.do_beam:
            beam_state = self.beam_model.simulate()
        else:
            beam_state = None
        if self.do_ionosphere:
            ionosphere_state = self.ionosphere_model.simulate()
        else:
            ionosphere_state = None
        if self.do_dish_defects:
            dish_state = self.dish_model.simulate()
        else:
            dish_state = None


        # TODO: shard predict visibilities

        # Load the visibility coordinates, and predict visibilities

        self.predict(
            time=run_config.start_time,
            antennas=run_config.antennas,
            pointing=run_config.pointing,
            freq=run_config.start_freq_hz,
            visibility_coords=run_config.visibility_coords,
            beam_state=beam_state,
            ionosphere_state=ionosphere_state,
            dish_state=dish_state
        )

        # flag visibilities

        # average visibilities

        # calibrate visibilities

        # subtract bright source

        # image visibilities
        ...
