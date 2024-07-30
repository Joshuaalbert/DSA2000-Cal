import dataclasses

import numpy as np
from astropy import units as au, time as at

from dsa2000_cal.assets.content_registry import fill_registries, NoMatchFound
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.gain_models.spherical_interpolator import SphericalInterpolatorGainModel
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet


@dataclasses.dataclass(eq=False)
class BeamGainModel(SphericalInterpolatorGainModel):
    ...


def beam_gain_model_factory(ms: MeasurementSet) -> BeamGainModel:
    return build_beam_gain_model(
        array_name=ms.meta.array_name,
        model_times=ms.meta.times,
        full_stokes=ms.is_full_stokes()
    )


def build_beam_gain_model(
        array_name: str,
        full_stokes: bool = True,
        model_times: at.Time | None = None,
        **kwargs
) -> BeamGainModel:
    fill_registries()
    try:
        array = array_registry.get_instance(array_registry.get_match(array_name))
    except NoMatchFound as e:
        raise ValueError(f"Array {array_name} not found in registry. Add it to use the BeamGainModel factory.") from e

    if model_times is None:
        # Mock, else compute over time in observation
        model_times = at.Time.now().reshape((1,))

    dish_model = array.get_antenna_model()

    # Create direction mesh
    model_theta = dish_model.get_theta()
    model_phi = dish_model.get_phi()
    model_theta, model_phi = np.meshgrid(model_theta, model_phi, indexing='ij')
    model_theta = model_theta.reshape((-1,))
    model_phi = model_phi.reshape((-1,))

    antennas = array.get_antennas()
    model_freqs = dish_model.get_freqs()
    amplitude = dish_model.get_amplitude()  # [num_theta, num_phi, num_freqs, 2, 2]
    voltage_gain = dish_model.get_voltage_gain()  # [num_freqs]
    amplitude = au.Quantity(amplitude / voltage_gain[:, None, None])  # [num_theta, num_phi, num_freqs, 2, 2]

    phase = dish_model.get_phase()  # [num_theta, num_phi, num_freqs, 2, 2]
    gains = amplitude * np.exp(1j * phase.to('rad').value)  # [num_theta, num_phi, num_freqs, 2, 2]

    # Right now all beams are assumed stationary over time, so we can simply put the same gain for all times.
    gains = gains.reshape(
        (len(model_phi), len(model_freqs), 2, 2)
    ) * au.dimensionless_unscaled  # [num_dir, num_freqs, 2, 2]

    # Repeat twice to set the same gain for all times
    gains = au.Quantity(np.repeat(gains[None, ...], 2, axis=0))  # [num_times, num_dir, num_freqs, 2, 2]
    # Just to min and max times
    model_times = at.Time([model_times.tt.min(), model_times.tt.max()])

    if not full_stokes:
        gains = gains[..., 0, 0]

    beam_gain_model = BeamGainModel(
        antennas=antennas,
        model_freqs=model_freqs,
        model_theta=model_theta,
        model_phi=model_phi,
        model_times=model_times,
        model_gains=gains,
        tile_antennas=True,
        **kwargs
    )
    return beam_gain_model
