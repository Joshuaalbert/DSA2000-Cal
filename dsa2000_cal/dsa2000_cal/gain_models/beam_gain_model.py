import dataclasses

import numpy as np
from astropy import units as au, time as at

from dsa2000_cal.assets.content_registry import fill_registries, NoMatchFound
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.gain_models.spherical_interpolator import SphericalInterpolatorGainModel


@dataclasses.dataclass(eq=False)
class BeamGainModel(SphericalInterpolatorGainModel):
    ...


def beam_gain_model_factory(array_name: str, **kwargs) -> BeamGainModel:
    fill_registries()
    try:
        array = array_registry.get_instance(array_registry.get_match(array_name))
    except NoMatchFound as e:
        raise ValueError(f"Array {array_name} not found in registry. Add it to use the BeamGainModel factory.") from e

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

    # Mock, else compute over time in observation
    model_times = at.Time.now().reshape((1,))

    phase = dish_model.get_phase()  # [num_theta, num_phi, num_freqs, 2, 2]
    gains = amplitude * np.exp(1j * phase.to('rad').value)  # [num_theta, num_phi, num_freqs, 2, 2]
    gains = gains.reshape(
        (len(model_times), len(model_phi), len(model_freqs), 2, 2)
    ) * au.dimensionless_unscaled  # [num_times, num_dir, num_freqs, 2, 2]

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
