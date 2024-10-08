import numpy as np
from astropy import units as au, time as at

from dsa2000_cal.assets.content_registry import fill_registries, NoMatchFound
from dsa2000_cal.assets.registries import array_registry
from dsa2000_cal.gain_models.base_spherical_interpolator import BaseSphericalInterpolatorGainModel, \
    build_spherical_interpolator


def build_beam_gain_model(
        array_name: str,
        full_stokes: bool = True,
        model_times: at.Time | None = None
) -> BaseSphericalInterpolatorGainModel:
    """
    Build a beam gain model for an array.

    Args:
        array_name: the name of the array
        full_stokes: whether to use full stokes
        model_times: the times at which to compute the model

    Returns:
        A beam gain model using the spherical interpolator.
    """
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

    if len(model_times) == 1:
        gains = au.Quantity(gains[None, ...])  # [num_times, num_dir, num_freqs, 2, 2]
    else:
        gains = au.Quantity(
            np.repeat(gains[None, ...], len(model_times), axis=0)
        )  # [num_times, num_dir, num_freqs, 2, 2]

    if not full_stokes:
        # TODO: may need to convert basis. Assume linear for now.
        gains = gains[..., 0, 0]

    return build_spherical_interpolator(
        antennas=antennas,
        model_freqs=model_freqs,
        model_times=model_times,
        model_theta=model_theta,
        model_phi=model_phi,
        model_gains=gains,
        ref_time=model_times[0],
        tile_antennas=True
    )
