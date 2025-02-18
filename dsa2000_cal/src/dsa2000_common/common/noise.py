import jax
import jax.numpy as jnp


def calc_baseline_noise(system_equivalent_flux_density: float | jax.Array, chan_width_hz: float | jax.Array,
                        t_int_s: float | jax.Array) -> jax.Array:
    """Calculate the per visibility rms for identical antennas.

    Args:
        system_equivalent_flux_density (float): System Equivalent Flux Density (SEFD) per antennas in Jy
            (already includes efficiency)
        chan_width_hz (float): Channel width in Hz.
        t_int_s (float): Accumulation time in seconds.

    Returns:
        float: noise standard devation per part visibility.
    """
    # The 2 is for number of polarizations.
    return system_equivalent_flux_density / jnp.sqrt(chan_width_hz * t_int_s)


def calc_image_noise(system_equivalent_flux_density: float, bandwidth_hz: float, t_int_s: float, num_antennas: int,
                     flag_frac: float, num_pol: int = 1) -> jax.Array:
    """
    Calculate the image noise for the central pixel.
    
    Args:
        system_equivalent_flux_density: the system equivalent flux density in Jy (already includes efficiency)
        bandwidth_hz: the bandwidth in Hz
        t_int_s: the integration time in seconds
        num_antennas: the number of antennas
        flag_frac: the fraction of flagged visibilities

    Returns:
        the image noise in Jy
    """
    # central pixel = (sum_b V_b cos(0)) / N_b, with V_b = 1 ==> var(central pixel) = sum_b var(V_b) / N_b
    num_baselines = (1. - flag_frac) * num_pol * num_antennas * (num_antennas - 1) / 2.
    return calc_baseline_noise(
        system_equivalent_flux_density=system_equivalent_flux_density,
        chan_width_hz=bandwidth_hz,
        t_int_s=t_int_s) / jnp.sqrt(2 * num_baselines)  # 2 is for real component only
