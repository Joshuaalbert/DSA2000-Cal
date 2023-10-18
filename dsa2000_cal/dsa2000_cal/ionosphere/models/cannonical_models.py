from typing import Literal

from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from tomographic_kernel.tomographic_kernel import MultiLayerTomographicKernel

SPECIFICATION = Literal['simple', 'light_dawn', 'dawn', 'dusk', 'dawn_challenge', 'dusk_challenge']


def build_ionosphere_tomographic_kernel(x0: jnp.ndarray, earth_centre: jnp.ndarray,
                                        specification: SPECIFICATION,
                                        S_marg: int = 25, compute_tec: bool = False,
                                        northern_hemisphere: bool = True) -> MultiLayerTomographicKernel:
    """
    Build an ionospheric tomographic kernel against a standard specification.

    Args:
        x0: The coordinates of a reference point in the array, against ionosphere height is measured, and
            differential TEC is evaluated. This can be seen as a reference antenna.
        earth_centre: The coordinates of the Earth centre in ENU frame.
        specification: Literal['simple', 'light_dawn', 'dawn', 'dusk', 'dawn_challenge', 'dusk_challenge']
        S_marg: resolution of quadrature
        compute_tec: if true then only compute TEC, not differential TEC.
        northern_hemisphere: Set to true if in northern hemisphere.

    Returns:
        a MultiLayerTomographicKernel of given specification
    """
    if specification == 'simple':  # E, F layers
        bottoms = [150.]  # km
        widths = [100.]  # km
        fed_mus = [1e11 * 1e-10]  # 1E10 [electron / m^3]
        fed_sigmas = [fed_mus[0] * 0.1]  # 1E10 [electron / m^3]
        fed_ls = [5.]  # km

        fed_kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=fed_sigma, length_scale=fed_l)
                       for fed_sigma, fed_l in zip(fed_sigmas, fed_ls)]
        if northern_hemisphere:
            wind_velocities = [jnp.asarray([-0.200, 0.030, 0.])]
        else:
            wind_velocities = [jnp.asarray([-0.200, -0.030, 0.])]
    elif specification == 'light_dawn':  # E, F layers
        bottoms = [90., 250.]  # km
        widths = [10., 100.]  # km
        fed_mus = [1e10 * 1e-10, 1e11 * 1e-10]  # 1E10 [electron / m^3]
        fed_sigmas = [fed_mus[0] * 0.1, fed_mus[1] * 0.1]  # 1E10 [electron / m^3]
        fed_ls = [0.5, 5.]  # km

        fed_kernels = [tfp.math.psd_kernels.MaternThreeHalves(amplitude=fed_sigma, length_scale=fed_l)
                       for fed_sigma, fed_l in zip(fed_sigmas, fed_ls)]
        if northern_hemisphere:
            wind_velocities = [jnp.asarray([-0.200, 0.030, 0.]), jnp.asarray([-0.300, 0.030, 0.])]
        else:
            wind_velocities = [jnp.asarray([-0.200, -0.030, 0.]), jnp.asarray([-0.300, -0.030, 0.])]
    elif specification == 'dawn':  # E, F layers
        bottoms = [90., 150.]  # km
        widths = [10., 100.]  # km
        fed_mus = [1e10 * 1e-10, 1e11 * 1e-10]  # 1E10 [electron / m^3]
        fed_sigmas = [fed_mus[0] * 0.1, fed_mus[1] * 0.2]  # 1E10 [electron / m^3]
        fed_ls = [0.5, 5.]  # km

        fed_kernels = [tfp.math.psd_kernels.MaternThreeHalves(amplitude=fed_sigma, length_scale=fed_l)
                       for fed_sigma, fed_l in zip(fed_sigmas, fed_ls)]
        if northern_hemisphere:
            wind_velocities = [jnp.asarray([-0.200, 0.030, 0.]), jnp.asarray([-0.300, 0.030, 0.])]
        else:
            wind_velocities = [jnp.asarray([-0.200, -0.030, 0.]), jnp.asarray([-0.300, -0.030, 0.])]
    elif specification == 'dawn_challenge':  # E, F layers
        bottoms = [100., 250.]  # km
        widths = [20., 250.]  # km
        fed_mus = [5e10 * 1e-10, 5e11 * 1e-10]  # 1E10 [electron / m^3]
        fed_sigmas = [fed_mus[0] * 0.2, fed_mus[1] * 0.5]  # 1E10 [electron / m^3]
        fed_ls = [0.3, 5.]  # km

        fed_kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=fed_sigma, length_scale=fed_l)
                       for fed_sigma, fed_l in zip(fed_sigmas, fed_ls)]
        if northern_hemisphere:
            wind_velocities = [jnp.asarray([-0.200, 0.030, 0.]), jnp.asarray([-0.300, 0.030, 0.])]
        else:
            wind_velocities = [jnp.asarray([-0.200, -0.030, 0.]), jnp.asarray([-0.300, -0.030, 0.])]
    elif specification == 'dusk':  # E, F layers
        bottoms = [90., 150.]  # km
        widths = [10., 100.]  # km
        fed_mus = [1e10 * 1e-10, 1e11 * 1e-10]  # 1E10 [electron / m^3]
        fed_sigmas = [fed_mus[0] * 0.1, fed_mus[1] * 0.2]  # 1E10 [electron / m^3]
        fed_ls = [0.5, 10.]  # km

        fed_kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=fed_sigma, length_scale=fed_l)
                       for fed_sigma, fed_l in zip(fed_sigmas, fed_ls)]
        if northern_hemisphere:
            wind_velocities = [jnp.asarray([-0.050, 0.030, 0.]), jnp.asarray([-0.100, 0.030, 0.])]
        else:
            wind_velocities = [jnp.asarray([-0.050, -0.030, 0.]), jnp.asarray([-0.100, -0.030, 0.])]
    elif specification == 'dusk_challenge':  # E, F layers
        bottoms = [100., 250.]  # km
        widths = [20., 250.]  # km
        fed_mus = [3e10 * 1e-10, 5e11 * 1e-10]  # 1E10 [electron / m^3]
        fed_sigmas = [fed_mus[0] * 0.2, fed_mus[1] * 0.5]  # 1E10 [electron / m^3]
        fed_ls = [0.3, 5.]  # km

        fed_kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=fed_sigma, length_scale=fed_l)
                       for fed_sigma, fed_l in zip(fed_sigmas, fed_ls)]
        if northern_hemisphere:
            wind_velocities = [jnp.asarray([-0.200, 0.030, 0.]), jnp.asarray([-0.300, 0.030, 0.])]
        else:
            wind_velocities = [jnp.asarray([-0.200, -0.030, 0.]), jnp.asarray([-0.300, -0.030, 0.])]
    else:
        raise ValueError(f"Invalid spec: {specification}")

    print("Ionosphere parameters:")
    for layer, (bottom, width, l, fed_mu, fed_sigma, wind_velocity) in enumerate(
            zip(bottoms, widths, fed_ls, fed_mus, fed_sigmas, wind_velocities)):
        print(f"Layer {layer}:\n"
              f"\tbottom={bottom} km\n"
              f"\twidth={width} km\n"
              f"\tlengthscale={l} km\n"
              f"\twind velocity={wind_velocity}km/s\n"
              f"\tfed_mu={fed_mu} mTECU/km\n"
              f"\tfed_sigma={fed_sigma} mTECU/km")

    return MultiLayerTomographicKernel(x0=x0,
                                       earth_centre=earth_centre,
                                       width=widths,
                                       bottom=bottoms,
                                       wind_velocity=wind_velocities,
                                       fed_mu=fed_mus,
                                       fed_kernel=fed_kernels,
                                       compute_tec=compute_tec,
                                       S_marg=S_marg)
