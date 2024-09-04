import itertools
import time

import jax
import pytest
from jax import numpy as jnp

from dsa2000_cal.delay_models.far_field import VisibilityCoords
from dsa2000_cal.visibility_model.source_models.celestial.gaussian_source_model import GaussianPredict, \
    GaussianModelData
from dsa2000_cal.visibility_model.source_models.celestial.point_source_model import PointPredict, PointModelData


@pytest.mark.parametrize("di_gains", [True, False, None])
def test_benchmark_gaussian_performance(di_gains):
    # Benchmark performance
    # di_gains = True Time taken for 2048 antennas, 1 channels, 100 sources: 14.163064 s | 9.324887 s | 8.610048 s | 8.988812 s | 11.194738 s | 9.603148 s
    # di_gains = False Time taken for 2048 antennas, 1 channels, 100 sources: 9.601373 s | 13.285119 s | 11.940015 s | 12.057437 s | 9.140454 s | 9.324197 s
    # di_gains = None Time taken for 2048 antennas, 1 channels, 100 sources: 5.436340 s | 8.067334 s | 6.951573 s | 7.103243 s | 5.576627 s | 4.885742 s
    dft_predict = GaussianPredict(dtype=jnp.complex128)
    num_time = 1
    for num_ant in [2048]:
        for num_chan in [1]:
            for num_source in [100]:
                freqs = jnp.linspace(700e6, 2000e6, num_chan)

                antennas = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (num_ant, 3))
                antenna_1, antenna_2 = jnp.asarray(
                    list(itertools.combinations_with_replacement(range(num_ant), 2))).T

                num_rows = len(antenna_1)

                uvw = antennas[antenna_2] - antennas[antenna_1]
                uvw = uvw.at[:, 2].mul(1e-3)

                times = jnp.arange(num_time) * 1.5
                time_idx = jnp.zeros((num_rows,), jnp.int64)
                time_obs = times[time_idx]

                image = jnp.zeros((num_source, num_chan, 2, 2), dtype=jnp.complex64)
                image = image.at[..., 0, 0].set(0.5)
                image = image.at[..., 1, 1].set(0.5)
                if di_gains is None:
                    gains = None
                else:
                    if di_gains:
                        gain_shape = (num_time, num_ant, num_chan, 2, 2)
                    else:
                        gain_shape = (num_source, num_time, num_ant, num_chan, 2, 2)

                    gains = jnp.ones(gain_shape) + 1j * jnp.zeros(gain_shape)
                    gains = gains.at[..., 1, 0].set(0.)
                    gains = gains.at[..., 0, 1].set(0.)

                lmn = jax.random.normal(jax.random.PRNGKey(42), (num_source, 3))
                lmn /= jnp.linalg.norm(lmn, axis=-1, keepdims=True)

                ellipse_params = jnp.zeros((num_source, 3))

                model_data = GaussianModelData(
                    freqs=freqs,
                    image=image,
                    gains=gains,
                    lmn=lmn,
                    ellipse_params=ellipse_params
                )

                visibility_coords = VisibilityCoords(
                    uvw=uvw,
                    time_obs=time_obs,
                    antenna_1=antenna_1,
                    antenna_2=antenna_2,
                    time_idx=time_idx
                )

                f = jax.jit(dft_predict.predict).lower(model_data=model_data,
                                                       visibility_coords=visibility_coords).compile()

                t0 = time.time()
                visibilities = f(model_data=model_data, visibility_coords=visibility_coords).block_until_ready()
                t1 = time.time()
                print(f"Time taken for {num_ant} antennas, {num_chan} channels, {num_source} sources: {t1 - t0:.6f} s")


@pytest.mark.parametrize("di_gains", [True, False, None])
def test_benchmark_performance_point_sources(di_gains):
    # Benchmark performance
    # di_gains=True: Time taken for 2048 antennas, 1 channels, 100 sources: 9.648704 s | 6.206855 s | 6.150985 s | 7.489174 s | 6.170542 s | 7.400250 s | 6.066451 s
    # di_gains=False: Time taken for 2048 antennas, 1 channels, 100 sources: 5.895023 s | 10.000695 s | 9.863059 s | 5.838677 s | 5.838487 s | 5.802181 s | 5.789162 s
    # di_gains=None: Time taken for 2048 antennas, 1 channels, 100 sources: 2.050352 s | 4.495360 s | 4.380370 s | 1.956469 s | 2.811478 s | 1.797925 s | 1.742006 s
    dft_predict = PointPredict()
    num_time = 1
    for num_ant in [2048]:
        for num_chan in [1]:
            for num_source in [100]:
                freqs = jnp.linspace(700e6, 2000e6, num_chan)

                antennas = 20e3 * jax.random.normal(jax.random.PRNGKey(42), (num_ant, 3))
                antenna_1, antenna_2 = jnp.asarray(
                    list(itertools.combinations_with_replacement(range(num_ant), 2))).T

                num_rows = len(antenna_1)

                uvw = antennas[antenna_2] - antennas[antenna_1]
                uvw = uvw.at[:, 2].mul(1e-3)

                times = jnp.arange(num_time) * 1.5
                time_idx = jnp.zeros((num_rows,), jnp.int64)
                time_obs = times[time_idx]

                image = jnp.zeros((num_source, num_chan, 2, 2), dtype=jnp.complex64)
                image = image.at[..., 0, 0].set(0.5)
                image = image.at[..., 1, 1].set(0.5)
                if di_gains is None:
                    gains = None
                else:
                    if di_gains:
                        gain_shape = (num_time, num_ant, num_chan, 2, 2)
                    else:
                        gain_shape = (num_source, num_time, num_ant, num_chan, 2, 2)

                    gains = jnp.ones(gain_shape) + 1j * jnp.zeros(gain_shape)
                    gains = gains.at[..., 1, 0].set(0.)
                    gains = gains.at[..., 0, 1].set(0.)

                lmn = jax.random.normal(jax.random.PRNGKey(42), (num_source, 3))
                lmn /= jnp.linalg.norm(lmn, axis=-1, keepdims=True)

                model_data = PointModelData(
                    freqs=freqs,
                    image=image,
                    gains=gains,
                    lmn=lmn
                )

                visibility_coords = VisibilityCoords(
                    uvw=uvw,
                    time_obs=time_obs,
                    antenna_1=antenna_1,
                    antenna_2=antenna_2,
                    time_idx=time_idx
                )

                f = jax.jit(dft_predict.predict).lower(model_data=model_data,
                                                       visibility_coords=visibility_coords).compile()

                t0 = time.time()
                visibilities = f(model_data=model_data, visibility_coords=visibility_coords).block_until_ready()
                t1 = time.time()
                print(f"Time taken for {num_ant} antennas, {num_chan} channels, {num_source} sources: {t1 - t0:.6f} s")
