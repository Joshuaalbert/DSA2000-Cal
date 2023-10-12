import os
from timeit import default_timer

import jax
import pytest

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=6"

from jax import random, numpy as jnp

from dsa2000_cal.dft import im_to_vis, im_to_vis_with_gains


# Parameterizing the test with different environment variables
@pytest.mark.parametrize('chunksize', list(range(1, len(jax.devices()) + 1)))
def test_im_to_vis_random(chunksize: int):
    # Test im_to_vis for random input values

    num_sources = 10
    num_channels = 5
    num_ant = 7
    num_times = 3
    num_corrs = 4
    num_rows = (num_ant * num_ant // 2) * num_times  # with autocorrs
    image = random.normal(random.PRNGKey(42), (num_sources, num_channels, num_corrs))
    lm = 0.01 * random.normal(random.PRNGKey(42), (num_sources, 2))
    uvw = random.normal(random.PRNGKey(42), (num_rows, 3))
    frequency = jnp.linspace(1e3, 2e3, num_channels)
    vis = im_to_vis(image, uvw, lm, frequency, chunksize=chunksize)
    vis.block_until_ready()
    t0 = default_timer()
    vis = im_to_vis(image, uvw, lm, frequency, chunksize=chunksize)
    vis.block_until_ready()
    print(f"im_to_vis_with_gains took {default_timer() - t0} seconds with {chunksize} chunksize")
    assert vis.shape == (num_rows, num_channels, num_corrs)


@pytest.mark.parametrize('chunksize', list(range(1, len(jax.devices()) + 1)))
def test_im_to_vis_with_gains_random(chunksize: int):
    # Test im_to_vis for random input values

    # Create a random sky model
    num_sources = 10
    num_channels = 5
    num_ant = 7
    num_times = 3
    num_rows = (num_ant * num_ant // 2) * num_times  # with autocorrs
    image = random.normal(random.PRNGKey(42), (num_sources, num_channels, 2, 2))
    gains = random.normal(random.PRNGKey(42), (num_times, num_ant, num_sources, num_channels, 2, 2))
    antenna_1 = random.randint(random.PRNGKey(42), (num_rows,), 0, num_ant)
    antenna_2 = random.randint(random.PRNGKey(42), (num_rows,), 0, num_ant)
    time_idx = random.randint(random.PRNGKey(42), (num_rows,), 0, num_times)
    lm = 0.01 * random.normal(random.PRNGKey(42), (num_sources, 2))
    uvw = random.normal(random.PRNGKey(42), (num_rows, 3))
    frequency = jnp.linspace(1e3, 2e3, num_channels)
    vis = im_to_vis_with_gains(image, gains, antenna_1, antenna_2, time_idx, uvw, lm, frequency, chunksize=chunksize)
    vis.block_until_ready()
    t0 = default_timer()
    vis = im_to_vis_with_gains(image, gains, antenna_1, antenna_2, time_idx, uvw, lm, frequency, chunksize=chunksize)
    vis.block_until_ready()
    print(f"im_to_vis_with_gains took {default_timer() - t0} seconds with {chunksize} chunksize")
    assert vis.shape == (num_rows, num_channels, 2, 2)
