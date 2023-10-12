import os
from timeit import default_timer

import pytest
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=6"

from jax import random, numpy as jnp

from dsa2000_cal.dft import im_to_vis


# Parameterizing the test with different environment variables
@pytest.mark.parametrize('chunksize', [1, 2, 3, 4, 5, 6])
def test_im_to_vis_random(chunksize: int):
    # Test im_to_vis for random input values

    # Create a random sky model
    num_rows = 2000 * 2000
    num_sources = 10
    num_channels = 5
    num_corrs = 4
    image = random.normal(random.PRNGKey(42), (num_sources, num_channels, num_corrs))
    lm = 0.01 * random.normal(random.PRNGKey(42), (num_sources, 2))
    uvw = random.normal(random.PRNGKey(42), (num_rows, 3))
    frequency = jnp.linspace(1e3, 2e3, num_channels)
    vis = im_to_vis(image, uvw, lm, frequency, chunksize=chunksize)
    vis.block_until_ready()
    t0 = default_timer()
    vis = im_to_vis(image, uvw, lm, frequency, chunksize=chunksize)
    vis.block_until_ready()
    print(f"im_to_vis took {default_timer() - t0} seconds with {chunksize} chunksize")
    assert vis.shape == (num_rows, num_channels, num_corrs)
