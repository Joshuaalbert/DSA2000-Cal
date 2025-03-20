import asyncio
import logging

import jax.numpy as jnp
import ray

from dsa2000_common.common.logging import dsa_logger as logger


async def main():
    ray.init(address='local', num_gpus=1)

    @ray.remote
    class MockActor:

        async def __call__(self, x):
            # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            return jnp.asarray(x)

    actor = MockActor.options(runtime_env={'CUDA_VISIBLE_DEVICES': "0"}).remote()

    await actor.__call__.remote(1)


if __name__ == '__main__':
    asyncio.run(main())
