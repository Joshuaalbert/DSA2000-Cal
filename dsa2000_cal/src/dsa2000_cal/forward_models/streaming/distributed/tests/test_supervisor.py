import asyncio

import pytest
import ray

from dsa2000_cal.forward_models.streaming.distributed.supervisor import Supervisor, SupervisorParams, create_supervisor


@pytest.mark.asyncio
async def test_run_supervisor():
    ray.init(address='local')

    @ray.remote
    class MockActor:

        async def __call__(self, x):
            await asyncio.sleep(0.1)
            print(x)
            return x

    supervisor = create_supervisor(MockActor, 'test', 10)

    async def assert_(x):
        task = asyncio.create_task(supervisor(x))
        await asyncio.sleep(0.1)
        print(f"{await supervisor.num_running()} running, {await supervisor.num_available()} available")
        assert await supervisor.num_running() <= 10
        assert await supervisor.num_available() >= 0
        return await task

    results = await asyncio.gather(*[assert_(a) for a in range(100)])

    second_supervisor = create_supervisor(MockActor, 'other_test', 10)
