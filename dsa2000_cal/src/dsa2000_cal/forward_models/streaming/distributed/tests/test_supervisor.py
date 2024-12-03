import asyncio

import pytest
import ray

from dsa2000_cal.forward_models.streaming.distributed.supervisor import Supervisor, SupervisorParams


@pytest.mark.asyncio
async def test_run_supervisor():
    ray.init(address='local')

    @ray.remote
    class MockActor:

        async def __call__(self, x):
            await asyncio.sleep(0.1)
            print(x)
            return x

    actors = [MockActor.remote() for _ in range(10)]
    supervisor = Supervisor(worker_id='test', params=SupervisorParams(actors=actors, name='mock'))

    async def assert_(x):
        task = asyncio.create_task(supervisor(x))
        await asyncio.sleep(0.1)
        print(f"{await supervisor.num_running()} running, {await supervisor.num_available()} available")
        assert await supervisor.num_running() <= len(actors)
        assert await supervisor.num_available() >= 0
        return await task

    results = await asyncio.gather(*[assert_(a) for a in range(100)])
