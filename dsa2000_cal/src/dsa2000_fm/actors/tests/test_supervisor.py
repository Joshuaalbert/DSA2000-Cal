import asyncio

import pytest
import ray

from dsa2000_fm.actors.supervisor import create_supervisor, Supervisor


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
        assert (await task) == x
        return x

    results = await asyncio.gather(*[assert_(a) for a in range(100)])

    second_supervisor = create_supervisor(MockActor, 'other_test', 10)


@pytest.mark.asyncio
async def test_run_supervisor_stream():
    ray.init(address='local')

    @ray.remote
    class MockActor:

        async def __call__(self, x):
            for i in range(5):
                yield x
                await asyncio.sleep(0.1)

    supervisor: Supervisor[int] = create_supervisor(MockActor, 'test', 10)

    gen = supervisor.stream(1)
    async for ref in gen:
        print(ref)
        assert await ref == 1
