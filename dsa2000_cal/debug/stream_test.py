import time

import ray

import asyncio

@ray.remote
class Actor:
    async def f(self):
        for i in range(5):
            print('Actor', i)
            yield i
            time.sleep(1)
            print('Actor', i, 'done')

@ray.remote
class AsyncActor:
    async def f(self):
        actor = Actor.remote()
        async for ref in actor.f.remote():
            print('yielding ref', ref)
            yield ref

async def main():
    actor = AsyncActor.remote()
    async for ref in actor.f.remote():
        print(ref)
        print(await (ref))
        print(await (await ref))


if __name__ == '__main__':
    ray.init(address='local')
    asyncio.run(main())