import asyncio
import dataclasses
import time
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pylab as plt

RUN_TIME_MODIFIER = 1e-1

usage = dict(
    times=[time.time() / RUN_TIME_MODIFIER],
    cpus=[0],
    mem=[0],
    gpus=[0],
    aggregator=dict(times=[time.time() / RUN_TIME_MODIFIER], count=[0]),
    gridder=dict(times=[time.time() / RUN_TIME_MODIFIER], count=[0]),
    calibrator=dict(times=[time.time() / RUN_TIME_MODIFIER], count=[0]),
    data_streamer=dict(times=[time.time() / RUN_TIME_MODIFIER], count=[0]),
    model_streamer=dict(times=[time.time() / RUN_TIME_MODIFIER], count=[0]),
    dft_predictor=dict(times=[time.time() / RUN_TIME_MODIFIER], count=[0]),
    degridding_predictor=dict(times=[time.time() / RUN_TIME_MODIFIER], count=[0]),
    systematic_simulator=dict(times=[time.time() / RUN_TIME_MODIFIER], count=[0])
)


@dataclasses.dataclass
class Actor:
    name: str
    num_cpus: int
    gpus: float
    mem: float
    run_time: timedelta
    deps: List[Tuple['Supervisor', ...]]

    async def call(self):
        for dep in self.deps:
            tasks = [actor.call() for actor in dep]
            await asyncio.gather(*tasks)
        total_time = self.run_time.total_seconds() * RUN_TIME_MODIFIER * np.random.normal(0.9, 1.1)
        dt = total_time / 10.
        t0 = time.time()
        usage['times'].append(time.time() / RUN_TIME_MODIFIER)
        usage['cpus'].append(usage['cpus'][-1] + self.num_cpus)
        usage['gpus'].append(usage['gpus'][-1] + self.gpus)
        usage['mem'].append(usage['mem'][-1] + self.mem)
        usage[self.name]['times'].append(time.time() / RUN_TIME_MODIFIER)
        usage[self.name]['count'].append(usage[self.name]['count'][-1] + 1)

        while time.time() - t0 < total_time:
            waited_time = time.time() - t0
            remaining = max(0., total_time - waited_time)
            await asyncio.sleep(min(remaining, dt))

        usage['times'].append(time.time() / RUN_TIME_MODIFIER)
        usage['cpus'].append(usage['cpus'][-1] - self.num_cpus)
        usage['gpus'].append(usage['gpus'][-1] - self.gpus)
        usage['mem'].append(usage['mem'][-1] - self.mem)
        usage[self.name]['times'].append(time.time() / RUN_TIME_MODIFIER)
        usage[self.name]['count'].append(usage[self.name]['count'][-1] - 1)


@dataclasses.dataclass
class Supervisor:
    actors: List[Actor]

    def __post_init__(self):
        self._actor_queue = asyncio.Queue()
        for i, _ in enumerate(self.actors):
            self._actor_queue.put_nowait(i)

    async def call(self):
        actor_idx = await self._actor_queue.get()
        actor = self.actors[actor_idx]
        await actor.call()
        self._actor_queue.put_nowait(actor_idx)


def create_supervisor(actor: Actor, num_actors: int):
    return Supervisor(actors=[actor] * num_actors)


async def main():
    num_channels = 40
    num_times_per_sol_int = 4
    num_freqs_per_sol_int = 40
    num_sol_ints = num_channels // num_freqs_per_sol_int
    num_coh = 4

    systematic_simulator = create_supervisor(
        Actor(
            name='systematic_simulator',
            num_cpus=1,
            gpus=0,
            mem=3.,
            run_time=timedelta(seconds=4.),
            deps=[]
        ),
        num_actors=num_channels * num_times_per_sol_int
    )

    dft_predictor = create_supervisor(
        Actor(
            name='dft_predictor',
            num_cpus=0,
            gpus=0.1,
            mem=10.,
            run_time=timedelta(seconds=1),
            deps=[]
        ),
        num_actors=2 * num_channels * num_times_per_sol_int
    )

    degridding_predictor = create_supervisor(
        Actor(
            name='degridding_predictor',
            num_cpus=32,
            gpus=0.,
            mem=50.,
            run_time=timedelta(seconds=10),
            deps=[]
        ),
        num_actors=10 * num_channels * num_times_per_sol_int
    )

    data_streamer = create_supervisor(
        Actor(
            name='data_streamer',
            num_cpus=1,
            gpus=0.,
            mem=10.,
            run_time=timedelta(seconds=0.1),
            deps=[(systematic_simulator,), (dft_predictor, degridding_predictor)]
        ),
        num_actors=num_channels * num_times_per_sol_int
    )

    model_streamer = create_supervisor(
        Actor(
            name='model_streamer',
            num_cpus=1,
            gpus=0.,
            mem=30.,
            run_time=timedelta(seconds=0.1),
            deps=[(dft_predictor, degridding_predictor)]
        ),
        num_actors=num_channels * num_times_per_sol_int
    )

    calibrator = create_supervisor(
        Actor(
            name='calibrator',
            num_cpus=1,
            gpus=0.,
            mem=30.,
            run_time=timedelta(seconds=10.),
            deps=[(data_streamer,) * num_times_per_sol_int * num_freqs_per_sol_int + (
                model_streamer,) * num_times_per_sol_int * num_freqs_per_sol_int]
        ),
        num_actors=num_sol_ints
    )

    gridder = create_supervisor(
        Actor(
            name='gridder',
            num_cpus=32,
            gpus=0.,
            mem=50.,
            run_time=timedelta(seconds=10.),
            deps=[(calibrator,)]
        ),
        num_actors=num_sol_ints
    )

    aggregator = create_supervisor(
        Actor(
            name='aggregator',
            num_cpus=1,
            gpus=0.,
            mem=10.,
            run_time=timedelta(seconds=0.),
            deps=[(gridder,) * num_sol_ints]
        ),
        num_actors=1
    )

    t0 = time.time()
    await aggregator.call()
    # await aggregator.call()
    # await aggregator.call()
    wall_time = (time.time() - t0)
    est_run_time = wall_time / RUN_TIME_MODIFIER
    print(f"Estimated Runtime: {est_run_time} seconds from Walltime: {wall_time} seconds")
    plt.plot(usage['times'], usage['cpus'])
    plt.savefig("cpu_usage.png")
    plt.show()
    plt.plot(usage['times'], usage['gpus'])
    plt.savefig("gpu_usage.png")
    plt.show()
    plt.plot(usage['times'], usage['mem'])
    plt.savefig("mem_usage.png")
    plt.show()

    plt.figure(figsize=(10,10))
    for name in ['aggregator', 'gridder', 'calibrator', 'data_streamer', 'model_streamer', 'dft_predictor',
                 'degridding_predictor', 'systematic_simulator']:
        plt.plot(usage[name]['times'],  usage[name]['count'], label=name)
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.savefig("actor_usage.png")
    plt.show()


if __name__ == '__main__':
    asyncio.run(main())
