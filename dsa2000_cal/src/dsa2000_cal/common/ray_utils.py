import asyncio
import dataclasses
import datetime
import logging
import os
import socket
import threading
import time
import traceback
from contextlib import ContextDecorator
from datetime import timedelta
from typing import Coroutine, Callable, Any, Tuple

import psutil
import ray
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount, \
    nvmlDeviceGetComputeRunningProcesses
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray.util.metrics import Gauge

logger = logging.getLogger("ray")


def get_gpu_with_most_memory() -> Tuple[int, int]:
    """
    Get the GPU with the most free memory.

    Returns:
        the index of the GPU with the most free memory, and the amount of free memory on that GPU
    """
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    max_free_memory = 0
    best_gpu = None
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem_info.free
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu = i
    return best_gpu, max_free_memory


def set_all_gpus_visible():
    """
    Set all GPUs to be visible to the current process.
    """
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    if device_count > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(device_count))
    else:
        raise RuntimeError("No GPUs available!")


def get_head_node_id() -> str:
    """Get the head node id.

    Iterate through all nodes in the ray cluster and return the node id of the first
    alive node with head node resource.
    """
    head_node_id = None
    for node in ray.nodes():
        if HEAD_NODE_RESOURCE_NAME in node["Resources"] and node["Alive"]:
            head_node_id = node["NodeID"]
            break
    assert head_node_id is not None, "Cannot find alive head node."

    return head_node_id


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get a running async event loop if one exists, otherwise create one.

    This function serves as a proxy for the deprecating get_event_loop().
    It tries to get the running loop first, and if no running loop
    could be retrieved:
    - For python version <3.10: it falls back to the get_event_loop
        call.
    - For python version >= 3.10: it uses the same python implementation
        of _get_event_loop() at asyncio/events.py.

    Ideally, one should use high level APIs like asyncio.run() with python
    version >= 3.7, if not possible, one should create and manage the event
    loops explicitly.
    """
    import sys

    vers_info = sys.version_info
    if vers_info.major >= 3 and vers_info.minor >= 10:
        # This follows the implementation of the deprecating `get_event_loop`
        # in python3.10's asyncio. See python3.10/asyncio/events.py
        # _get_event_loop()
        try:
            loop = asyncio.get_running_loop()
            assert loop is not None
            return loop
        except RuntimeError as e:
            # No running loop, relying on the error message as for now to
            # differentiate runtime errors.
            if "no running event loop" in str(e):
                return asyncio.get_event_loop_policy().get_event_loop()
            else:
                raise e

    return asyncio.get_event_loop()


async def loop_task(task_fn: Callable[[], Coroutine[Any, Any, None]], interval: timedelta | Callable[[], timedelta],
                    raise_on_exception: bool = False):
    """
    Runs a task in a loop.

    Args:
        task_fn: the task
        interval: the interval between runs
        raise_on_exception: if True, raise exceptions, otherwise only log them
    """
    while True:
        try:
            await task_fn()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if raise_on_exception:
                raise e
            logger.exception(f"Problem with task {task_fn.__name__}: {str(e)}")
        if callable(interval):
            await asyncio.sleep(interval().total_seconds())
            continue
        await asyncio.sleep(interval.total_seconds())


class LogErrors:
    """
    Creates a context that catches all exceptions, logs them with a detailed message,
    including a stack trace, and then raises them. For use in ray actors that don't
    have good debugging.
    """

    def __init__(self, logfile, max_stack_depth: int = None):
        """
        :param max_stack_depth: Maximum depth of the stack trace to be logged.
        """
        self.logfile = logfile
        self.max_stack_depth = max_stack_depth

    def __enter__(self):
        # No setup required for this context manager, just return self.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If an exception was raised in the context, log it.
        if exc_type is not None:
            now = datetime.datetime.now()
            # Format the stack trace with the specified max depth
            formatted_traceback = ''.join(
                traceback.format_tb(exc_tb, limit=self.max_stack_depth)
            )
            # Log the error message and the formatted stack trace
            msg = (
                f"{now.isoformat()} An error occurred: {str(exc_val)}:\n"
                f"{formatted_traceback}"
            )
            print(msg)
            logger.error(
                msg,
                exc_info=(exc_type, exc_val, exc_tb)
            )
            with open(self.logfile, 'a') as f:
                f.write(
                    f"\n\n########\n\n"
                    f"{msg}"
                )
        # Returning False will re-raise the exception after logging.
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


def get_free_port() -> int:
    """
    Get a free port on the local machine.

    Returns:
        int: The free port number.
    """
    # Create a new socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind the socket to port 0, which tells the OS to pick a free port
    s.bind(('', 0))
    # Get the port number that was assigned
    free_port = s.getsockname()[1]
    # Close the socket
    s.close()
    return free_port


async def memory_logger(task: str, cadence: timedelta):
    """
    Log the memory usage of the process in MB.

    Args:
        task: the task name
        cadence: the cadence at which to log the memory usage
    """
    pid = os.getpid()
    python_process = psutil.Process(pid)
    nvmlInit()

    # Get the GPU handle (assuming single GPU at index 0)
    try:
        gpu_handles = [nvmlDeviceGetHandleByIndex(idx) for idx in range(nvmlDeviceGetCount())]
    except Exception as e:
        print(f"Unable to initialize NVML GPU tracking: {e}")
        gpu_handle = None

    memory_gauge = Gauge(
        name=f"memory_usage_gauge_MB",
        description="The memory usage of the process in MB",
        tag_keys=("task", "device_type")
    )
    memory_gauge.set_default_tags({"task": task})

    async def log_memory():
        logger.info(f"Logging memory usage for task {task} for pid {pid}")
        mem_MB = python_process.memory_info()[0] / 2 ** 20
        memory_gauge.set(mem_MB, tags={"device_type": "cpu", "task": task})
        # GPU memory usage by this process
        gpu_mem_MB = 0
        for gpu_handle in gpu_handles:
            if gpu_handle is not None:
                try:
                    processes = nvmlDeviceGetComputeRunningProcesses(gpu_handle)
                    for process in processes:
                        logger.info(f"Process: {process.pid}, Memory: {process.usedGpuMemory / 2 ** 20:.2f} MB")
                        if process.pid == pid:
                            gpu_mem_MB += process.usedGpuMemory / 2 ** 20
                except Exception as e:
                    print(f"Error logging GPU memory for process: {e}")
        memory_gauge.set(gpu_mem_MB, tags={"device_type": "gpu", "task": task})

    await loop_task(log_memory, cadence)


class MemoryLogger(ContextDecorator):
    def __init__(self, log_file='memory_usage.log', interval=1, kill_threshold=None, mode: str = 'w'):
        self.mode = mode
        self.log_file = log_file
        self.interval = interval
        self.kill_threshold = kill_threshold
        self._stop_event = threading.Event()
        self.logging_thread = None

    def _log_memory_usage(self):
        pid = os.getpid()
        python_process = psutil.Process(pid)

        with open(self.log_file, self.mode) as f:
            f.write("#Time (s), Memory Usage (MB)\n")
            start_time = time.time()

            while not self._stop_event.is_set():
                # Get memory usage in MB
                mem_MB = python_process.memory_info()[0] / 2 ** 20
                elapsed_time = time.time() - start_time

                # Write the data to the file
                f.write(f"{elapsed_time:.2f}, {mem_MB:.2f}\n")
                f.flush()

                # Check if memory exceeds the kill threshold
                if self.kill_threshold is not None and mem_MB > self.kill_threshold:
                    f.write(f"\nMemory threshold exceeded: {mem_MB:.2f} MB (Threshold: {self.kill_threshold} MB)\n")
                    f.flush()
                    # Terminate the process
                    os._exit(1)  # Forcefully terminate the process

                # Sleep for the specified interval
                time.sleep(self.interval)

    def __enter__(self):
        # Start the memory logging thread
        self.logging_thread = threading.Thread(target=self._log_memory_usage, daemon=True)
        self.logging_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Signal the thread to stop
        self._stop_event.set()
        # Wait for the thread to finish
        if self.logging_thread is not None:
            self.logging_thread.join()


@dataclasses.dataclass
class TimerLog:
    msg: str

    def __post_init__(self):
        self.t0 = time.time()

    def __enter__(self):
        logger.info(f"{self.msg}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"... took {time.time() - self.t0:.3f} seconds")
        return False
