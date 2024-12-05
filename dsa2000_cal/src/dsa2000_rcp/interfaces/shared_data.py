from multiprocessing import shared_memory
from typing import Tuple

import numpy as np


class SharedData:
    """
    Shared memory data for single machine array sharing between processes. This class is picklable.
    """

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        return (self._deserialise, (self._serialised_data,))

    @classmethod
    def _deserialise(cls, kwargs):
        # Create a new instance, bypassing __init__ and setting the actor directly
        return cls(**kwargs)

    def __init__(self, create: bool, shape: Tuple[int, ...], dtype: str, shm_name: str | None = None):
        if create:
            if shm_name is not None:
                raise ValueError(f"Expected `shm_name` to be None when `create` is True.")
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self._shm = shared_memory.SharedMemory(create=True, size=size)
            self._shared_arr = np.ndarray(shape, dtype=dtype, buffer=self._shm.buf)
            self._shared_arr[:] = 0.
            self._serialised_data = dict(create=False, shape=shape, dtype=dtype, shm_name=self._shm.name)
        else:
            if shm_name is None:
                raise ValueError(f"Expected `shm_name`.")
            self._shm = shared_memory.SharedMemory(name=shm_name)
            self._shared_arr = np.ndarray(shape=shape, dtype=dtype, buffer=self._shm.buf)
            self._serialised_data = dict(create=False, shape=shape, dtype=dtype, shm_name=shm_name)

    @property
    def shm_name(self) -> str:
        return self._shm.name

    def close(self):
        self._shm.close()

    def unlink(self):
        self._shm.unlink()

    def __getitem__(self, item):
        return self._shared_arr[item]

    def __setitem__(self, key, value):
        self._shared_arr[key] = value
