import dataclasses
from typing import Tuple

import jax.numpy as jnp


@dataclasses.dataclass(eq=False)
class Kahan:
    shape: Tuple[int, ...] = ()
    dtype: jnp.dtype = jnp.float64

    def __post_init__(self):
        self.sum = self.c = jnp.zeros(self.shape, dtype=self.dtype)

    def __add__(self, other):
        y = other - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t
        return self.sum

    def __sub__(self, other):
        return self.__add__(-other)

    @property
    def value(self):
        return self.sum
