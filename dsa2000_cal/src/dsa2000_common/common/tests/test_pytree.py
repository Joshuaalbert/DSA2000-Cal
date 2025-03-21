import dataclasses
from typing import Tuple, List, Any

import jax
import pytest
from jax import numpy as jnp

from dsa2000_common.common.array_types import FloatArray
from dsa2000_common.common.pytree import Pytree


@dataclasses.dataclass(eq=False)
class MockPytree(Pytree):
    a: int
    b: FloatArray
    c: str

    @classmethod
    def flatten(cls, this) -> Tuple[List[Any], Tuple[Any, ...]]:
        return (
            [this.b],
            (this.a, this.c)
        )

    @classmethod
    def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]):
        [b] = children
        a, c = aux_data
        return cls(a=a, b=b, c=c)


MockPytree.register_pytree()


def test_pytree():
    m = MockPytree(1, jnp.array([1.0, 2.0]), 'hello')
    m.save('mock.pkl')
    m2 = MockPytree.load('mock.pkl')
    assert m.a == m2.a
    assert jnp.allclose(m.b, m2.b)
    assert m.c == m2.c

    @jax.jit
    def f(x):
        return x

    m3 = f(m)
    assert m3.a == m.a
    assert jnp.allclose(m3.b, m.b)
    assert m3.c == m.c

    @dataclasses.dataclass(eq=False)
    class LocalMockPytree(Pytree):
        a: int
        b: FloatArray
        c: str

        @classmethod
        def flatten(cls, this) -> Tuple[List[Any], Tuple[Any, ...]]:
            return (
                [this.b],
                (this.a, this.c)
            )

        @classmethod
        def unflatten(cls, aux_data: Tuple[Any, ...], children: List[Any]):
            [b] = children
            a, c = aux_data
            return cls(a=a, b=b, c=c)

    LocalMockPytree.register_pytree()

    m = LocalMockPytree(1, jnp.array([1.0, 2.0]), 'hello')
    with pytest.raises(AttributeError):
        m.save('mock.pkl')

    @jax.jit
    def f(x):
        return x

    m3 = f(m)
    assert m3.a == m.a
    assert jnp.allclose(m3.b, m.b)
    assert m3.c == m.c
