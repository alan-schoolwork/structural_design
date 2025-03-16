from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import sympy as s
from jax import Array, ShapeDtypeStruct, lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.typing import ArrayLike
from pintax import areg, convert_unit

from .utils import cast_unchecked


def _batch_dims(v: tuple[int, ...], suffix: tuple[int, ...]) -> tuple[int, ...]:
    assert v[-len(suffix) :] == suffix
    return v[: -len(suffix)]


class batched[T](eqx.Module):
    _bufs: list[Array]
    _shapes: list[ShapeDtypeStruct] = eqx.field(static=True)
    _pytree: jtu.PyTreeDef = eqx.field(static=True)

    @staticmethod
    def one(val: T) -> batched:
        bufs, tree = jtu.tree_flatten(val)
        assert len(bufs) > 0
        _bufs: list[Array] = []
        _shapes: list[ShapeDtypeStruct] = []

        for x in bufs:
            if not isinstance(x, Array):
                x = jnp.array(x)
            _bufs.append(x)
            _shapes.append(
                ShapeDtypeStruct(
                    x.shape,
                    x.dtype,
                    weak_type=x.weak_type,
                )
            )
        return batched(_bufs, _shapes, tree)

    def batch_dims(self) -> tuple[int, ...]:
        ans = [_batch_dims(b.shape, s.shape) for b, s in zip(self._bufs, self._shapes)]
        for x in ans:
            assert x == ans[0]
        return ans[0]

    def unwrap(self) -> T:
        assert self.batch_dims() == ()
        return jtu.tree_unflatten(self._pytree, self._bufs)
