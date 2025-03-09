from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Concatenate, Iterable, Never, final

import jax
import sympy as s
from beartype import beartype as typechecker
from jax import Array
from jax import numpy as jnp
from jax import tree_util as jtu
from jax._src.typing import ArrayLike
from jax.interpreters.ad import JVPTrace, JVPTracer
from jaxtyping import Array, Bool, Float, Int, PyTree, jaxtyped
from sympy import Number
from sympy.core.basic import Printable

proj = Path(__file__).parent.parent


fval = Float[Array, ""]
ival = Int[Array, ""]
bval = Bool[Array, ""]


@final
class _empty_t:
    pass


_empty = _empty_t()


class cast[T]:
    def __init__(self, _: T | _empty_t = _empty) -> None:
        pass

    def __call__(self, a: T) -> T:
        return a


class cast_unchecked[T]:
    def __init__(self, _: T | _empty_t = _empty) -> None:
        pass

    def __call__(self, a) -> T:
        return a


def unique[T](x: Iterable[T]) -> T:
    it = iter(x)
    first = next(it)
    try:
        sec = next(it)
        raise ValueError(
            f"expected unique, got at least two elements: {first} and {sec}"
        )
    except StopIteration:
        return first


def _wrap_jit[F: Callable, **P](
    jit_fn: Callable[Concatenate[Callable, P], Any],
) -> Callable[Concatenate[F, P], F]:
    return jit_fn  # type: ignore


jit = _wrap_jit(jax.jit)


def debug_callback[**P](f: Callable[P, None], *args: P.args, **kwargs: P.kwargs):
    jax.debug.callback(f, *args, **kwargs)
