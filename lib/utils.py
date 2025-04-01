from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Concatenate,
    Iterable,
    Protocol,
    final,
)

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp
from jax._src import core
from jax._src.typing import ArrayLike
from jax.experimental.checkify import check
from jaxtyping import Array, Bool, Float, Int

proj = Path(__file__).parent.parent


fval = Float[Array, ""]
ival = Int[Array, ""]
bval = Bool[Array, ""]

blike = Bool[Array, ""] | bool


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

    @staticmethod
    def from_fn[R](f: Callable[..., R]) -> cast_unchecked[R]:
        return cast_unchecked()

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


class _jit_wrapped[**P, R]:
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def trace(self, *args: P.args, **kwargs: P.kwargs) -> jax.stages.Traced: ...
    def lower(self, *args: P.args, **kwargs: P.kwargs) -> jax.stages.Lowered: ...


class _jit_fn[**JitP]:
    def __call__[**P, R](
        self, f: Callable[P, R], /, *args: JitP.args, **kwargs: JitP.kwargs
    ) -> _jit_wrapped[P, R]: ...


def _wrap_jit[**P](
    jit_fn: Callable[Concatenate[Callable, P], Any],
) -> _jit_fn[P]:
    return jit_fn  # type: ignore


jit = _wrap_jit(jax.jit)


def debug_callback[**P](f: Callable[P, None], *args: P.args, **kwargs: P.kwargs):
    jax.debug.callback(f, *args, **kwargs)


class _marker_fn[N]:
    def __call__(self, x: N) -> N:
        return x


def tree_at_[T, N](
    where: Callable[[T], N],
    pytree: T,
    replace: N | None = None,
    replace_fn: Callable[[N], N] | None = None,
) -> T:
    kwargs = {}
    if replace is not None:
        kwargs["replace"] = replace
    if replace_fn is not None:
        kwargs["replace_fn"] = replace_fn

    return eqx.tree_at(where=where, pytree=pytree, **kwargs)


def tree_at_2_[T, *Ns](
    where: Callable[[T], tuple[*Ns]],
    pytree: T,
    replace: tuple[*Ns] | None = None,
) -> T:
    return eqx.tree_at(where=where, pytree=pytree, replace=replace)


class custom_vmap_res[*P, R](Protocol):
    def __call__(self, *args: *P) -> R: ...

    def def_vmap(
        self, f: Callable[[int, tuple[*P], *P], tuple[R, R]], /
    ) -> Callable: ...


def custom_vmap_[*P, R](f: Callable[[*P], R]) -> custom_vmap_res[*P, R]:
    return cast_unchecked()(jax.custom_batching.custom_vmap(f))


def dict_set[K, V](d: dict[K, V], k: K) -> Callable[[V], V]:
    def inner(v: V):
        d[k] = v
        return v

    return inner


def shape_of(x: ArrayLike) -> tuple[int, ...]:
    return core.get_aval(x).shape  # type: ignore


def return_of_fake[R](f: Callable[..., R]) -> R:
    return None  # type: ignore


def check_nan(x: ArrayLike):
    check(~jnp.any(jnp.isnan(x)), "got nan")


class _v_and_g[**Opts](Protocol):
    def __call__[T, R, A](
        self, f: Callable[[T], tuple[R, A]], /, *args: Opts.args, **kwargs: Opts.kwargs
    ) -> Callable[[T], tuple[tuple[R, A], T]]: ...


def _wrap_value_and_grad_aux[**P](
    f: Callable[Concatenate[Callable, P], Any],
) -> _v_and_g[P]:
    return partial(cast_unchecked()(f), has_aux=True)


value_and_grad_aux_ = _wrap_value_and_grad_aux(jax.value_and_grad)


@dataclass
class objwrapper:
    obj: Any


def _allow_autoreload_get(x):
    return object.__getattribute__(x, "_func").obj


class _allow_autoreload(type):

    def __new__(cls, func):
        del func
        return super().__new__(cls, "", (), {})

    def __init__(self, func):
        type.__setattr__(self, "_func", objwrapper(func))

    def __getattribute__(self, name: str):
        return getattr(_allow_autoreload_get(self), name)

    def __setattr__(self, name: str, val):
        return setattr(_allow_autoreload_get(self), name, val)

    @property
    def __call__(self):
        return _allow_autoreload_get(self).__call__


def allow_autoreload[T](x: T) -> T:
    return cast_unchecked(x)(_allow_autoreload(x))
