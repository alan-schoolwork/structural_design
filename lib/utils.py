from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Concatenate,
    Protocol,
    final,
)

import equinox as eqx
import jax
import jax._src.pretty_printer as pp
import oryx
from jax import Array
from jax import numpy as jnp
from jax import tree_util as jtu
from jax import util as ju
from jax._src import core
from jax._src.typing import ArrayLike
from jax.experimental.checkify import check
from jax.typing import DTypeLike
from jaxtyping import Bool, Float, Int

proj = Path(__file__).parent.parent


fval = Float[Array, ""]
ival = Int[Array, ""]
bval = Bool[Array, ""]

blike = Bool[Array, ""] | bool

flike = Float[ArrayLike, ""]


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


def cast_unchecked_(x):
    return cast_unchecked()(x)


def cast_fsig[**P, R](f1: Callable[P, Any]):
    def inner(f2: Callable[..., R]) -> Callable[P, R]:
        return f2

    return inner


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
    return cast_unchecked_(jit_fn)


jit = _wrap_jit(jax.jit)


def debug_callback[**P](f: Callable[P, None], *args: P.args, **kwargs: P.kwargs):
    jax.debug.callback(f, *args, **kwargs)


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
    return core.get_aval(x).shape  # pyright: ignore[reportAttributeAccessIssue]


def return_of_fake[R](f: Callable[..., R]) -> R:
    return cast_unchecked_(None)


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

    def __init__(self, func):  # pyright: ignore[reportMissingSuperCall]
        type.__setattr__(self, "_func", objwrapper(func))

    def __getattribute__(self, name: str):
        return getattr(_allow_autoreload_get(self), name)

    def __setattr__(self, name: str, val):
        return setattr(_allow_autoreload_get(self), name, val)

    @property
    def __call__(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return _allow_autoreload_get(self).__call__


def allow_autoreload[T](x: T) -> T:
    return cast_unchecked(x)(_allow_autoreload(x))


# modified from equinox
_comma_sep = pp.concat([pp.text(","), pp.brk()])


def bracketed(
    name: pp.Doc | None,
    indent: int,
    objs: Sequence[pp.Doc],
    lbracket: str,
    rbracket: str,
) -> pp.Doc:
    nested = pp.concat(
        [
            pp.nest(indent, pp.concat([pp.brk(""), pp.join(_comma_sep, objs)])),
            pp.brk(""),
        ]
    )
    concated = []
    if name is not None:
        concated.append(name)
    concated.extend([pp.text(lbracket), nested, pp.text(rbracket)])
    return pp.group(pp.concat(concated))


def named_objs(pairs):
    return [
        pp.concat([pp.text(key + "="), pretty_print(value)]) for key, value in pairs
    ]


def pformat_dataclass(obj) -> pp.Doc:
    objs = named_objs(
        [
            (field.name, getattr(obj, field.name, pp.text("<uninitialised>")))
            for field in dataclasses.fields(obj)
            if field.repr
        ]
    )
    return bracketed(
        name=pp.text(obj.__class__.__name__),
        indent=2,
        objs=objs,
        lbracket="(",
        rbracket=")",
    )


def pformat_repr(self: Any):
    return pformat_dataclass(self).format()


def pretty_print(x: Any) -> pp.Doc:
    if isinstance(x, pp.Doc):
        return x
    # if isinstance(x, core.Tracer):
    #     return x._pretty_print()
    return pp_join(*repr(x).splitlines())


def _pp_doc(x: pp.Doc | str) -> pp.Doc:
    if isinstance(x, pp.Doc):
        return x
    return pp.text(x)


def pp_join(*docs: pp.Doc | str, sep: pp.Doc | str | None = None) -> pp.Doc:
    if sep is None:
        sep = pp.brk()
    return pp.join(_pp_doc(sep), [_pp_doc(x) for x in docs])


def pp_nested(*docs: pp.Doc | str) -> pp.Doc:
    return pp.group(pp.nest(2, pp_join(*docs)))


def concatenate(
    arrays: Sequence[Any], axis: int | None = 0, dtype: DTypeLike | None = None
) -> Array:
    return jnp.concatenate([jnp.array(x) for x in arrays], axis, dtype)


def vmap[*A, R](args: tuple[*A], f: Callable[[*A], R]) -> R:
    return jax.vmap(f)(*args)


def wraps(fn):
    return ju.wraps(fn)


def _tree_map[T](
    f: Callable, tree: T, *rest: T, is_leaf: Callable[[Any], bool] | None = None
) -> T:
    assert False


tree_map = cast_unchecked(_tree_map)(jtu.tree_map)
