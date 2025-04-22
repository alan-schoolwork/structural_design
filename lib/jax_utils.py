from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np
from jax import Array, core
from jax import numpy as jnp
from jax._src import linear_util as lu
from jax._src import traceback_util
from jax._src.traceback_util import api_boundary
from jax._src.typing import ArrayLike

from lib.utils import cast, cast_unchecked_

traceback_util.register_exclusion(__file__)


def compose_fn[**P, T, A](f1: Callable[P, T], f2: Callable[[T], A]) -> Callable[P, A]:
    def inner(*args: P.args, **kwargs: P.kwargs):
        t = f1(*args, **kwargs)
        return f2(t)

    return inner


class Empty_t:
    pass


_empty = Empty_t()

is_leaf_t = Callable[[Any], bool] | None


def cache_by_jaxpr[**P, T](f: Callable[P, T]) -> Callable[P, T]:

    prev_in_tree = None
    out_tree = None
    jaxpr = None

    def inner(*args, **kwargs):
        nonlocal prev_in_tree, out_tree, jaxpr

        in_arrs, in_tree = jtu.tree_flatten((args, kwargs))

        if jaxpr is not None:
            assert in_tree == prev_in_tree
        else:
            prev_in_tree = in_tree

            def inner(bufs):
                nonlocal out_tree

                x, y = jtu.tree_unflatten(in_tree, bufs)
                ans = f(*x, **y)  # pyright: ignore[reportCallIssue]
                out_arrs, out_tree = jtu.tree_flatten(ans)
                return out_arrs

            jaxpr = jax.make_jaxpr(inner)(in_arrs)

        ans = core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *in_arrs)
        assert out_tree is not None
        return jtu.tree_unflatten(out_tree, ans)

    return inner


class jaxpr_with_tree[**P, T](eqx.Module):
    in_tree: jtu.PyTreeDef = eqx.field(static=True)
    out_tree: jtu.PyTreeDef = eqx.field(static=True)
    jaxpr: core.Jaxpr = eqx.field(static=True)
    jaxpr_consts: list[ArrayLike]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        in_arrs = self.in_tree.flatten_up_to((args, kwargs))
        ans = core.eval_jaxpr(self.jaxpr, self.jaxpr_consts, *in_arrs)
        return self.out_tree.unflatten(ans)


def fn_as_traced[**P, T](f: Callable[P, T]) -> Callable[P, jaxpr_with_tree[P, T]]:

    def inner(*args: P.args, **kwargs: P.kwargs) -> jaxpr_with_tree[P, T]:

        out_tree: jtu.PyTreeDef | None = cast_unchecked_(None)
        in_arrs, in_tree = jtu.tree_flatten((args, kwargs))

        def inner2(bufs: list[Array]):
            nonlocal out_tree
            x, y = jtu.tree_unflatten(in_tree, bufs)
            ans = f(*x, **y)  # pyright: ignore[reportCallIssue]
            out_arrs, out_tree = jtu.tree_flatten(ans)
            return out_arrs

        jaxpr = jax.make_jaxpr(inner2)(in_arrs)
        assert out_tree is not None

        return jaxpr_with_tree(in_tree, out_tree, jaxpr.jaxpr, jaxpr.consts)

    return inner


class flatten_handler[T](eqx.Module):

    flatten: jaxpr_with_tree[[T], Array]
    unflatten: jaxpr_with_tree[[Array], T]

    @staticmethod
    def create[T2](val: T2) -> tuple[flatten_handler[T2], Array]:

        def inner(val_: T2) -> Array:
            bufs, tree = jtu.tree_flatten(val_)
            return jnp.concat([x.ravel() for x in bufs])

        flatten = fn_as_traced(inner)(val)
        flattened = flatten(val)

        def inner2(x: Array) -> T2:
            (unflat_,) = jax.linear_transpose(flatten, val)(x)
            return unflat_

        unflatten = fn_as_traced(inner2)(flattened)

        return flatten_handler(flatten, unflatten), flatten(val)

    def __repr__(self):
        return f"flatten_handler({repr(self.flatten.in_tree)})"


def _callback_wrapped(fn: Callable, tree: jtu.PyTreeDef, np_printoptions):
    def inner(*bufs):
        args, kwargs = jtu.tree_unflatten(tree, bufs)
        with np.printoptions(**np_printoptions):
            fn(*args, **kwargs)

    return inner


def debug_callback[**P](fn: Callable, *args: P.args, **kwargs: P.kwargs):
    bufs, tree = jtu.tree_flatten((args, kwargs))
    jax.debug.callback(
        _callback_wrapped(fn, tree, np.get_printoptions()), *bufs, ordered=True
    )


@cast(print)
def debug_print(*args, **kwargs):
    debug_callback(print, *args, **kwargs)
