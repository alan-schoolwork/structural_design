from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Sequence

import equinox as eqx
import jax
import sympy as s
from jax import Array, ShapeDtypeStruct, lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax._src import ad_util, core, traceback_util
from jax._src.interpreters import batching
from jax.typing import ArrayLike
from jax.util import safe_map as map
from jax.util import safe_zip as zip
from pintax import areg, convert_unit
from pintax._utils import pp_obj, pretty_print
from pintax.unstable import unitify_rules

from .utils import cast_unchecked, dict_set, shape_of


def _remove_prefix(v: tuple[int, ...], prefix: tuple[int, ...]) -> tuple[int, ...]:
    assert v[: len(prefix)] == prefix
    return v[len(prefix) :]


def _remove_suffix(v: tuple[int, ...], suffix: tuple[int, ...]) -> tuple[int, ...]:
    assert v[len(v) - len(suffix) :] == suffix
    return v[: len(v) - len(suffix)]


class batched[T](eqx.Module):
    _bufs: list[Array]
    _shapes: list[ShapeDtypeStruct] = eqx.field(static=True)
    _pytree: jtu.PyTreeDef = eqx.field(static=True)

    @staticmethod
    def create[T2](val: T2, batch_dims: tuple[int, ...] = ()) -> batched[T2]:
        bufs, tree = jtu.tree_flatten(val)
        assert len(bufs) > 0
        _bufs: list[Array] = []
        _shapes: list[ShapeDtypeStruct] = []

        for x in bufs:
            if not isinstance(x, Array):
                x = jnp.array(x)
            _bufs.append(x)
            _shapes.append(
                ShapeDtypeStruct(_remove_prefix(x.shape, batch_dims), x.dtype)
            )
        return batched(_bufs, _shapes, tree)

    @staticmethod
    def create_unbatch[T2](val: T2, dims: tuple[int, ...]) -> batched[T2]:
        return tree_unbatch(batched.create(val), dims)

    def batch_dims(self) -> tuple[int, ...]:
        ans = [
            _remove_suffix(b.shape, s.shape) for b, s in zip(self._bufs, self._shapes)
        ]
        for x in ans:
            assert x == ans[0]
        return ans[0]

    def count(self) -> int:
        return math.prod(self.batch_dims())

    def unflatten(self) -> T:
        return jtu.tree_unflatten(self._pytree, self._bufs)

    def unwrap(self) -> T:
        assert self.batch_dims() == ()
        return self.unflatten()

    def __repr__(self):
        return pp_obj("batched", pretty_print(self.unflatten())).format()

    def repeat(self, n: int) -> batched[T]:
        return jax.vmap(lambda: self, axis_size=n)()

    def reshape(self, *new_shape: int) -> batched[T]:
        bd = self.batch_dims()

        return jtu.tree_map(
            lambda x: x.reshape(*new_shape, *x.shape[len(bd) :]),
            self,
        )

    @staticmethod
    def concat[T2](*args: batched[T2], axis: int = 0) -> batched[T2]:
        try:
            for x in args:
                assert x._pytree == args[0]._pytree
                assert x._shapes == args[0]._shapes
        except Exception as e:
            raise TypeError("invalid args:", args) from e

        return jtu.tree_map(lambda *args_leaf: jnp.concat(args_leaf, axis=axis), *args)

    def __getitem__(self, idx: Any) -> batched[T]:
        ans = jtu.tree_map(lambda x: x[idx], self)
        _ = ans.batch_dims()
        return ans

    def dynamic_slice(
        self, start_indices: Sequence[ArrayLike], slice_sizes: Sequence[int]
    ) -> batched[T]:
        bds = self.batch_dims()
        assert len(bds) == len(start_indices)
        assert len(bds) == len(slice_sizes)

        def slice_one(x):
            s = shape_of(x)
            rest = _remove_prefix(s, bds)
            return lax.dynamic_slice(
                x, [*start_indices, *(0 for _ in rest)], [*slice_sizes, *rest]
            )

        return jtu.tree_map(slice_one, self)

    def __len__(self) -> int:
        return self.batch_dims()[0]


unbatch_p = core.Primitive("unbatch")


def unbatch(x: ArrayLike, dims: tuple[int, ...]) -> ArrayLike:
    # transform stack: (bot) vmap1 > vmap2 (top)
    # dims == [vmap1, vmap2]
    # x -> (vmap1, vmap2, *x.shape)
    if len(dims) == 0:
        return x
    return unbatch_p.bind(x, dims=dims)


def tree_unbatch[T](x: T, dims: tuple[int, ...]) -> T:
    return jtu.tree_map(lambda l: unbatch(l, dims), x)


@unbatch_p.def_impl
def _(x: ArrayLike):
    assert False
    # return x


@dict_set(batching.fancy_primitive_batchers, unbatch_p)
def _(
    axis_data: batching.AxisData,
    batched_args: tuple[ArrayLike],
    batch_dims: tuple[int | None],
    dims: tuple[int, ...],
):
    (x,) = batched_args
    (bd,) = batch_dims

    assert dims[-1] == axis_data.size
    if bd != 0:
        x = jax.vmap(
            lambda v: v,
            in_axes=bd,
            out_axes=0,
            axis_size=axis_data.size,
        )(x)

    return unbatch(x, dims[:-1]), batching.not_mapped


# @unitify_rules(unbatch_p)
# def _(arg):
#     return arg


do_batch_p = core.Primitive("do_batch")


def do_batch(x: ArrayLike, dims: tuple[int, ...]) -> ArrayLike:
    # transform stack: (bot) vmap1 > vmap2 (top)
    # dims == [vmap1, vmap2]
    # ([vmap1, vmap2, *ans.shape], ref) -> ans
    if len(dims) == 0:
        return x
    return do_batch_p.bind(x, dims=dims)


def tree_do_batch[T](x: T, dims: tuple[int, ...]) -> T:
    return jtu.tree_map(lambda leaf: do_batch(leaf, dims), x)


@do_batch_p.def_impl
def _(*_, **__):
    assert False


@dict_set(batching.fancy_primitive_batchers, do_batch_p)
def _(
    axis_data: batching.AxisData,
    batched_args: tuple[ArrayLike],
    batch_dims: tuple[int | None],
    dims: tuple[int, ...],
):
    (x,) = batched_args
    (x_b,) = batch_dims

    assert x_b == batching.not_mapped
    assert dims[-1] == axis_data.size
    assert shape_of(x)[len(dims) - 1] == axis_data.size

    ans = do_batch(x, dims[:-1])
    assert shape_of(ans)[0] == axis_data.size

    return ans, 0


# @unitify_rules(do_batch_p)
# def _(x, ref):
#     return x
