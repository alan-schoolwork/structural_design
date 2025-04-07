from __future__ import annotations

import math
from typing import Any, Callable, Concatenate, Sequence, overload

import equinox as eqx
import jax
from jax import Array, ShapeDtypeStruct, lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax._src import core, traceback_util
from jax._src.interpreters import batching
from jax.typing import ArrayLike
from jax.util import safe_zip as zip

from pintax._utils import pp_obj, pretty_print

from .utils import blike, dict_set, ival, shape_of, tree_at_

traceback_util.register_exclusion(__file__)


def _remove_prefix(v: tuple[int, ...], prefix: tuple[int, ...]) -> tuple[int, ...]:
    assert v[: len(prefix)] == prefix
    return v[len(prefix) :]


def _remove_suffix(v: tuple[int, ...], suffix: tuple[int, ...]) -> tuple[int, ...]:
    assert v[len(v) - len(suffix) :] == suffix
    return v[: len(v) - len(suffix)]


def _batched_treemap_of[**P](fn: Callable[Concatenate[tuple[Array, ...], P], Array]):
    def inner[T](
        batches: Sequence[batched[T]], *args: P.args, **kwargs: P.kwargs
    ) -> batched[T]:
        return batched_treemap(lambda *bufs: fn(bufs, *args, **kwargs), *batches)

    return inner


def _batched_treemap_of_one[**P](
    fn: Callable[Concatenate[Array, P], Array],
):
    def inner[T](arg: batched[T], *args: P.args, **kwargs: P.kwargs) -> batched[T]:
        return batched_treemap(lambda buf: fn(buf, *args, **kwargs), arg)

    return inner


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

        if new_shape == (-1,) and len(bd) == 1:
            return self

        return jtu.tree_map(
            lambda x: x.reshape(*new_shape, *x.shape[len(bd) :]),
            self,
        )

    concat = staticmethod(_batched_treemap_of(jnp.concat))
    stack = staticmethod(_batched_treemap_of(jnp.stack))
    roll = _batched_treemap_of_one(jnp.roll)

    def __getitem__(self, idx: Any) -> batched[T]:
        return batched_treemap(lambda x: x[idx], self)

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

    def map[T2](self, f: Callable[[T], T2]) -> batched[T2]:
        return batched_vmap(f, self)

    def tuple_map[*T1, T2](
        self: batched[tuple[*T1]], f: Callable[[*T1], T2]
    ) -> batched[T2]:
        return batched_vmap(lambda x: f(*x), self)

    @overload
    def split_tuple[T1, T2](
        self: batched[tuple[T1, T2]],
    ) -> tuple[batched[T1], batched[T2]]: ...

    @overload
    def split_tuple[T1, T2, T3](
        self: batched[tuple[T1, T2, T3]],
    ) -> tuple[batched[T1], batched[T2], batched[T3]]: ...

    def split_tuple(self: batched[tuple]) -> tuple[batched, ...]:
        bds = self.batch_dims()
        me = self.unflatten()
        assert isinstance(me, tuple)
        return tuple(batched.create(x, bds) for x in me)

    def filter(self, f: Callable[[T], blike]) -> tuple[batched[T], ival]:
        return self.filter_arr(self.map(f))

    def filter_arr(self, bools_: batched[blike]) -> tuple[batched[T], ival]:
        (n,) = self.batch_dims()

        bools = bools_.unflatten()
        assert isinstance(bools, Array)
        assert bools.shape == (n,)

        idxs = jnp.nonzero(bools, size=n, fill_value=0)

        ans: batched[T] = jtu.tree_map(lambda x: x[idxs], self)
        assert ans.batch_dims() == (n,)
        return ans, bools.sum()


@overload
def batched_vmap[T1, R](f: Callable[[T1], R], a1: batched[T1], /) -> batched[R]: ...
@overload
def batched_vmap[T1, T2, R](
    f: Callable[[T1, T2], R], a1: batched[T1], a2: batched[T2], /
) -> batched[R]: ...
@overload
def batched_vmap[T1, T2, T3, R](
    f: Callable[[T1, T2, T3], R], a1: batched[T1], a2: batched[T2], a3: batched[T3], /
) -> batched[R]: ...


def batched_vmap[R](f: Callable[..., R], *args: batched) -> batched[R]:
    bds = [x.batch_dims() for x in args]
    for bd in bds:
        assert bd == bds[0]
    if bds[0] == ():
        return batched.create(f(*(x.unwrap() for x in args)))

    def inner(*args: batched) -> batched[R]:
        return batched_vmap(f, *args)

    return jax.vmap(inner)(*args)


def batched_zip[T1, T2](a1: batched[T1], a2: batched[T2], /) -> batched[tuple[T1, T2]]:
    return batched_vmap(lambda *args: args, a1, a2)


@overload
def batched_treemap[T](
    f: Callable[[Array], Array], a1: batched[T], /
) -> batched[T]: ...
@overload
def batched_treemap[T](
    f: Callable[[Array, Array], Array], a1: batched[T], a2: batched[T], /
) -> batched[T]: ...
@overload
def batched_treemap[T](
    f: Callable[[*tuple[Array, ...]], Array], /, *args: batched[T]
) -> batched[T]: ...


def batched_treemap[T](f: Callable[..., Array], /, *args: batched[T]) -> batched[T]:
    try:
        for x in args:
            assert x._pytree == args[0]._pytree
            assert x._shapes == args[0]._shapes
    except Exception as e:
        raise TypeError("invalid args:", args) from e

    bds = [x.batch_dims() for x in args]

    def handle_one(shape: tuple[int, ...], *bufs: Array):
        if len(shape) == 0:
            ans = f(*bufs)
            # TODO: dont use args[0] shape as return so that dtype can be changed
            assert ans.dtype == bufs[0].dtype
            return ans
        else:
            return jax.vmap(
                lambda *bufs: handle_one(shape[:-1], *bufs),
                in_axes=-1,
                out_axes=-1,
                axis_size=shape[-1],
            )(*bufs)

    new_bufs = [
        handle_one(s.shape, *bufs)
        for (s, *bufs) in zip(args[0]._shapes, *(x._bufs for x in args))
    ]
    ans = tree_at_(lambda x: x._bufs, args[0], new_bufs)
    _ = ans.batch_dims()
    return ans


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
