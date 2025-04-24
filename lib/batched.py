from __future__ import annotations

import functools
import math
import typing
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Generic,
    Never,
    Protocol,
    TypeVar,
    TypeVarTuple,
    final,
    overload,
)

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

from .utils import (
    blike,
    cast,
    cast_fsig,
    cast_unchecked,
    dict_set,
    ival,
    pp_nested,
    shape_of,
    tree_at_,
    unreachable,
    vmap,
    wraps,
)

if TYPE_CHECKING:
    from jax._src.basearray import _IndexUpdateRef as _IndexUpdateRef_jax

traceback_util.register_exclusion(__file__)

type Tree[T] = T


def _remove_prefix(
    v: tuple[int, ...], prefix: tuple[int, ...], check: bool = False
) -> tuple[int, ...]:
    if check:
        assert v[: len(prefix)] == prefix
    return v[len(prefix) :]


def _remove_suffix(v: tuple[int, ...], suffix: tuple[int, ...]) -> tuple[int, ...]:
    assert v[len(v) - len(suffix) :] == suffix
    return v[: len(v) - len(suffix)]


def _batched_treemap_of[**P](fn: Callable[Concatenate[tuple[Array, ...], P], Array]):
    def inner[T](
        batches: Sequence[batched[T]], *args: P.args, **kwargs: P.kwargs
    ) -> batched[T]:
        return batched_treemap(
            wraps(fn)(lambda *bufs: fn(bufs, *args, **kwargs)), *batches
        )

    return inner


def _batched_treemap_of_one[**P](
    fn: Callable[Concatenate[Array, P], Array],
):
    def inner[T](arg: batched[T], *args: P.args, **kwargs: P.kwargs) -> batched[T]:
        return batched_treemap(lambda buf: fn(buf, *args, **kwargs), arg)

    return inner


T_co = TypeVar("T_co", covariant=True)


class batched(eqx.Module, Generic[T_co]):
    _bufs: list[Array]
    _shapes: list[ShapeDtypeStruct] = eqx.field(static=True)
    _pytree: jtu.PyTreeDef = eqx.field(static=True)
    _tracking: Array | None = None

    @staticmethod
    def create[T2](
        val: T2, batch_dims: tuple[int, ...] = (), *, broadcast: bool = False
    ) -> batched[T2]:
        bufs, tree = jtu.tree_flatten(val)
        if len(bufs) == 0:
            return batched([], [], tree, jnp.zeros(batch_dims))

        try:
            _bufs: list[Array] = []
            _shapes: list[ShapeDtypeStruct] = []

            for x in bufs:
                if not isinstance(x, Array):
                    x = jnp.array(x)
                shape = _remove_prefix(x.shape, batch_dims, check=not broadcast)
                _shapes.append(ShapeDtypeStruct(shape, x.dtype))
                if broadcast:
                    x, _ = jnp.broadcast_arrays(x, jnp.zeros(batch_dims + shape))
                    assert x.shape == batch_dims + shape
                _bufs.append(x)

            return batched(_bufs, _shapes, tree)
        except Exception as e:
            raise Exception(f"failed to create batched: {batch_dims}\n{val}") from e

    def __check_co(self) -> batched[object]:  # pyright: ignore[reportUnusedFunction]
        return self

    def batch_dims(self) -> tuple[int, ...]:
        if self._tracking is not None:
            return self._tracking.shape
        ans = [
            _remove_suffix(b.shape, s.shape) for b, s in zip(self._bufs, self._shapes)
        ]
        for x in ans:
            assert x == ans[0]
        return ans[0]

    def item_shape(self) -> Tree[T_co]:
        return jtu.tree_unflatten(self._pytree, self._shapes)

    @staticmethod
    def create_unbatch[T2](val: T2, dims: tuple[int, ...]) -> batched[T2]:
        return tree_unbatch(batched.create(val), dims)

    @staticmethod
    def _unreduce[T2](bds: tuple[int, ...], val: T2) -> batched[T2]:
        return batched.create(val, bds)

    @staticmethod
    def __reduce__typed[*P, T2](f: Callable[[*P], T2], obj: tuple[*P]):
        return f, obj

    def __reduce__(self):
        return self.__reduce__typed(
            self._unreduce,
            (
                self.batch_dims(),
                self.unflatten(),
            ),
        )

    def count(self) -> int:
        return math.prod(self.batch_dims())

    def unflatten(self) -> T_co:
        return jtu.tree_unflatten(self._pytree, self._bufs)

    @property
    def uf[T2: batched[object]](self: batched[T2]) -> T2:
        return self.unflatten()

    @property
    def arr(self: batched[ArrayLike]) -> Array:
        return jnp.array(self.unflatten())

    def get_arr(self, f: Callable[[T_co], ArrayLike]) -> Array:
        return batched_vmap(f, self).arr

    def unwrap(self) -> T_co:
        assert self.batch_dims() == ()
        return self.unflatten()

    def __repr__(self):
        return pp_obj("batched", pretty_print(self.unflatten())).format()

    @staticmethod
    @cast_fsig(jnp.arange)
    def arange(*args, **kwargs) -> batched[Array]:
        ans = jnp.arange(*args, **kwargs)
        (n,) = ans.shape
        return batched.create(ans, (n,))

    def repeat(self, n: int) -> batched[T_co]:
        return jax.vmap(lambda: self, axis_size=n)()

    def reshape(self, *new_shape: int) -> batched[T_co]:
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
    transpose = _batched_treemap_of_one(jnp.transpose)

    def __getitem__(self, idx: Any) -> batched[T_co]:
        return batched_treemap(lambda x: x[idx], self)

    def dynamic_slice(
        self, start_indices: Sequence[ArrayLike], slice_sizes: Sequence[int]
    ) -> batched[T_co]:
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

    def __iter__(self) -> Never:
        assert False

    def map[T2](self, f: Callable[[T_co], T2]) -> batched[T2]:
        return batched_vmap(f, self)

    def map1d[T2](self, f: Callable[[batched[T_co]], T2]) -> batched[T2]:
        return vmap((self,), lambda me: batched.create(f(me)))

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

    def filter(self, f: Callable[[T_co], blike]) -> tuple[batched[T_co], ival]:
        return self.filter_arr(self.map(f))

    def filter_concrete(self, f: Callable[[T_co], blike]) -> batched[T_co]:
        ans, ct = self.filter(f)
        return ans[: int(ct)]

    def filter_arr(self, bools_: batched[blike]) -> tuple[batched[T_co], ival]:
        (n,) = self.batch_dims()

        bools = bools_.unflatten()
        assert isinstance(bools, Array)
        assert bools.shape == (n,)

        idxs = jnp.nonzero(bools, size=n, fill_value=0)

        ans: batched[T_co] = jtu.tree_map(lambda x: x[idxs], self)
        assert ans.batch_dims() == (n,)
        return ans, bools.sum()

    def scan[T2, T3](
        self, f: Callable[[T2, T_co], tuple[T2, T3]], init: T2
    ) -> tuple[T2, batched[T3]]:
        (_,) = self.batch_dims()

        def inner(c: T2, x: batched[T_co]):
            c, y = f(c, x.unwrap())
            return c, batched.create(y)

        return lax.scan(inner, init=init, xs=self)

    def thread[T2](self: batched[Callable[[T2], T2]], init: T2) -> T2:
        (_,) = self.batch_dims()

        def inner(c: T2, f: batched[Callable[[T2], T2]]):
            return f.unwrap()(c), None

        ans, _ = lax.scan(inner, init=init, xs=self)
        return ans

    def all_idxs(self) -> batched[tuple[ival, ...]]:
        bds = self.batch_dims()
        if len(bds) == 0:
            return batched.create(())
        n = bds[0]

        def inner(i: batched[ival], v: batched[T_co]) -> batched[tuple[ival, ...]]:
            return v.all_idxs().map(lambda rest: (i.unwrap(), *rest))

        return jax.vmap(inner)(batched.arange(n), self)

    def enumerate1d[R](self, f: Callable[[ival, batched[T_co]], R]) -> batched[R]:
        (n, *_) = self.batch_dims()
        return vmap((jnp.arange(n), self), lambda i, me: batched.create(f(i, me)))

    def enumerate[R](
        self,
        f: (
            Callable[[T_co], R]
            | Callable[[T_co, ival], R]
            | Callable[[T_co, ival, ival], R]
            | Callable[[T_co, ival, ival, ival], R]
        ),
    ) -> batched[R]:
        f_ = cast_unchecked["Callable[[T_co, *tuple[ival, ...]], R]"]()(f)
        idxs = self.all_idxs()
        return batched_zip(self, idxs).tuple_map(lambda v, idx: f_(v, *idx))

    def split_batch_dims(
        self,
        *,
        outer: tuple[int, ...] | None = None,
        inner: tuple[int, ...] | None = None,
    ) -> batched[batched[T_co]]:
        bds = self.batch_dims()

        if inner is None:
            assert not outer is None
            inner = bds[len(outer) :]
        if outer is None:
            outer = bds[: -len(inner)]

        assert bds == outer + inner
        return batched.create(self, outer)

    def sort(self, key: Callable[[T_co], Array]):
        def inner(v: T_co):
            ans = key(v)
            assert isinstance(ans, Array)
            assert ans.shape == ()
            return ans

        keys = self.map(inner)
        sorted_indices = jnp.argsort(keys.arr)
        return self[sorted_indices]

    @property
    def at(self):
        return _IndexUpdateHelper(self)


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


@overload
def batched_zip[T1, T2](
    a1: batched[T1], a2: batched[T2], /
) -> batched[tuple[T1, T2]]: ...
@overload
def batched_zip[T1, T2, T3](
    a1: batched[T1], a2: batched[T2], a3: batched[T3], /
) -> batched[tuple[T1, T2, T3]]: ...


def batched_zip[T](*args: batched[T]) -> batched[tuple]:
    return batched_vmap(lambda *args: args, *args)


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
            for s1, s2 in zip(x._shapes, args[0]._shapes):
                assert s1.shape == s2.shape
    except Exception as e:
        raise TypeError(
            pp_nested(
                "batched_treemap: invalid args:",
                pretty_print(f),
                *(pretty_print(x.item_shape()) for x in args),
            )
        ) from e

    # bds = [x.batch_dims() for x in args]

    def handle_one(shape: tuple[int, ...], *bufs: Array):
        if len(shape) == 0:
            return f(*bufs)
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
    # ShapeDtypeStruct(_remove_prefix(x.shape, batch_dims), x.dtype)

    ans = batched(
        _bufs=new_bufs,
        _shapes=[
            ShapeDtypeStruct(x.shape, dtype=b.dtype)
            for x, b in zip(args[0]._shapes, new_bufs)
        ],
        _pytree=args[0]._pytree,
    )
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


class _IndexUpdateHelper[T](eqx.Module):
    _v: batched[T]

    def __getitem__(self, index: Any) -> _IndexUpdateRef[T]:
        return _IndexUpdateRef(self._v, index)


def _index_update_meth1[**P](
    fn: Callable[
        [type[_IndexUpdateRef_jax]],
        Callable[Concatenate[_IndexUpdateRef_jax, P], Array],
    ],
):
    def inner[T](
        self: _IndexUpdateRef[T], /, *args: P.args, **kwargs: P.kwargs
    ) -> batched[T]:

        def inner2(x: Array):
            _ref = x.at[self._idx]
            return fn(type(_ref))(_ref, *args, **kwargs)

        return batched_treemap(inner2, self._v)

    return inner


def _index_update_meth2[**P](
    fn: Callable[
        [type[_IndexUpdateRef_jax]],
        Callable[Concatenate[_IndexUpdateRef_jax, Array, P], Array],
    ],
):
    def inner[T](
        self: _IndexUpdateRef[T],
        values: batched[T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> batched[T]:

        def inner2(x: Array, y: Array):
            _ref = x.at[self._idx]
            return fn(type(_ref))(_ref, y, *args, **kwargs)

        return batched_treemap(inner2, self._v, values)

    return inner


class _IndexUpdateRef[T](eqx.Module):
    _v: batched[T]
    _idx: Any

    get = _index_update_meth1(lambda x: x.get)
    set = _index_update_meth2(lambda x: x.set)

    def dynamic_slice(self, slice_sizes: Sequence[int]):
        return self._v.dynamic_slice(self._idx, slice_sizes)

    # def dynamic_update(
    #     self, values: batched[T], allow_negative_indices: bool | Sequence[bool] = True
    # ):
    #     def update_one(x: Array, y: Array):
    #         return lax.dynamic_update_slice(
    #             x, y, self._idx, allow_negative_indices=allow_negative_indices
    #         )

    #     return batched_treemap(update_one, self._v, values)
