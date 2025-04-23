from __future__ import annotations

import math
from typing import Callable

import equinox as eqx
import jax
from jax import Array, lax
from jax import numpy as jnp
from jax.typing import ArrayLike

from lib.batched import (
    batched,
    batched_treemap,
    batched_zip,
    tree_do_batch,
)
from pintax._utils import pp_obj, pretty_print

from .utils import blike, flike, pformat_repr, tree_at_


class point(eqx.Module):
    coords: Array

    fixed: blike
    accepts_force: blike

    __repr__ = pformat_repr

    def maybe_pin(self) -> point:
        ans = lax.select(
            self.fixed, on_true=lax.stop_gradient(self.coords), on_false=self.coords
        )
        return tree_at_(lambda p: p.coords, self, ans)


class pointid(eqx.Module):
    _idx: Array

    __repr__ = pformat_repr

    @staticmethod
    def ex():
        return pointid(jnp.array(0))


class connection(eqx.Module):
    a: pointid
    b: pointid
    force_per_deform: ArrayLike = 0.0
    weight: ArrayLike = 0.0
    density: ArrayLike = 0.0

    def get_weight(self, l: Array):
        return l * self.density + self.weight

    __repr__ = pformat_repr


class force_annotation(eqx.Module):
    p: pointid
    f: Array

    __repr__ = pformat_repr


class graph_t(eqx.Module):
    dim: int = eqx.field(static=True)
    _points: batched[point]
    _connections: batched[connection]
    _external_forces: batched[force_annotation]

    def __repr__(self):
        return pp_obj(
            "graph_t",
            pretty_print(self._points.unflatten()),
            pretty_print(self._connections.unflatten()),
        ).format()

    @staticmethod
    def create(dim: int, *, connection_ex: connection | None = None) -> graph_t:
        _id = pointid.ex()
        if connection_ex is None:
            connection_ex = connection(_id, _id)
        return graph_t(
            dim=dim,
            _points=batched.create(point(jnp.zeros(dim), False, False)).repeat(0),
            _connections=batched.create(connection_ex).repeat(0),
            _external_forces=batched.create(
                force_annotation(_id, jnp.zeros(dim))
            ).repeat(0),
        )

    def __add__(self, mod: Callable[[graph_t], graph_t]) -> graph_t:
        return mod(self)

    def add_point(
        self,
        c: Array,
        unbatch_dims: tuple[int, ...] = (),
        *,
        fixed: blike = False,
        accepts_force: blike = False,
    ) -> tuple[graph_t, pointid]:
        return self._add_point(
            point(c, fixed=fixed, accepts_force=accepts_force), unbatch_dims
        )

    def add_point_batched(
        self,
        c: batched[Array],
        *,
        fixed: blike = False,
        accepts_force: blike = False,
    ) -> tuple[graph_t, batched[pointid]]:
        return self._add_points_batched(
            c.map(lambda c: point(c, fixed=fixed, accepts_force=accepts_force))
        )

    def _add_point(
        self, x: point, unbatch_dims: tuple[int, ...] = ()
    ) -> tuple[graph_t, pointid]:
        xs = batched.create_unbatch(x, unbatch_dims)
        self, ids = self._add_points_batched(xs)
        return self, tree_do_batch(ids, dims=unbatch_dims).unwrap()

    def _add_points_batched(
        self, points: batched[point]
    ) -> tuple[graph_t, batched[pointid]]:
        dims = points.batch_dims()
        count = math.prod(dims)
        idxs = self._points.count() + jnp.arange(count)
        new_points = batched.concat([self._points, points.reshape(-1)])
        ans_pids = batched.create(pointid(idxs), (count,)).reshape(*dims)
        return tree_at_(lambda me: me._points, self, new_points), ans_pids

    # def add_connection(
    #     self, p1: pointid, p2: pointid
    # ) -> graph_t:
    #     return self._add_connection(connection(p1, p2))

    def add_connection_unbatched(
        self, c: connection, unbatch_dims: tuple[int, ...] = ()
    ) -> graph_t:
        return self.add_connection_batched(batched.create_unbatch(c, unbatch_dims))

    def add_connection_batched(self, cs: batched[connection]) -> graph_t:
        return tree_at_(
            lambda me: me._connections,
            self,
            batched.concat([self._connections, cs.reshape(-1)]),
        )

    # def add_connection_batched(
    #     self, p1: batched[pointid], p2: batched[pointid]
    # ) -> graph_t:
    #     return self._add_connection_batched(batched_zip(p1, p2).tuple_map(connection))

    def get_point(self, pid: pointid) -> point:
        return self._points[pid._idx].unwrap()

    def set_point(self, pid: pointid, new: point) -> graph_t:
        ans = batched_treemap(
            lambda x, y: x.at[pid._idx].set(y), self._points, batched.create(new)
        )
        return tree_at_(lambda me: me._points, self, ans)

    def get_lines(self) -> Array:
        # return accepted by Line3DCollection after tolist
        # which wants
        # list[tuple[list[float], list[float]]]

        def inner(c: connection):
            p1 = self.get_point(c.a).coords
            p2 = self.get_point(c.b).coords
            return jnp.stack([p1, p2])

        return self._connections.map(inner).unflatten()

    def sum_annotations(self) -> batched[Array]:
        n = len(self._points)

        anno_buf = self._external_forces.unflatten()

        buf = jnp.zeros_like(self._points.uf.coords)

        return batched.create(buf.at[anno_buf.p._idx].add(anno_buf.f), batch_dims=(n,))

    def add_external_force(self, xs: force_annotation):
        return self.add_external_force_batched(batched.create(xs))

    def add_external_force_batched(self, xs: batched[force_annotation]):
        return tree_at_(
            lambda me: me._external_forces,
            self,
            batched.concat([self._external_forces, xs.reshape(-1)]),
        )

    # def add_external_force(
    #     self, x: force_annotation, unbatch_dims: tuple[int, ...] = ()
    # ):
    #     xs = batched.create_unbatch(x, unbatch_dims)
    #     return tree_at_(
    #         lambda me: me._external_forces,
    #         self,
    #         batched.concat([self._external_forces, xs.reshape(-1)]),
    #     )

    def forces_aggregate(
        self, connection_forces: batched[Array]
    ) -> batched[force_annotation]:
        # connection_forces: >0 ==> compression
        def inner(
            c_: batched[connection], f_: batched[Array]
        ) -> batched[force_annotation]:
            c = c_.unwrap()
            f = f_.unwrap()

            a = self.get_point(c.a)
            b = self.get_point(c.b)

            v = b.coords - a.coords
            v_len = jnp.linalg.norm(v)
            v_dir = v / v_len

            force_vec_on_a = -f * v_dir

            ans1 = batched.create(force_annotation(c.a, force_vec_on_a))
            ans2 = batched.create(force_annotation(c.b, -force_vec_on_a))

            return batched.stack([ans1, ans2])

        ans = jax.vmap(inner)(self._connections, connection_forces)
        return ans.reshape(-1)

    def maybe_pin_points(self):
        return tree_at_(
            lambda me: me._points, self, self._points.map(lambda x: x.maybe_pin())
        )
