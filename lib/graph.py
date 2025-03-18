from __future__ import annotations

import math
from typing import Callable

import equinox as eqx
import jax
from jax import Array, lax
from jax import numpy as jnp
from pintax._utils import pp_obj, pretty_print

from lib.batched import (
    batched,
    tree_do_batch,
)

from .utils import blike, fval, tree_at_


class point(eqx.Module):
    coords: Array

    fixed: blike
    accepts_force: blike

    def __repr__(self):
        return pp_obj("point", pretty_print(self.coords)).format()

    def maybe_pin(self) -> point:
        ans = lax.select(
            self.fixed, on_true=lax.stop_gradient(self.coords), on_false=self.coords
        )
        return tree_at_(lambda p: p.coords, self, ans)


class pointid(eqx.Module):
    _idx: Array

    def __repr__(self):
        return pp_obj("pointid", pretty_print(self._idx)).format()


class connection(eqx.Module):
    a: pointid
    b: pointid

    def __repr__(self):
        return pp_obj("connection", pretty_print(self.a), pretty_print(self.b)).format()


class force_annotation(eqx.Module):
    p: pointid
    f: Array


class graph_t(eqx.Module):
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
    def create() -> graph_t:
        _id = pointid(jnp.array(0))
        return graph_t(
            _points=batched.create(point(jnp.zeros(3), False, False)).repeat(0),
            _connections=batched.create(connection(_id, _id)).repeat(0),
            _external_forces=batched.create(force_annotation(_id, jnp.zeros(3))).repeat(
                0
            ),
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
        tmp = batched.create(pointid(idxs), (count,))
        ans_pids = batched.create(pointid(idxs), (count,)).reshape(*dims)
        return tree_at_(lambda me: me._points, self, new_points), ans_pids

    def add_connection(
        self, p1: pointid, p2: pointid, unbatch_dims: tuple[int, ...] = ()
    ) -> graph_t:
        return self._add_connection(connection(p1, p2), unbatch_dims)

    def _add_connection(
        self, c: connection, unbatch_dims: tuple[int, ...] = ()
    ) -> graph_t:
        cs = batched.create_unbatch(c, unbatch_dims)
        return tree_at_(
            lambda me: me._connections,
            self,
            batched.concat([self._connections, cs.reshape(-1)]),
        )

    def get_point(self, pid: pointid) -> point:
        return self._points[pid._idx].unwrap()

    def get_lines(self) -> Array:
        # return accepted by Line3DCollection after tolist
        # which wants
        # list[tuple[list[float], list[float]]]

        def inner(c: connection):
            p1 = self.get_point(c.a).coords
            p2 = self.get_point(c.b).coords
            return jnp.stack([p1, p2])

        return self._connections.map(inner).unflatten()

    def sum_annotations(
        self, prev: batched[Array], annotations: batched[force_annotation]
    ) -> batched[Array]:
        n = len(self._points)
        assert prev.batch_dims() == (n,)
        assert len(annotations.batch_dims()) == 1

        anno_buf = annotations.unflatten()
        buf = prev.unflatten()

        return batched.create(buf.at[anno_buf.p._idx].add(anno_buf.f), batch_dims=(n,))

    def add_external_force(
        self, x: force_annotation, unbatch_dims: tuple[int, ...] = ()
    ):
        xs = batched.create_unbatch(x, unbatch_dims)
        return tree_at_(
            lambda me: me._external_forces,
            self,
            batched.concat([self._external_forces, xs.reshape(-1)]),
        )

    def forces_aggregate(
        self, connection_forces: batched[Array], density: fval
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

            force_weight = jnp.array([0.0, 0.0, -v_len * density / 2])

            ans1 = batched.create(force_annotation(c.a, force_vec_on_a + force_weight))
            ans2 = batched.create(force_annotation(c.b, -force_vec_on_a + force_weight))

            return batched.stack([ans1, ans2])

        ans = jax.vmap(inner)(self._connections, connection_forces)
        return ans.reshape(-1)

    def maybe_pin_points(self):
        return tree_at_(
            lambda me: me._points, self, self._points.map(lambda x: x.maybe_pin())
        )
