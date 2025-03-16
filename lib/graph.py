from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import sympy as s
from jax import Array, lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.typing import ArrayLike
from pintax import areg, convert_unit
from pintax._utils import pp_obj, pretty_print

from lib.batched import batched, do_batch, tree_do_batch, unbatch

from .utils import cast_unchecked, custom_vmap_, ival, tree_at_


class point(eqx.Module):
    coords: Array

    def __repr__(self):
        return pp_obj("point", pretty_print(self.coords)).format()


class pointid(eqx.Module):
    _idx: Array

    def __repr__(self):
        return pp_obj("pointid", pretty_print(self._idx)).format()


class connection(eqx.Module):
    a: pointid
    b: pointid

    def __repr__(self):
        return pp_obj("connection", pretty_print(self.a), pretty_print(self.b)).format()


class graph_t(eqx.Module):
    _points: batched[point]
    _connections: batched[connection]

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
            _points=batched.create(point(jnp.zeros(3))).repeat(0),
            _connections=batched.create(connection(_id, _id)).repeat(0),
        )

    def __add__(self, mod: Callable[[graph_t], graph_t]) -> graph_t:
        return mod(self)

    def add_point(
        self, c: Array, unbatch_dims: tuple[int, ...] = ()
    ) -> tuple[graph_t, pointid]:
        return self._add_point(point(c), unbatch_dims)

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
        new_points = batched.concat(self._points, points.reshape(-1))
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
            batched.concat(self._connections, cs.reshape(-1)),
        )

    def get_point(self, pid: pointid) -> point:
        return self._points[pid._idx].unwrap()

    def get_lines(self, mul: ArrayLike = 0.0):

        def inner(c_: batched[connection]):
            c = c_.unwrap()
            p1 = self.get_point(c.a).coords * mul
            p2 = self.get_point(c.b).coords * mul
            return p1, p2
            # ps = jnp.array([p1, p2])
            # return ps[:, 0], ps[:, 1], ps[:, 2]

        return jax.vmap(inner)(self._connections)

        # return p1_
        # xs, ys, zs = jax.vmap(inner)(self._connections)

        # return xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)
